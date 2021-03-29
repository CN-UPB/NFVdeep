import logging
import gym
import json
import numpy as np
from gym import spaces
from pathlib import Path
from tabulate import tabulate
from copy import deepcopy
from nfvdeep.environment.network import Network
from nfvdeep.environment.arrival import ArrivalProcess
from nfvdeep.environment.sfc import ServiceFunctionChain


class Env(gym.Env):
    def __init__(self, overlay_path, arrival_config):
        """A new Gym environment. This environment represents the environment of NFVdeep.

        Args:
          overlay_path: A connection to the network where the agent will act
          arrival_config: Dictionary that specifies the properties of the requests."""

        self.overlay_path = overlay_path
        self.arrival_config = arrival_config

        # define action and statespace
        _, properties = Network.check_overlay(self.overlay_path)
        num_nodes = properties['num_nodes']
        num_node_resources = properties['num_node_resources']

        # action `num_nodes` refers to volutarily rejecting the VNF embedding
        self.action_space = spaces.Discrete(num_nodes + 1)

        obs_dim = num_nodes * num_node_resources + num_node_resources + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float16)

        self.reward = 0

    def step(self, action):
        """Process the action of the agent.

        Args:
          action(int): The action is the number of the server where the current VNF will be embedded to. 
          Action `|nodes|` refers to voluntarily rejecting an embedding of the VNF.
        """

        assert(not self.done), 'Episode was already finished, but step() was called'

        info = {'accepted': False, 'rejected': False}
        logging.debug('(SFC, VNF): {} -> Node: ({})'.format(
            (self.sfc_idx, self.vnf_idx), action))

        # get the, to be processed, VNF for the current SFC
        sfc = self.request_batch[self.sfc_idx]
        vnf = sfc.vnfs[self.vnf_idx]

        '''
        Embed the VNF to the network. An embedding is invalid if either the VNF embedding or there do
        not remain sufficient resources to embed further VNFs of the current SFC / the SFC 
        will not satisfy its constraints (e.g. TTL constraint).
        '''
        is_valid_sfc = self.vnf_backtrack.embed_vnf(sfc, vnf, action)
        logging.debug('VNF embedding of {} to `vnf_backtrack` network on node: {} was {}.'.format(
            self.vnf_idx, action, is_valid_sfc))

        if is_valid_sfc:
            # check whether a valid SFC embedding remains possible after embedding the VNF, i.e.
            # if sufficient resources remain and the SFC constraints (e.g. latency) can still be satisfied
            is_valid_sfc = self.vnf_backtrack.check_embeddable(
                sfc, vnf_offset=self.vnf_idx + 1)

            logging.debug('SFC embedding (remains): {}.'.format(
                'possible' if is_valid_sfc else 'impossible'))

        '''
        Process the determined action. Backtrack to the latest network state if the action was invalid or progress 
        its state after a successful SFC embedding. 
        '''
        # determine whether the, to be processed, VNF is the last VNF of the current SFC
        is_last_of_sfc = self.vnf_idx >= len(sfc.vnfs) - 1

        # accept `vnf_backtrack` if the embedding (VNF and SFC) was valid and the SFC embedding is completed
        if is_valid_sfc and is_last_of_sfc:
            self.network = deepcopy(self.vnf_backtrack)
            self.sfc_idx, self.vnf_idx = (self.sfc_idx + 1, 0)

            # update info regarding successful embedding
            info['accepted'] = True
            info['placements'] = self.vnf_backtrack.sfc_embedding[sfc] 
            info['sfc'] = sfc

            logging.debug(
                'Updating network state to `vnf_backtrack` after completing SFC: {}'.format(self.sfc_idx))

        # invoke the agent for the next VNF of the current SFC
        elif is_valid_sfc and not is_last_of_sfc:
            self.sfc_idx, self.vnf_idx = (self.sfc_idx, self.vnf_idx + 1)

        # backtrack to the latest network state if the (VNF or SFC) embedding was invalid
        elif not is_valid_sfc:
            self.vnf_backtrack = deepcopy(self.network)

            # update info regarding unsuccessful embedding
            info['rejected'] = True
            info['placements'] = None 
            info['sfc'] = sfc

            self.sfc_idx, self.vnf_idx = (self.sfc_idx + 1, 0)

            logging.debug(
                'Backtracking `vnf_backtrack` to the latest network state')

        logging.debug('SFC, VNF indices set to: {}, {}'.format(
            self.sfc_idx, self.vnf_idx))

        '''
        Progress time such that the agent will only be invoked to embed SFC that are embeddable, i.e. for whom
        a valid sequence of VNF embeddings exist.
        '''
        # progress to next SFC within the request batch that can be embedded succsessfully
        # , i.e. progress within the intra timeslot
        batch_completed = self.progress_intra_timeslot()

        # determine the agent's reward before progressing time
        self.reward = self.compute_reward(
            sfc, is_last_of_sfc, is_valid_sfc, batch_completed)
        logging.debug(
            'Environment will attribute reward: {}'.format(self.reward))

        # progress inter timeslots until a `request_batch` with embeddable SFCs arrives
        if batch_completed:
            while True:
                self.done = self.progress_inter_timeslots()
                # progress to a SFC in the novel `request_batch` that is embeddable
                # if non exists continue to progress inter timeslots

                batch_completed = self.progress_intra_timeslot()
                if not batch_completed or self.done:
                    break

        # log the occupied resources and operation costs, log the number of operating nodes
        resource_utilization = self.vnf_backtrack.calculate_resource_utilization()
        resource_costs = self.vnf_backtrack.calculate_resource_costs()
        info.update({res + '_utilization': val for res,
                     val in resource_utilization.items()})
        info.update({res + '_cost': val for res,
                     val in resource_costs.items()})
        num_operating = len(self.vnf_backtrack.get_operating_servers())
        info.update({'operating_servers': num_operating})

        return self.compute_state(done=self.done), self.reward, self.done, info

    def reset(self):
        """Resets the environment to the default state"""

        self.done = False
        # initialize arrival process that will generate SFC requests
        self.arrival_process = ArrivalProcess.factory(self.arrival_config)

        # initialize network and backtracking network, s.t. `vnf_backtrack` is updated after each successful VNF embedding
        self.network = Network(self.overlay_path)
        self.vnf_backtrack = deepcopy(self.network)

        # track indices for current SFC & VNF
        self.sfc_idx, self.vnf_idx = 0, 0

        # current batch of SFC requests, i.e. list of SFC objects to be processed
        self.request_batch = []

        # progress time until first batch of requests arrives
        self.progress_inter_timeslots()

        return self.compute_state()

    def progress_intra_timeslot(self):
        """ Progress (SFC, VNF) indices to the next combination in the request batch that requires an 
        invokation of the agent. 
        """

        # progress the SFC indices to a SFC for whom an interaction with the agent is required,
        # i.e. there exists a valid embedding
        while self.sfc_idx < len(self.request_batch):
            sfc = self.request_batch[self.sfc_idx]
            is_embeddable = self.vnf_backtrack.check_embeddable(sfc)

            # the current SFC can be embedded, invoke the agent
            if is_embeddable:
                return False

            self.sfc_idx, self.vnf_idx = self.sfc_idx + 1, 0

        # no embeddable SFC remains in the batch, hence the indices are reset
        batch_completed = self.sfc_idx >= len(self.request_batch) - 1
        self.sfc_idx, self.vnf_idx = 0, 0
        return batch_completed

    def progress_inter_timeslots(self):
        """ Progress `request_batch` and `network` over inter timeslots, i.e. until novel requests arrive.
        """
        # empty the `request_batch` since we assume that all requests have been processed
        self.request_batch = []

        try:
            while len(self.request_batch) <= 0:
                self.network.update()
                self.request_batch = next(self.arrival_process)

                logging.debug('Progressing time, next SFC request batch: {}'.format(
                    self.request_batch))

            # initialize backtracking for the invoked intra timeslot
            self.vnf_backtrack = deepcopy(self.network)
            return False

        except StopIteration:
            # the `arrival_process` is done generating further requests, finish the episode
            return True

    def render(self, mode='human'):
        req_batch = str(self.request_batch)
        resources = [[num, *[str(avail_res) + ' out of ' + str(max_res) for avail_res, max_res in zip(res[0].values(), res[1].values())]]
                     for num, res in enumerate(zip(self.vnf_backtrack.calculate_resources(), self.vnf_backtrack.calculate_resources(False)))]

        sfcs = [[res.arrival_time, res.ttl, res.bandwidth_demand, res.max_response_latency, '\n'.join([str(vnf) for vnf in res.vnfs]), '\n'.join([str(node) for node in nodes])]
                for res, nodes in self.vnf_backtrack.sfc_embedding.items()]

        rep = str(tabulate(resources, headers=[
                  'Node', 'Cpu', 'Memory', 'Bandwidth'], tablefmt="presto"))
        rep += '\n \n \n'
        rep += 'Currently active Service Function Chains in the network:\n \n'
        rep += str(tabulate(sfcs, headers=['Arrival time', 'TTL', 'Bandwidth',
                                           'Max latency', 'VNFs (CPU, Memory)', ' Embedding Nodes'], tablefmt="grid"))
        rep += '\n \n'
        rep += 'Last reward = '+str(self.reward)+'\n'
        rep += 'Current request batch for (network) timestep ' + str(
            self.network.timestep)+': \n' + str(req_batch)
        rep += '\n \n \n'
        rep += '======================================================================================================================'
        rep += '\n'
        return rep

    def compute_state(self, done=False):
        """Compute the environment's state representation."""

        if (done == True):
            return np.asarray([])

        # compute remaining resources of backtrack network, i.e. (cpu, memory, bandwidth) for each node
        network_resources = self.vnf_backtrack.calculate_resources(
            remaining=True)
        network_resources = np.asarray([list(node_res.values())
                                        for node_res in network_resources], dtype=np.float16)

        # normalize network resources by their max capacity (over all nodes)
        max_resources = self.vnf_backtrack.calculate_resources(remaining=False)
        max_resources = np.asarray([list(node_res.values())
                                    for node_res in max_resources], dtype=np.float16)
        max_resources = np.max(max_resources, axis=0)
        network_resources = network_resources / max_resources
        # reshaping to 1d vector (flatten)
        network_resources = network_resources.reshape(-1)

        # compute (normalized) information regarding the VNF which is to be placed next
        sfc = self.request_batch[self.sfc_idx]
        vnf = sfc.vnfs[self.vnf_idx]

        norm_vnf_resources = np.asarray([*vnf, sfc.bandwidth_demand])
        norm_vnf_resources = list(norm_vnf_resources / max_resources)

        # TODO: parameterize normalization of constants such as TTL?
        norm_residual_latency = (sfc.max_response_latency -
                                 self.vnf_backtrack.calculate_current_latency(sfc)) / 1000
        norm_undeployed = (len(sfc.vnfs) - (self.vnf_idx + 1)) / 7
        norm_ttl = sfc.ttl / 1000

        observation = np.concatenate((network_resources,
                                      norm_vnf_resources, norm_residual_latency, norm_undeployed, norm_ttl), axis=None)

        return observation

    def compute_reward(self, sfc, last_in_sfc, sfc_valid, batch_completed):
        """Computes the reward signal dependent on whether a SFC has been succsessfully embedded
        Args:
          sfc: The current SFC
          last_in_sfc (bool): determines, if the current sfc is finished i.e. the last vnf was processed
          sfc_valid (bool): determines, if the current sfc was valid
          batch_completed (bool): determines, if the current batch of sfc is completed"""

        if not (last_in_sfc and sfc_valid):
            reward = 0
        else:
            reward = (sfc.ttl*sfc.bandwidth_demand) / 10

        if batch_completed:
            # compute total operation costs of the network first from individual costs of each resource
            costs = self.vnf_backtrack.calculate_resource_costs()
            costs = sum(costs.values())
            reward -= costs

        return reward