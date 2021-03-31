import functools
import operator
import networkx as nx
from collections import Counter
from networkx.exception import NetworkXNoPath
from environment.sfc import ServiceFunctionChain


class Network:
    def __init__(self, overlay, costs={'cpu': 0.2, 'memory': 0.2, 'bandwidth': 0.006}):
        """Internal representation of the network & embedded VNFs.

        Args:
          overlay (str or networkx graph): Path to an overlay graph that specifies the network's properties
                or a valid networkx graph.
          costs (tuple of float): The cost for one unit of cpu, memory and bandwidth"""

        # parse and validate the overlay network
        self.overlay, properties = Network.check_overlay(overlay)
        self.num_nodes = properties['num_nodes']

        self.timestep = 0
        self.costs = costs

        # the service function chains with its mapping to the network nodes:
        self.sfc_embedding = dict()

    def update(self, time_increment=1):
        """Let the network run for the given time"""

        # TODO: increment timestep before / after updating SFCs?
        self.timestep += time_increment

        # delete all SFCs that exceed their TTL
        def check_ttl(sfc): return sfc.arrival_time + sfc.ttl >= self.timestep
        sfc_embedding = {sfc: nodes for sfc,
                         nodes in self.sfc_embedding.items() if check_ttl(sfc)}

        self.sfc_embedding = sfc_embedding

    def embed_vnf(self, sfc: ServiceFunctionChain, vnf: tuple, node: int):
        """Embeds the Virtual Network Function to a specified node."""

        # reject an embedding if the network does not provide sufficient VNF resources or the action
        # voluntarily chooses no embedding
        if node >= self.num_nodes or not self.check_vnf_resources(vnf, sfc, node):
            return False

        if sfc not in self.sfc_embedding:
            self.sfc_embedding[sfc] = []

        self.sfc_embedding[sfc].append(node)

        return True

    def calculate_resources(self, remaining=True) -> list:
        """Calculates the remaining resources for all nodes.

        Returns:
            List of dictionaries s.t. the i-th entry is a representation of the remaining resources of the i-th node
        """
        resources = [{res: max_val for res, max_val in res.items()}
                     for _, res in self.overlay.nodes(data=True)]

        if remaining:
            # calculated remaining resources
            for sfc, nodes in self.sfc_embedding.items():
                for vnf_idx, node_idx in enumerate(nodes):
                    resources[node_idx]['cpu'] -= sfc.vnfs[vnf_idx][0]
                    resources[node_idx]['memory'] -= sfc.vnfs[vnf_idx][1]

                    # bandwidth is demanded if successive VNFs are embedded on different servers, i.e.
                    # no bandwidth is required if the VNF is placed on the same server as the previous VNF,
                    # unless for the last VNF of the chain, for whom we always demand bandwidth
                    if vnf_idx == len(nodes) - 1:
                        resources[node_idx]['bandwidth'] -= sfc.bandwidth_demand

                    elif not nodes[vnf_idx] == nodes[vnf_idx+1]:
                        resources[node_idx]['bandwidth'] -= sfc.bandwidth_demand

        return resources

    def check_sfc_constraints(self, sfc):
        """ Check whether the (partial) SFC embedding satisfies the bandwidth and latency constraints, i.e.
        if the bandwidth demand can be satisfied by ANY node and if the SFC can still satisfy its latency constraints.
        """

        # check if there exists a node which can cover the request's bandwidth demand
        resources = self.calculate_resources()
        max_available_bandwidth = max([res['bandwidth'] for res in resources])
        bandwidth_constraint = sfc.bandwidth_demand <= max_available_bandwidth

        # check if the current bandwidth constrains hold
        if not all([node['bandwidth'] >= 0 for node in resources]):
            return False

        # distinguish whether SFC is already (partially) embedded or a novel request
        if not sfc in self.sfc_embedding:
            # solely verify bandwidth constraint for novel requests
            return bandwidth_constraint

        # if there is a next VNF and it can be embedded to the same node as the previous VNF,
        # no additional bandwidth is demanded
        elif not bandwidth_constraint and len(self.sfc_embedding[sfc]) < len(sfc.vnfs):
            last_node = self.sfc_embedding[sfc][-1]
            next_vnf = sfc.vnfs[len(self.sfc_embedding[sfc])]
            bandwidth_constraint = self.check_vnf_resources(
                next_vnf, sfc, last_node)

        # also verify the latency constraints for a (partially) embedded SFC, i.e.
        # does the latency of prior VNF embeddings violate the maximum latency of the request?
        try:
            latency = self.calculate_current_latency(sfc)
            latency_constraint = latency <= sfc.max_response_latency
        except NetworkXNoPath:
            latency_constraint = False

        return bandwidth_constraint and latency_constraint

    def calculate_current_latency(self, sfc):
        """Calculates the current latency of the SFC i.e the end to end delay from the start of the SFC to the currently
            last embedded VNF of the SFC

           Throws NetworkXNoPath, if there is no possible path between two VNFs"""

        latency = 0

        # compute transmission and processing delay if the SFC is already (partially) embedded
        if sfc in self.sfc_embedding:
            nodes = self.sfc_embedding[sfc]
            latency = sum([nx.dijkstra_path_length(self.overlay, nodes[idx-1],
                                                   nodes[idx], weight='latency') for idx in range(1, len(nodes))])

            latency += sum(sfc.processing_delays[:len(nodes)])

        return latency

    def check_embeddable(self, sfc: ServiceFunctionChain, vnf_offset=0):
        """ Check whether the (partial) SFC embedding can still satisfy its constraints, i.e. check if for all remaining VNFs
        some node with sufficient resources exists and if the SFC constraints can still be satisfied.
        """
        vnfs = sfc.vnfs[vnf_offset:]
        # check whether remaining resources are sufficient to embed further VNFs
        # TODO: we check only if there exists a node that provides sufficient resources for the next VNF
        # TODO: the paper does not specifiy how exactly this test is done, i.e. how to specify
        # that the agent will only be invoked if a valid embedding exists
        vnf_constraints = len(vnfs) <= 0
        if not vnf_constraints:
            next_vnf = next(iter(vnfs))
            vnf_constraints = self.check_vnf_resources(
                next_vnf, sfc)

        # check whether SFC can still fullfill service constraints (SFC constraints)
        sfc_constraints = self.check_sfc_constraints(sfc)

        return vnf_constraints and sfc_constraints

    def check_vnf_resources(self, vnf, sfc, node=None):
        """ Check whether some node exists with sufficient resources or if the specified node provides enough resources. """

        resources = self.calculate_resources()
        if node is not None:
            resources = [resources[node]]

        def constraints(res, vnf): return all(
            [res['cpu'] >= vnf[0], res['memory'] >= vnf[1]])
        nodes = set(num for num, res in enumerate(
            resources) if constraints(res, vnf))

        return bool(nodes)

    def calculate_occupied_resources(self):
        """Calculates a dictionary that summarizes the amount of occupied resources per type."""

        # compute the amount of maximum resources / available resources per node
        resources = self.calculate_resources(remaining=False)
        avail_resources = self.calculate_resources(remaining=True)

        # reduce the respective resources over the entire network (i.e. summed over all nodes)
        resources = dict(functools.reduce(operator.add,
                                          map(Counter, resources)))
        avail_resources = dict(functools.reduce(operator.add,
                                                map(Counter, avail_resources)))

        # get the amount of depleted resources and their costs
        depleted = {key: resources[key] -
                    avail_resources[key] for key in resources}
        costs = {key: depleted[key]
                 * self.costs[key] for key in depleted}

        return costs

    def calculate_resource_utilization(self):
        """Calculates a dictionary that summarizes the resource utilization per resource type."""

        # compute the amount of maximum resources / available resources per node
        max_resources = self.calculate_resources(remaining=False)
        avail_resources = self.calculate_resources(remaining=True)

        max_resources = dict(functools.reduce(operator.add,
                                              map(Counter, max_resources)))
        avail_resources = dict(functools.reduce(operator.add,
                                                map(Counter, avail_resources)))

        # compute resource utilization per type as a dictionary
        utilization = {key: (
            max_resources[key] - avail_resources[key]) / max_resources[key] for key in max_resources}

        return utilization

    def get_operating_servers(self):
        """Computes the set of indices that describe all operating servers."""
        operating_servers = {
            server for sfc in self.sfc_embedding for server in self.sfc_embedding[sfc]}
        return operating_servers

    def calculate_resource_costs(self):
        """Computes a dictionary that summarizes the current operation costs per resource type."""
        # get a set of currently operating nodes
        operating_servers = self.get_operating_servers()

        if not operating_servers:
            return {key: 0 for key in self.costs}

        # filter the maximum available resources with respect to the operating servers
        resources = [res for idx, res in enumerate(
            self.calculate_resources(remaining=False)) if idx in operating_servers]

        # reduce the amount of resources per type
        resources = dict(functools.reduce(operator.add,
                                          map(Counter, resources)))

        # calculate the cost per resource type
        cost = {res: resources[res]
                * self.costs[res] for res in resources}

        return cost

    @staticmethod
    def check_overlay(overlay):
        """Checks whether the overlay adhers to the expected parameter types and attributes, returns the parsed network instance."""

        # parse overlay from gpickle if
        if isinstance(overlay, str) and overlay.endswith('.gpickle'):
            overlay = nx.read_gpickle(overlay)

        node_attributes = {'cpu': int, 'memory': float, 'bandwidth': float}
        for _, data in overlay.nodes(data=True):
            assert(all([nattr in data for nattr, ntype in node_attributes.items(
            )])), 'Overlay must specify all required node attributes.'
            assert(all([type(data[nattr]) == ntype for nattr, ntype in node_attributes.items(
            )])), 'Overlay must specify the correct data types.'

        edge_attributes = {'latency': float}
        for _, _, data in overlay.edges(data=True):
            assert(all([eattr in data for eattr, etype in edge_attributes.items(
            )])), 'Overlay must specify all required edge attributes.'
            assert(all([type(data[eattr]) == etype for eattr, etype in edge_attributes.items(
            )])), 'Overlay must specify the correct data types.'

        # compute properties of the parsed graph
        properties = {}
        properties['num_nodes'] = overlay.number_of_nodes()
        _, resource = next(iter(overlay.nodes(data=True)))
        properties['num_node_resources'] = len(resource)

        return overlay, properties