from typing import List, Optional, Tuple
import gym

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class MetricLoggingCallback(BaseCallback):
    """Custom callback that logs the agent's performance to TensorBoard."""

    def __init__(self, verbose=0):
        super(MetricLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        """Logs step information of custom NFVDeep environment to TensorBoard"""

        if not np.any(self.locals["dones"]):
            return True

        # get monitor object and log network metrics to TensorBoard
        monitors: List[NFVDeepMonitor] = [env for env, done in zip(self.training_env.envs, self.locals["dones"]) if done]

        num_requests = [max(monitor.num_accepted + monitor.num_rejected, 1) for monitor in monitors]
        acceptance = [monitor.num_accepted / nrequests for monitor, nrequests in zip(monitors, num_requests)]
        rejection = [monitor.num_rejected / nrequests for monitor, nrequests in zip(monitors, num_requests)]

        self.logger.record('acceptance_ratio', np.mean(acceptance))
        self.logger.record('rejection_ratio', np.mean(rejection))
        
        costs = [{key: monitor.resource_costs[key] / monitor.episode_length for key in monitor.resource_costs} for monitor in monitors]
        costs = {key: np.mean([dic[key] for dic in costs]) for key in costs[0]}

        # log mean episode costs per unique resource type
        for key, value in costs.items():
            self.logger.record('mean_{}'.format(key), value)

        # log total mean costs for the respective episode
        total = np.mean([sum(monitor.resource_costs.values()) / monitor.episode_length for monitor in monitors])
        self.logger.record('mean_total_costs', total) 

        # log the mean amount of occupied resources per resource type
        occupied = [{key: monitor.resource_utilization[key] for key in monitor.resource_utilization} for monitor in monitors]
        occupied = {key: np.mean([dic[key] for dic in occupied]) for key in occupied[0]}

        for key, value in occupied.items():
            self.logger.record('mean_{}'.format(key), value)

        # log the mean number of operating servers per step in an episode
        operating =  np.mean([monitor.operating_servers / monitor.episode_length for monitor in monitors])
        self.logger.record('mean_operating_servers', operating)


class NFVDeepMonitor(Monitor):
    """Custom monitor tracking additional metrics that ensures compatability with StableBaselines."""

    def __init__(self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self._last_step_reset = True
        self._reset()

    def _reset(self, **kwargs):
        """Augments the environment's monitor with network related metrics."""
        
        self.num_accepted = 0
        self.num_rejected = 0
        self.episode_length = 1
        self.operating_servers = 0

        self.placements = {}

        self.resource_costs = {'cpu_cost': 0,
                               'memory_cost': 0,
                               'bandwidth_cost': 0}

        self.resource_utilization = {'cpu_utilization': 0,
                                     'memory_utilization': 0,
                                     'bandwidth_utilization': 0}


    def step(self, action):
        """Extract the environment's information to the monitor."""

        observation, reward, done, info = super(
            NFVDeepMonitor, self).step(action)

        if self._last_step_reset:
            self._reset()

        self._last_step_reset = done

        # add sfc related information to the monitor
        self.num_accepted += info['accepted']
        self.num_rejected += info['rejected']

        if info['accepted'] or info['rejected']:
            sfc = info['sfc']
            self.placements[sfc] = info['placements']

        # add resource cost related information to the monitor
        for key in self.resource_costs:
            self.resource_costs[key] += info[key]

        # log moving average for resource utilization
        for key in self.resource_utilization:
            prev = self.resource_utilization[key]
            self.resource_utilization[key] = (
                (self.episode_length - 1) / self.episode_length) * prev
            self.resource_utilization[key] += (1 /
                                               self.episode_length) * info[key]

        # add number of operating_servers servers to the monitor
        self.operating_servers += info['operating_servers']
        self.episode_length += 1

        return observation, reward, done, info
