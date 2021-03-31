from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class MetricLoggingCallback(BaseCallback):
    """Custom callback that logs the agent's performance to TensorBoard."""

    def __init__(self, verbose=0):
        super(MetricLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        """Logs step information of custom NFVDeep environment to TensorBoard"""
        # get monitor object and log network metrics to TensorBoard
        monitor: NFVDeepMonitor = self.training_env.envs[0]
        num_requests = max(monitor.num_accepted + monitor.num_rejected, 1)
        self.logger.record('acceptance_ratio',
                           monitor.num_accepted / num_requests)
        self.logger.record('rejection_ratio',
                           monitor.num_rejected / num_requests)

        # log mean episode costs per unique resource type
        for key in monitor.resource_costs:
            self.logger.record('mean_{}'.format(
                key), monitor.resource_costs[key] / monitor.episode_length)

        # log total mean costs for the respective episode
        self.logger.record('mean_total_costs', sum(
            monitor.resource_costs.values()) / monitor.episode_length)

        # log the mean amount of occupied resources per resource type
        for key in monitor.resource_utilization:
            self.logger.record('mean_{}'.format(
                key), monitor.resource_utilization[key])

        # log the mean number of operating servers per step in an episode
        self.logger.record('mean_operating_servers',
                           monitor.operating_servers / monitor.episode_length)


class NFVDeepMonitor(Monitor):
    """Custom monitor tracking additional metrics that ensures compatability with StableBaselines."""

    def reset(self, **kwargs):
        """Augments the environment's monitor with network related metrics."""
        self.num_accepted = 0
        self.num_rejected = 0
        self.episode_length = 1
        self.resource_costs = {'cpu_cost': 0,
                               'memory_cost': 0,
                               'bandwidth_cost': 0}

        self.resource_utilization = {'cpu_utilization': 0,
                                     'memory_utilization': 0,
                                     'bandwidth_utilization': 0}

        self.operating_servers = 0

        self.placements = {}

        return super().reset(**kwargs)

    def step(self, action):
        """Extract the environment's information to the monitor."""
        observation, reward, done, info = super(
            NFVDeepMonitor, self).step(action)

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
