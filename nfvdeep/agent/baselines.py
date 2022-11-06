import time
import random
import torch as th
import numpy as np
import networkx as nx
import random
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean


class BaselineHeuristic(BaseAlgorithm):
    """Agent wrapper for implementing heuristic baselines that adhere to the StableBaselines interface"""

    def __init__(self, **kwargs):
        # set steps similar to PPO implementation
        self.n_steps = 2048

        # initialize BaseAlgorithm
        kwargs["learning_rate"] = 0.0
        kwargs["policy_base"] = None
        super(BaselineHeuristic, self).__init__(**kwargs)
        self._setup_model()

    def learn(self, **kwargs):
        pass

    def predict(
        self, observation, state=None, mask=None, deterministic=False, **kwargs
    ):
        scalar_tensor = self.policy._predict(
            observation, **{"deterministic": deterministic}
        )
        return scalar_tensor.numpy().reshape(-1), None

    def _setup_model(self):
        """Setup heuristic baseline policy with modified constructor"""
        self._setup_lr_schedule()
        self.policy = self.policy_class(
            self.env,
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )

    def collect_rollouts(self, env, callback, n_rollout_steps):
        """Rollout baseline policy to environment and collect performance in monitor wrapper"""
        n_steps = 0
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            action = self.policy._predict(self._last_obs)
            action = np.asarray([action])

            new_obs, rewards, dones, infos = env.step(action)

            if callback.on_step() is False:
                return False

            # update internal information buffer after each interaction
            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += 1
            self._last_obs = new_obs

        return True

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="run",
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=5,
        eval_log_path=None,
        reset_num_timesteps=True,
    ):

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                logger.record("time/fps", fps)
                logger.record(
                    "time/time_elapsed",
                    int(time.time() - self.start_time),
                    exclude="tensorboard",
                )
                logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


class BaselinePolicy(BasePolicy):
    """Stem for implementing baseline policies that adhere to the StableBaseline3 interface"""

    def __init__(
        self, env, observation_space, action_space, lr_schedule, device="auto", **kwargs
    ):
        super(BaselinePolicy, self).__init__(
            observation_space, action_space, device, **kwargs
        )
        # set internal NFVdeep environment assuming solely one execution environment (no parallel training)
        assert env.num_envs == 1, "Heuristic must be trained sequentially"
        self.env = next(iter(env.envs)).env

    def forward(self, obs, deterministic=False):
        pass


class RandomPolicy(BaselinePolicy):
    def _predict(self, observation: th.Tensor, **kwargs):
        # randomly sample action from the action space
        return th.scalar_tensor(self.env.action_space.sample(), dtype=th.int16)


class FirstFitPolicy(BaselinePolicy):
    """Agent that takes as action the first fitting node"""

    def _predict(self, state, **kwargs):
        """Chooses the first fitting node as the action."""
        current_sfc = self.env.request_batch[self.env.sfc_idx]
        current_vnf = current_sfc.vnfs[self.env.vnf_idx]

        for node in range(self.env.vnf_backtrack.num_nodes):
            if self.env.vnf_backtrack.check_vnf_resources(
                current_vnf, current_sfc.bandwidth_demand, node
            ):
                return th.scalar_tensor(node, dtype=th.int16)

        return self.env.vnf_backtrack.num_nodes


class FirstFitPolicy2(FirstFitPolicy):
    """Agent that takes as action the first fitting node, but rejects sfc that
    have high costs compared to the revenue."""

    def _predict(self, state, factor=1, **kwargs):
        current_sfc = self.env.request_batch[self.env.sfc_idx]
        current_vnf = current_sfc.vnfs[self.env.vnf_idx]
        positive_reward = current_sfc.bandwidth_demand
        costs = self.env.vnf_backtrack.costs
        negative_reward = sum(
            [
                vnf[0] * costs["cpu"] + vnf[1] * costs["memory"]
                for vnf in current_sfc.vnfs
            ]
        )

        if positive_reward < factor * negative_reward:
            # reject embedding of VNF
            return th.scalar_tensor(self.env.vnf_backtrack.num_nodes, dtype=th.int16)

        return super()._predict(state, **kwargs)


class FirstFitPolicy3(BaselinePolicy):
    """Agent that takes as action the first fitting node which is
    close to the previous embedded node in the SFC."""

    def _predict(self, state, **kwargs):
        current_sfc = self.env.request_batch[self.env.sfc_idx]
        current_vnf = current_sfc.vnfs[self.env.vnf_idx]

        if self.env.vnf_idx == 0:
            source = random.randint(0, self.env.vnf_backtrack.num_nodes - 1)
            last_node = None

        else:
            source = self.env.vnf_backtrack.sfc_embedding[current_sfc][
                self.env.vnf_idx - 1
            ]
            last_node = source

        path_length = nx.shortest_path_length(
            self.env.vnf_backtrack.overlay, source=source, weight="latency"
        )
        sorted_nodes = [
            node for node, _ in sorted(path_length.items(), key=lambda item: item[1])
        ]

        for node in sorted_nodes:
            if self.env.vnf_backtrack.check_vnf_resources(
                current_vnf, current_sfc.bandwidth_demand, node
            ):
                if not node == last_node:
                    # check if the bandwidth constraint holds
                    remaining_bandwidth = self.env.vnf_backtrack.calculate_resources()[
                        node
                    ]["bandwidth"]
                    if remaining_bandwidth - current_sfc.bandwidth_demand < 0:
                        continue
                return th.scalar_tensor(node, dtype=th.int16)

        return th.scalar_tensor(self.env.vnf_backtrack.num_nodes, dtype=th.int16)


class FirstFitPolicy4(FirstFitPolicy3):
    """Agent that takes as action the first fitting node which is close to the previous embedded
    node , but rejects sfc that have high costs compared to the revenue."""

    def _predict(self, state, factor=1, **kwargs):
        current_sfc = self.env.request_batch[self.env.sfc_idx]
        positive_reward = current_sfc.bandwidth_demand
        costs = self.env.vnf_backtrack.costs
        negative_reward = sum(
            [
                vnf[0] * costs["cpu"] + vnf[1] * costs["memory"]
                for vnf in current_sfc.vnfs
            ]
        )

        if positive_reward < factor * negative_reward:
            return th.scalar_tensor(self.env.vnf_backtrack.num_nodes, dtype=th.int16)

        return super()._predict(state, **kwargs)
