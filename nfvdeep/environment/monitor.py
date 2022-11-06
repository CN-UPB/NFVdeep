from pathlib import Path
from typing import List
from collections import defaultdict

import gym
import numpy as np
from tabulate import tabulate
from stable_baselines3.common.callbacks import BaseCallback


class StatsWrapper(gym.Wrapper):
    STATS = ["accepted", "rejected", "operating_servers"]
    COSTS = ["cpu_cost", "memory_cost", "bandwidth_cost"]
    UTILIZATIONS = ["cpu_utilization", "memory_utilization", "bandwidth_utilization"]

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.placements = {}
        self.statistics = defaultdict(float)

    def clear(self):
        self.placements = {}
        self.statistics = defaultdict(float)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        keys = [
            key for key in info if key in self.STATS + self.COSTS + self.UTILIZATIONS
        ]
        for key in keys:
            self.statistics[key] += info[key]

        self.statistics["ep_length"] += 1

        if "sfc" in info:
            self.placements[info["sfc"]] = info["placements"]

        return obs, reward, done, info


class EvalLogCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = Path(log_path) / "placements.txt"
        self.niter = 0

    def _on_step(self):

        eval_envs: List[StatsWrapper] = self.locals["callback"].eval_env.envs
        assert all([isinstance(env, StatsWrapper) for env in eval_envs])

        num_requests = [
            max(env.statistics["accepted"] + env.statistics["rejected"], 1)
            for env in eval_envs
        ]
        acceptance = [
            env.statistics["accepted"] / nreqs
            for env, nreqs in zip(eval_envs, num_requests)
        ]
        rejection = [
            env.statistics["rejected"] / nreqs
            for env, nreqs in zip(eval_envs, num_requests)
        ]

        self.logger.record("eval/acceptance_ratio", np.mean(acceptance))
        self.logger.record("eval/rejection_ratio", np.mean(rejection))

        costs = [
            {
                key: env.statistics[key] / env.statistics["ep_length"]
                for key in env.COSTS
            }
            for env in eval_envs
        ]
        costs = {key: np.mean([dic[key] for dic in costs]) for key in costs[0]}

        for key, value in costs.items():
            self.logger.record("eval/mean_{}".format(key), value)

        occupied = [
            {key: env.statistics[key] for key in env.UTILIZATIONS} for env in eval_envs
        ]
        occupied = {key: np.mean([dic[key] for dic in occupied]) for key in occupied[0]}

        for key, value in occupied.items():
            self.logger.record("eval/mean_{}".format(key), value)

        operating = np.mean(
            [
                env.statistics["operating_servers"] / env.statistics["ep_length"]
                for env in eval_envs
            ]
        )
        self.logger.record("eval/mean_operating_servers", operating)

        placements = [env.placements for env in eval_envs]
        self.save_placements(placements)

        for env in eval_envs:
            env.clear()

        self.niter += 1

    def save_placements(self, placements: List) -> None:
        table = []
        for ep, episode in enumerate(placements):
            for sfc, placement in episode.items():
                row = [
                    self.niter,
                    ep,
                    sfc.arrival_time,
                    sfc.ttl,
                    sfc.bandwidth_demand,
                    sfc.max_response_latency,
                    sfc.vnfs,
                    placement,
                ]
                table.append(row)

        headers = [
            "Eval. Iter",
            "Episode",
            "Arrival",
            "TTL",
            "Bandwidth",
            "Max Latency",
            "VNFs",
            "Placements",
        ]
        table = "\n\n" + tabulate(table, headers=headers)

        with open(str(self.log_path), "a") as file:
            file.write(table)
