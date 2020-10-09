import logging
import gym
import numpy as np
from copy import deepcopy
from typing import Union
from ray.tune import report
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback


class OptimizationCallback(BaseCallback):

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 n_eval_episodes: int = 5,
                 deterministic: bool = True,
                 verbose=0):
        super(OptimizationCallback, self).__init__(verbose)
        self.eval_env = deepcopy(eval_env)
        self.eval_env.reset()
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self):
        sync_envs_normalization(self.training_env, self.eval_env)

        episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                           n_eval_episodes=self.n_eval_episodes,
                                                           render=False,
                                                           deterministic=self.deterministic,
                                                           return_episode_rewards=True)

        episode_reward_mean, std_reward = np.mean(
            episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(
            episode_lengths), np.std(episode_lengths)

        report(
            episode_reward_mean=episode_reward_mean,
            std_reward=std_reward,
            mean_ep_length=mean_ep_length,
            std_ep_length=std_ep_length
        )
