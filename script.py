import os
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DQN
from evaluation import evaluate_final_policy, safe_experiment
from nfvdeep.environment.arrival import StochasticProcess
from nfvdeep.environment.env import Env
from nfvdeep.environment.arrival import *
from nfvdeep.agent.baselines import *
from nfvdeep.agent.logging import MetricLoggingCallback, NFVDeepMonitor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # arguments to specify parameters of the experiment evaluation
    parser.add_argument('--total_train_timesteps', type=int,  nargs='?',
                        const=1, default=1000000, help='Number of training steps for the agent')
    parser.add_argument('--debug', action='store_false',
                        help='Whether to enable debugging logs of the environment')
    parser.add_argument('--overlay', type=str,
                        help='Path to overlay graph for the environment')
    parser.add_argument('--requests', type=str,
                        help='Either path to request file or config of stochastic arrival process')
    parser.add_argument('--agent', type=str,
                        help='Whether to use a RL agent or a baseline')
    parser.add_argument('--logs', type=str,  nargs='?', const=1,
                        default=r'./logs', help='Path of tensorboard logs')

    # arguments to specify the final policy's evaluation
    parser.add_argument('--eval_episodes', type=int,
                        default=20, help='Number of evaluation steps for one trained agent')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials evaluating the agent')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the folder where all results will be stored at')

    args = parser.parse_args()

    # set logging level according to --debug
    logging.basicConfig()
    debug_level = logging.INFO if args.debug else logging.DEBUG
    logging.getLogger().setLevel(debug_level)

    # Create log dir & monitor training so that episode rewards are logged
    os.makedirs(args.logs, exist_ok=True)
    agent_name = args.agent

    # Create agent from experiment configuration
    if agent_name == 'Random':
        agent_type = BaselineHeuristic
        policy = RandomPolicy

    elif agent_name == 'FirstFit_1':
        agent_type = BaselineHeuristic
        policy = FirstFitPolicy

    elif agent_name == 'FirstFit_2':
        agent_type = BaselineHeuristic
        policy = FirstFitPolicy2

    elif agent_name == 'FirstFit_3':
        agent_type = BaselineHeuristic
        policy = FirstFitPolicy3

    elif agent_name == 'FirstFit_4':
        agent_type = BaselineHeuristic
        policy = FirstFitPolicy4

    elif agent_name == 'A2C':
        agent_type = A2C
        policy = 'MlpPolicy'

    elif agent_name == 'PPO':
        agent_type = PPO
        policy = 'MlpPolicy'

    elif agent_name == 'DQN':
        agent_type = DQN
        policy = 'MlpPolicy'

    else:
        raise ValueError('An unknown agent was specified')

    # load the arrival processe's properties 
    with open(Path(args.requests), 'r') as file:
        arrival_config = json.load(file)

    results = dict()
    for trial in range(args.trials):
        # create the network's overlay structure & incoming requests for the environment
        arrival_config['seed'] = trial
        env = Env(args.overlay, arrival_config)
        env = NFVDeepMonitor(env, args.logs)

        callback = MetricLoggingCallback()
        agent = agent_type(**{'policy': policy, 'env': env,
                              'verbose': 1, 'tensorboard_log': args.logs})

        tb_log_name = agent.__class__.__name__ if isinstance(
            policy, str) else policy.__name__

        #if policy == 'MlpPolicy':
            # solely MLP policies require traning
        agent.learn(total_timesteps=args.total_train_timesteps,
                    tb_log_name=tb_log_name, callback=callback)

        # evaluate final policy and log performances
        results[trial] = evaluate_final_policy(args.eval_episodes, agent, env)

    # save experiments to disk at specified output path
    safe_experiment(results, vars(args))
