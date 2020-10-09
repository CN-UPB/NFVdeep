import os
import argparse
import logging
import ray
import json
from pathlib import Path
from copy import deepcopy
from evaluation import evaluate_final_policy, safe_experiment
from nfvdeep.environment.env import Env
from nfvdeep.environment.arrival import *
from nfvdeep.agent.baselines import *
from nfvdeep.agent.logging import MetricLoggingCallback, NFVDeepMonitor
from nfvdeep.tuning import OptimizationCallback
from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EveryNTimesteps


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
                        help='Either path to request file or key word for stochastic arrival process')
    parser.add_argument('--agent', type=str,
                        help='Whether to use a RL agent or a baseline')
    parser.add_argument('--logs', type=str,  nargs='?', const=1,
                        default=r'./logs', help='Path of tensorboard logs')

    # arguments to specify ray's hyperparameter optimization procedure
    parser.add_argument('--sample_timesteps', type=int, nargs='?', const=1, default=200000,
                        help='Number of timesteps used to train intermediate configurations')
    parser.add_argument('--report_interval', type=int, nargs='?', const=1, default=10000,
                        help='Interval between reportings from callback (in timesteps)')
    parser.add_argument('--ray_eval_episodes', type=int, nargs='?', const=1, default=1,
                        help='Maximum number of episodes for final (deterministic) evaluation')
    parser.add_argument('--ray_tune_samples', type=int, nargs='?', const=1,
                        default=128, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--ray_cpus', type=int, nargs='?', const=1, default=16,
                        help='Number of cpus ray tune will use for the optimization')

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

    # Create agent from experiment configuration
    if args.agent == 'Random':
        agent = BaselineHeuristic
        policy = RandomPolicy

    elif args.agent == 'FirstFit_1':
        agent = BaselineHeuristic
        policy = FirstFitPolicy

    elif args.agent == 'FirstFit_2':
        agent = BaselineHeuristic
        policy = FirstFitPolicy2

    elif args.agent == 'FirstFit_3':
        agent = BaselineHeuristic
        policy = FirstFitPolicy3

    elif args.agent == 'FirstFit_4':
        agent = BaselineHeuristic
        policy = FirstFitPolicy4

    elif args.agent == 'A2C':
        agent = A2C
        policy = 'MlpPolicy'

    elif args.agent == 'PPO':
        agent = PPO
        policy = 'MlpPolicy'

    elif args.agent == 'DQN':
        agent = DQN
        policy = 'MlpPolicy'

    else:
        raise ValueError('An unknown agent was specified')

    EVAL_EPISODES = args.ray_eval_episodes
    TOTAL_TIMESTEPS = args.total_train_timesteps
    RAY_TUNE_SAMPLES = args.ray_tune_samples

    # load parameter optimization space from file
    with open('./nfvdeep/spaces/{}_space.json'.format(args.agent), 'r') as search_space:
        parameters = json.load(search_space)

    # modifiy name for experiment generation
    args.agent = '(tuned) ' + args.agent
    results = dict()

    # load the arrival processe's properties 
    with open(Path(args.requests), 'r') as file:
        arrival_config = json.load(file)

    for trial in range(args.trials):
        # create the network's overlay structure & incoming requests for the environment
        arrival_config['seed'] = trial
        base_env = Env(args.overlay, arrival_config)

        # Define objective function for hyperparameter tuning
        def evaluate_objective(config):
            tune_env = deepcopy(base_env)
            tune_monitor = OptimizationCallback(tune_env, EVAL_EPISODES, True)
            monitor_callback = EveryNTimesteps(
                n_steps=args.report_interval, callback=tune_monitor)

            tune_agent = agent("MlpPolicy", tune_env, **config)
            tune_agent.learn(total_timesteps=args.sample_timesteps,
                             callback=monitor_callback)

        ax_client = AxClient(enforce_sequential_optimization=False)

        ax_client.create_experiment(
            name="tune_RL",
            parameters=parameters,
            objective_name='episode_reward_mean',
            minimize=False,
            overwrite_existing_experiment=True
        )

        # add scheduling of configurations, i.e. intensify solely
        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='episode_reward_mean', mode='max')

        ray.init(num_cpus=args.ray_cpus)
        ray.tune.run(
            evaluate_objective,
            num_samples=RAY_TUNE_SAMPLES,
            search_alg=AxSearch(ax_client),
            scheduler=asha_scheduler,
            verbose=2
        )

        # get best parameters, retrain agent and log results for best agent
        best_parameters, values = ax_client.get_best_parameters()

        ray.shutdown()

        env = NFVDeepMonitor(base_env, args.logs)
        callback = MetricLoggingCallback()
        eval_agent = agent(**{'policy': policy, 'env': env, 'verbose': 1,
                              'tensorboard_log': args.logs, **best_parameters})

        tb_log_name = eval_agent.__class__.__name__ if isinstance(
            policy, str) else policy.__name__
        eval_agent.learn(total_timesteps=args.total_train_timesteps,
                         tb_log_name=tb_log_name, callback=callback)

        # evaluate final policy and log performances
        results[trial] = evaluate_final_policy(
            args.eval_episodes, eval_agent, env)

    # save experiments to disk at specified output path
    safe_experiment(results, vars(args))
