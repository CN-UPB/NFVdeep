
import os
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_final_policy(n_eval_episodes, agent, env):
    """ Evaluates the final policy of an agent for the specified experiment.
    Args:
      n_eval_episodes (int): number of evaluation episodes
      agent: The trained agent
      env: The environment the agent acts in
    """

    episode_results = dict()
    for episode_num in range(n_eval_episodes):
        episode = dict()

        rew, _ = evaluate_policy(
            agent, env, n_eval_episodes=1, return_episode_rewards=True)

        episode['reward'] = rew[0]
        episode['acceptance_rate'] = env.num_accepted / \
            (env.num_accepted + env.num_rejected)

        for key in env.resource_utilization:
            episode['mean_{}'.format(key)] = env.resource_utilization[key]

        episode['mean_operating_servers'] = env.operating_servers / \
            env.episode_length

        episode_results[episode_num] = episode

    return episode_results


def safe_experiment(results, args):
    """ Safes one experiment together with its metainformation into a csv file."""

    if not os.path.exists(args['output']):
        os.makedirs(args['output'])

    # safe environemt & hyperparameters used to generate the results
    with open(Path(args['output']) / 'args.json', 'a') as file:
        file.write(json.dumps(args))
        file.write("\n")

    # safe agent's performances in csv format
    data = {(args['agent'], i, j): results[i][j] for i in results.keys()
            for j in results[i].keys()}
    data = pd.DataFrame.from_dict(data, orient='index')
    data.index.names = ['agent', 'trial', 'episode']

    results_path = Path(args['output']) / 'results.csv'

    if not os.path.exists(results_path):
        data.to_csv(results_path)

    else:
        data.to_csv(results_path, mode='a', header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    index_mapping = {'agent': 'Agent', 'trial': 'Trial', 'episode': 'Episode'}

    measure_mapping = {'reward': 'Reward', 'acceptance_rate': 'Acceptance Rate',
                       'mean_cpu_utilization': 'CPU Utilization',
                       'mean_memory_utilization': 'Memory Utilization',
                       'mean_bandwidth_utilization': 'Bandwidth Utilization',
                       'mean_operating_servers': 'Operating Servers'}

    results = pd.DataFrame()
    for table in ['firstfit.csv', 'default_ppo.csv', 'tune.csv']:
        table = pd.read_csv(Path(args.results) / table)
        results = pd.concat((results, table))
        
    results = results.rename(columns={**index_mapping, **measure_mapping})
    results = results.replace('FirstFit_3', 'FirstFit')
    results = results.groupby(['Agent', 'Trial']).mean()
    results = results.reset_index()

    sns.set_style("whitegrid")
    for measure in measure_mapping.values():
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x='Agent', y=measure, data=results, ax=ax)
        sns.despine()
        fig.savefig(Path(args.output) / f'{measure}.svg')

