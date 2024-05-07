import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_path(agent: str, env: str, extension: str):
    return f'logs/{agent}/{env}.{extension}'

def plot(agent: str, env: str):
    path_csv = get_path(agent, env, 'csv')
    data = pd.read_csv(path_csv)
    
    moving_avg_100 = data['reward'].rolling(window=100).mean()
    moving_avg_25 = data['reward'].rolling(window=25).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['episode'], data['reward'], alpha=0.3, color='lightblue', label='Raw Rewards')
    plt.plot(data['episode'], moving_avg_100, color='darkblue', label='Moving Average (100)')
    plt.plot(data['episode'], moving_avg_25, alpha=0.5, color='blue', label='Moving Average (25)')
    plt.title(f'Reward Dynamics for {agent} Agent in {env}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    save_path = get_path(agent, env, 'png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent in a gym environment')
    parser.add_argument('--agent', type=str, default='dqn', help='Specify the RL agent (dqn or ddqn)')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Specify the gym environment')
    args = parser.parse_args()

    plot(args.agent, args.env)
