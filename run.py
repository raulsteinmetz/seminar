import yaml
import argparse
import os
import csv

import gym
import torch
import numpy as np

from util.buffer import ReplayBuffer
from util.tools import set_all_seeds
from util.q_net import learn_ddqn_mlp, learn_dqn_mlp, make_q_nets, save_models, load_models

configs = {}

import numpy as np

def main(configs):
    set_all_seeds(configs['seed'])

    env = gym.make(configs['env'])

    q_net, target_q_net = make_q_nets(env, configs['units'], configs['layers'])

    if configs['agent'] == 'dqn':
        learn = learn_dqn_mlp
    elif configs['agent'] == 'ddqn':
        learn = learn_ddqn_mlp
        
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), configs['learning_rate'])

    buffer = ReplayBuffer(configs['replay_buffer_capacity'], env.observation_space.shape[0], 1)

    log_folder = f"logs/{configs['agent']}"
    os.makedirs(log_folder, exist_ok=True)
    op = 'test' if configs['test'] else 'train'
    log_file_path = f"{log_folder}/{configs['env']}_{op}.csv"

    rewards = []
    save_threshold = 100
    best_avg_reward = -float('inf')

    if configs['test'] == True:
        configs['warm_up_episodes'] = 0
        load_models(f"{log_folder}/{configs['env']}_best_model.pth", q_net, target_q_net)

    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'episode_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(configs['n_episodes']):
            observation, _ = env.reset()
            observation = torch.from_numpy(observation)
            finished = False
            episode_length = 0
            cumulative_reward = 0

            while not finished:
                if episode < configs['warm_up_episodes']:
                    action = env.action_space.sample()
                else:
                    action = q_net(observation.unsqueeze(0)).argmax(axis=-1).squeeze().item()

                next_observation, reward, terminated, truncated, _ = env.step(action)

                cumulative_reward += reward
                next_observation = torch.from_numpy(next_observation)
        
                action = np.array([action], dtype=np.float32)
                action = torch.from_numpy(action)

                if not configs['test']:
                    buffer.add(observation, action, next_observation, reward, terminated)

                observation = next_observation
                finished = terminated or truncated
                episode_length += 1

            rewards.append(cumulative_reward)
            writer.writerow({'episode': episode, 'reward': cumulative_reward, 'episode_length': episode_length})
            print("Episode: ", episode, "Cumulative Reward:", cumulative_reward, "Episode length:", episode_length)

            if len(rewards) >= save_threshold and not configs['test']:
                current_avg_reward = np.mean(rewards[-save_threshold:])
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    save_path = f"{log_folder}/{configs['env']}_best_model.pth"
                    save_models(q_net, target_q_net, save_path)
                    print(f"New best model saved with average reward: {best_avg_reward}")

            if len(buffer) >= configs['batch_size'] and not configs['test']:
                for _ in range(configs['train_steps']):
                    _ = learn(
                        q_net,
                        target_q_net,
                        buffer.sample(configs['batch_size']),
                        optimizer,
                        configs['gamma'],
                    )

                if episode % configs['target_update_freq'] == 0:
                    target_q_net.load_state_dict(q_net.state_dict())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent in a gym environment')
    parser.add_argument('--agent', type=str, default='dqn', help='Specify the RL agent (dqn or ddqn)')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Specify the gym environment')
    parser.add_argument('--test', type=bool, default=False, help='Test model?')
    parser.add_argument('--config', type=str, default='./configs.yaml', help='Path to configuration file')
    args = parser.parse_args()

    configs = {
        'agent': args.agent,
        'env': args.env,
        'test': args.test
    }

    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    set_all_seeds(configs['seed'])
    main(configs)
