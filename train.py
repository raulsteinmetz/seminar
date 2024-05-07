import yaml
import argparse

import gym
import torch
import torch.nn as nn
import numpy as np

from util.buffer import ReplayBuffer
from util.tools import set_all_seeds
from util.q_net import learn_ddqn_mlp, learn_dqn_mlp, make_q_nets

configs = {}

def main(configs):
    set_all_seeds(configs['seed']) # reproducibility

    env = gym.make(configs['env'])

    q_net, target_q_net = make_q_nets(env, configs['units'], configs['layers'])

    if configs['agent'] == 'dqn':
        learn = learn_dqn_mlp
    elif configs['agent'] == 'ddqn':
        learn = learn_ddqn_mlp
        
    target_q_net.load_state_dict(q_net.state_dict()) # copy net into target_net
    optimizer = torch.optim.Adam(q_net.parameters(), configs['learning_rate'])

    buffer = ReplayBuffer(configs['replay_buffer_capacity'], env.observation_space.shape[0], 1)

    for episode in range(configs['n_episodes']):
        observation, _ = env.reset()
        observation = torch.from_numpy(observation)
        finished = False
        episode_length = 0
        cumulative_reward = 0

        while not finished:
            if episode < configs['warm_up_episodes']:
                action = env.action_space.sample() # random action for filling buffer
            else:
               action = q_net(observation.unsqueeze(0)).argmax(axis=-1).squeeze().item() # actual q_network computation

            next_observation, reward, terminated, truncated, _ = env.step(action)

            cumulative_reward += reward
            next_observation = torch.from_numpy(next_observation)
    
            action = np.array([action], dtype=np.float32)
            action = torch.from_numpy(action)

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1

        print("Episode: ", episode, "Cumulative Reward:", cumulative_reward, "Episode length:", episode_length)


        if len(buffer) >= configs['batch_size']:
            for _ in range(configs['train_steps']):
                _ = learn(
                    q_net,
                    target_q_net,
                    buffer.sample(configs['batch_size']),
                    optimizer,
                    configs['gamma'],
                )

            if episode % configs['target_update_freq'] == 0: # delayed hard update
                target_q_net.load_state_dict(q_net.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent in a gym environment')
    parser.add_argument('--agent', type=str, default='dqn', help='Specify the RL agent (dqn or ddqn)')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Specify the gym environment')
    parser.add_argument('--config', type=str, default='./configs.yaml', help='Path to configuration file')
    args = parser.parse_args()

    configs = {
        'agent': args.agent,
        'env': args.env,
    }

    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    set_all_seeds(configs['seed'])
    main(configs)