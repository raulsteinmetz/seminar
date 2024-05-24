import yaml
import argparse
import os

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.buffer import ReplayBuffer
from util.tools import set_all_seeds
from util.q_net import learn_ddqn_mlp, learn_dqn_mlp, make_q_nets, save_models, load_models

configs = {}

def evaluate(env, q_net, final=False):
    if final:
        _, dummy = make_q_nets(env, configs['units'], configs['layers'])
        load_models(f"logs/{configs['env']}/{configs['agent']}/best_model.pth", q_net, dummy)
        eval_episodes = configs['final_eval_episodes']
    eval_episodes = configs['eval_episodes']
    rewards = []
    for _ in range(eval_episodes):
        observation, _ = env.reset()
        observation = torch.from_numpy(observation)
        finished = False
        cumulative_reward = 0

        while not finished:
            action = q_net(observation.unsqueeze(0)).argmax(axis=-1).squeeze().item()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            next_observation = torch.from_numpy(next_observation)
            observation = next_observation
            finished = terminated or truncated

        rewards.append(cumulative_reward)
    avg_reward = np.mean(rewards)
    if not final:
        print(f"Evaluation over {eval_episodes} episodes: Average Reward: {avg_reward}")
    return avg_reward

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

    log_folder = f"logs/{configs['env']}/{configs['agent']}"
    os.makedirs(log_folder, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_folder)

    rewards = []
    n = 100
    best_avg_reward = -float('inf')

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

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1

        rewards.append(cumulative_reward)
        if len(rewards) >= n:
            avg_train_reward = np.mean(rewards[-n:])
            writer.add_scalar('train_return', avg_train_reward, episode)
            if avg_train_reward > best_avg_reward:
                best_avg_reward = avg_train_reward
                save_path = f"logs/{configs['env']}/{configs['agent']}/best_model.pth"
                save_models(q_net, target_q_net, save_path)
                print(f"New best model saved with average reward: {best_avg_reward}")

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

            if episode % configs['target_update_freq'] == 0:
                target_q_net.load_state_dict(q_net.state_dict())

        if (episode + 1) % configs['eval_freq'] == 0:
            avg_reward = evaluate(env, q_net, False)
            writer.add_scalar('eval_return', avg_reward, episode)

    writer.close()

    # final eval
    print (f'Final evaluate moving average for {configs["final_eval_episodes"]} episodes: {evaluate(env, q_net, True)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent in a gym environment')
    parser.add_argument('--agent', type=str, default='dqn', help='Specify the RL agent (dqn or ddqn)')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Specify the gym environment')
    parser.add_argument('--config', type=str, default='./configs.yaml', help='Path to configuration file')
    args = parser.parse_args()

    configs = {
        'agent': args.agent,
        'env': args.env
    }

    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    set_all_seeds(configs['seed'])
    main(configs)
