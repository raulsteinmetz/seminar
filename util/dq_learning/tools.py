import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.dq_learning.buffer import ReplayBuffer
from util.dq_learning.q_net import make_q_nets, save_models, load_models

def _evaluate(env, q_net, configs, device, final=False):
    if final:
        _, dummy = make_q_nets(env, configs['units'])
        load_models(f"logs/{configs['env']}/{configs['agent']}/best_model.pth", q_net, dummy)
        eval_episodes = configs['final_eval_episodes']
    else:
        eval_episodes = configs['eval_episodes']

    q_net.to(device)
    rewards = []
    for _ in range(eval_episodes):
        observation, _ = env.reset()
        observation = torch.from_numpy(observation).to(device)
        finished = False
        cumulative_reward = 0

        while not finished:
            action = q_net(observation.unsqueeze(0)).argmax(axis=-1).squeeze().item()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            next_observation = torch.from_numpy(next_observation).to(device)
            observation = next_observation
            finished = terminated or truncated

        rewards.append(cumulative_reward)
    avg_reward = np.mean(rewards)
    return avg_reward

def train(configs, env, q_net, target_q_net, learn, device):

    q_net.to(device)
    target_q_net.to(device)

    target_q_net.load_state_dict(q_net.state_dict()) # target initiates as a hardcopy of q
    optimizer = torch.optim.Adam(q_net.parameters(), configs['learning_rate']) # adam is the most common in the area

    buffer = ReplayBuffer(configs['replay_buffer_capacity'], env.observation_space.shape[0], 1) # mem replay

    # logs 
    log_folder = f"logs/{configs['env']}/{configs['agent']}"
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    rewards = []
    n = 100 # moving average
    best_avg_reward = -float('inf')
    
    for episode in tqdm(range(configs['n_episodes']), desc="Training Episodes"):
        observation, _ = env.reset()
        observation = torch.from_numpy(observation).to(device)
        finished = False
        episode_length = 0
        episode_reward = 0

        while not finished:
            if episode < configs['warm_up_episodes']: # initially we warmup to fill the memory (random actions)
                action = env.action_space.sample()
            else:
                action = q_net(observation.unsqueeze(0)).argmax(axis=-1).squeeze().item()

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            next_observation = torch.from_numpy(next_observation).to(device)
    
            action = np.array([action], dtype=np.float32)
            action = torch.from_numpy(action).to(device)

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1

        rewards.append(episode_reward)
        if len(rewards) >= n:
            avg_train_reward = np.mean(rewards[-n:])
            writer.add_scalar('train_return', avg_train_reward, episode)
            if avg_train_reward > best_avg_reward:
                best_avg_reward = avg_train_reward
                save_path = f"logs/{configs['env']}/{configs['agent']}/best_model.pth"
                save_models(q_net, target_q_net, save_path)
                # tqdm.write(f"New best model saved with average reward: {best_avg_reward}")

        if len(buffer) >= configs['batch_size']:
            for _ in range(configs['train_steps']):
                _ = learn(
                    q_net,
                    target_q_net,
                    buffer.sample(configs['batch_size']),
                    optimizer,
                    configs['gamma'],
                    device
                )

            if episode % configs['target_update_freq'] == 0:
                target_q_net.load_state_dict(q_net.state_dict()) # hard update is default on dqn and ddqn

        if (episode + 1) % configs['eval_freq'] == 0:
            avg_reward = _evaluate(env, q_net, configs, device, False) # eval every n episodes
            writer.add_scalar('eval_return', avg_reward, episode)

    writer.close()

    # final eval
    print(f'Final evaluate moving average for {configs["final_eval_episodes"]} \
          episodes: {_evaluate(env, q_net, configs, device, True)}')
