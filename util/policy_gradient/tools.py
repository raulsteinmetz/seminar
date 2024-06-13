import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque

def train(configs, env, policy_net, device):
    policy_net.to(device)
    optimizer = Adam(policy_net.parameters(), lr=configs['learning_rate'])
    writer = SummaryWriter(log_dir=f"logs/{configs['env']}/reinforce")

    reward_queue = deque(maxlen=configs['moving_avg_window'])
    for episode in tqdm(range(configs['n_episodes']), desc="Training Episodes"):
        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        
        log_probs = []
        rewards = []
        done = False
        total_reward = 0

        while not done:
            mean, std = policy_net(observation)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            next_observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            total_reward += reward

            log_probs.append(log_prob)
            rewards.append(reward)

            observation = torch.tensor(next_observation, dtype=torch.float32).to(device)

        if len(log_probs) > 0 and len(rewards) > 0:
            learn(configs, optimizer, log_probs, rewards, device)

        reward_queue.append(total_reward)
        if len(reward_queue) == configs['moving_avg_window']:
            moving_avg_reward = np.mean(reward_queue)
            writer.add_scalar('train_return', moving_avg_reward, episode)

    writer.close()

    print(f'Training completed for {configs["n_episodes"]} episodes')

def learn(configs, optimizer, log_probs, rewards, device):
    # discounted returns
    discounts = [configs['gamma'] ** i for i in range(len(rewards))]
    returns = [sum(discounts[j] * rewards[j] for j in range(i, len(rewards))) for i in range(len(rewards))]
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # policy loss
    policy_loss = []
    for log_prob, Gt in zip(log_probs, returns):
        policy_loss.append(-log_prob * Gt)

    if len(policy_loss) > 0:
        policy_loss = torch.stack(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()