import torch
import gym

def learn_ddqn(net, target_net, data, optimizer, gamma=0.99, device='cpu'):
    observations, actions, next_observations, rewards, terminations = data
    observations, actions, next_observations, rewards, terminations = (
        observations.to(device),
        actions.to(device),
        next_observations.to(device),
        rewards.to(device),
        terminations.to(device)
    )

    with torch.no_grad():
        next_q_values = net(next_observations)
        next_actions = next_q_values.argmax(dim=1, keepdim=True)
        next_q_values_target = target_net(next_observations)
        selected_target_q_values = next_q_values_target.gather(1, next_actions).squeeze()
        td_target = rewards.flatten() + gamma * selected_target_q_values * (1 - terminations.flatten())

    current_q_values = net(observations).gather(1, actions.long()).squeeze()
    loss = torch.nn.functional.mse_loss(current_q_values, td_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def learn_dqn(net, target, data, optimizer, gamma=0.99, device='cpu'):
    observations, actions, next_observations, rewards, terminations = data
    observations, actions, next_observations, rewards, terminations = (
        observations.to(device),
        actions.to(device),
        next_observations.to(device),
        rewards.to(device),
        terminations.to(device)
    )

    with torch.no_grad():
        next_q_values_target = target(next_observations)
        target_max = next_q_values_target.max(dim=1)[0]
        td_target = rewards.flatten() + gamma * target_max * (1 - terminations.flatten())

    current_q_values = net(observations).gather(1, torch.tensor(actions, dtype=torch.int64).to(device)).squeeze()
    loss = torch.nn.functional.mse_loss(current_q_values, td_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def make_q_nets(env: gym.Env, units: int = 64):
    q_net = torch.nn.Sequential(
        torch.nn.Linear(env.observation_space.shape[0], units),
        torch.nn.ReLU(),
        torch.nn.Linear(units, env.action_space.n),
    )
    target_q_net = torch.nn.Sequential(
        torch.nn.Linear(env.observation_space.shape[0], units),
        torch.nn.ReLU(),
        torch.nn.Linear(units, env.action_space.n),
    )
    return q_net, target_q_net

def save_models(q_net, target_q_net, fpath):
    torch.save({
        'q_net_state_dict': q_net.state_dict(),
        'target_q_net_state_dict': target_q_net.state_dict(),
    }, fpath)

def load_models(fpath, q_net, target_q_net):
    checkpoint = torch.load(fpath)
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])