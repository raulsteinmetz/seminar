import random
import numpy as np
from tqdm import tqdm

def _choose_action(state, Q, epsilon, action_space, eval=False):
    if random.uniform(0, 1) < epsilon and not eval: # exploration (random action)
        return action_space.sample()
    else: # exploitation (action with the best q value)
        return np.argmax(Q[state, :])

def evaluate_policy(Q, env, configs, episodes=100):
    total_rewards = 0

    for _ in range(episodes):
        state = env.reset()[0] # gymnasium returns dictionary
        episode_reward = 0
        done = False
        for _ in range(configs['max_steps']):
            action = _choose_action(state, Q, 0, env.action_space, eval=True) # only exploitation
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            state = next_state

            if done:
                break
        
        total_rewards += episode_reward
    avg_reward = total_rewards / episodes
    print(f"Average reward after evaluation: {avg_reward}")

def train(Q, env, summary_writer, configs):
    epsilon = 1.0
    rewards = []
    moving_avg_rewards = []
    for episode in tqdm(range(configs['episodes'])):
        state = env.reset()[0] # gymnasium returns dictionary
        total_reward = 0

        for _ in range(configs['max_steps']):
            action = _choose_action(state, Q, epsilon, env.action_space, eval=False)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            best_next_action = np.argmax(Q[next_state, :]) # q is updated using the best action for the next state
            Q[state, action] = Q[state, action] + configs['alpha'] * (reward + configs['gamma'] * Q[next_state, best_next_action] - Q[state, action]) # updating table

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        epsilon = max(configs['epsilon_min'], epsilon * configs['epsilon_decay']) # reducing exploration

        # logging
        if len(rewards) >= 100:
            moving_avg = np.mean(rewards[-100:])
        else:
            moving_avg = np.mean(rewards)
        moving_avg_rewards.append(moving_avg)
        summary_writer.add_scalar('train_return_q', moving_avg, episode)