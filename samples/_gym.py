import argparse
import gym

def main(env: str):
    env = gym.make(env, render_mode='human')    

    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            state_, reward, done, _, _ = env.step(env.action_space.sample())
            env.render()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a q_table agent in frozen lake')
    parser.add_argument('--env', type=str, default='LunarLander-v2', help='Name of the Gym Environment')
    args = parser.parse_args()
    main(args.env)
