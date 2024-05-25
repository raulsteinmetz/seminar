import gymnasium as gym
import numpy as np
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from util.q_learning.tools import train as train, evaluate
from util.tools import set_all_seeds

configs = {}


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    summary_writer = SummaryWriter(configs['log_dir'])
    train(Q, env, summary_writer, configs)
    evaluate(Q, env, configs, episodes=1000)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a q_table agent in frozen lake')
    parser.add_argument('--config', type=str, default='./configs/config_q.yaml', help='Path to configuration file')
    args = parser.parse_args()


    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    set_all_seeds(configs['seed'])
    main()
