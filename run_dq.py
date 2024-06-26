import yaml
import argparse
import torch

import gym

from util.dq_learning.tools import train
from util.tools import set_all_seeds
from util.dq_learning.q_net import learn_ddqn, learn_dqn, make_q_nets

configs = {}


def main():
    set_all_seeds(configs['seed'])
    env = gym.make(configs['env'])
    q_net, target_q_net = make_q_nets(env, configs['units'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if configs['agent'] == 'dqn':
        learn = learn_dqn
    elif configs['agent'] == 'ddqn':
        learn = learn_ddqn
    train(configs, env, q_net, target_q_net, learn, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent in a gym environment')
    parser.add_argument('--agent', type=str, default='dqn', help='Specify the RL agent (dqn or ddqn)')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Specify the gym environment')
    parser.add_argument('--config', type=str, default='./configs/config_dq.yaml', help='Path to configuration file')
    args = parser.parse_args()

    configs = {
        'agent': args.agent,
        'env': args.env
    }

    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    main()
