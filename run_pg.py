import yaml
import argparse
import torch
import gym
from util.policy_gradient.tools import train
from util.tools import set_all_seeds
from util.policy_gradient.policy_net import make_policy_net

configs = {}

def main():
    set_all_seeds(configs['seed'])
    env = gym.make(configs['env'])
    policy_net = make_policy_net(env, configs['units'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(configs, env, policy_net, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agent with REINFORCE in a gym environment')
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0', help='Specify the gym environment')
    parser.add_argument('--config', type=str, default='./configs/config_pg.yaml', help='Path to configuration file')
    args = parser.parse_args()

    configs = {
        'env': args.env
    }

    with open(args.config, 'r') as file:
        file_configs = yaml.safe_load(file)
        configs.update(file_configs)

    main()
