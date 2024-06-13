import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, units):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.mean = nn.Linear(units, output_dim)
        self.log_std_layer = nn.Linear(units, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std

def make_policy_net(env, units):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    return PolicyNet(input_dim, output_dim, units)