import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy

class Net_Continuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net_Continuous, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim)-2)
        
        self.epsilon_input_weight = None
        self.epsilon_input_bias = None
        self.epsilon_mu_weight = None
        self.epsilon_mu_bias = None
        self.epsilon_action_log_std = None
        
    def forward(self, x):
        out = F.relu(self.input(x))
        mu = F.tanh(self.mu(out))
        action_log_std = self.action_log_std.expand_as(mu)
        return mu, action_log_std
    
    def mutate(self, sigma):
        ## call this for theta_i, after sync with theta_center
        ## to add epsilon
        input_weight_shape = self.input.weight.data.shape
        self.epsilon_input_weight = Normal(torch.zeros(input_weight_shape),sigma).sample()
        self.input.weight.data += self.epsilon_input_weight
        
        input_bias_shape = self.input.bias.data.shape
        self.epsilon_input_bias = Normal(torch.zeros(input_bias_shape),sigma).sample()
        self.input.bias.data += self.epsilon_input_bias
        
        mu_weight_shape = self.mu.weight.data.shape
        self.epsilon_mu_weight = Normal(torch.zeros(mu_weight_shape),sigma).sample()
        self.mu.weight.data += self.epsilon_mu_weight
        
        mu_bias_shape = self.mu.bias.data.shape
        self.epsilon_mu_bias = Normal(torch.zeros(mu_bias_shape),sigma).sample()
        self.mu.bias.data += self.epsilon_mu_bias
        
        als_shape = self.action_log_std.data.shape
        self.epsilon_action_log_std = Normal(torch.zeros(als_shape),sigma).sample()
        self.action_log_std.data += self.epsilon_action_log_std
        
    def update_theta_center(self,theta_i, alpha, fitness, pop_size, sigma):
        ## update theta center with a theta_i
        self.input.weight.data += alpha * fitness * theta_i.epsilon_input_weight/(pop_size*sigma)
        self.input.bias.data += alpha * fitness * theta_i.epsilon_input_bias/(pop_size*sigma)
        self.mu.weight.data += alpha * fitness * theta_i.epsilon_mu_weight/(pop_size*sigma)
        self.mu.bias.data += alpha * fitness * theta_i.epsilon_mu_bias/(pop_size*sigma)
        self.action_log_std.data += alpha * fitness * theta_i.epsilon_action_log_std/(pop_size*sigma)
        
class Net_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net_Baseline, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,1)
    def forward(self, x):
        out = F.relu(self.input(x))
        out = F.tanh(self.output(out))
        return out

''' 
###testing
net_center = Net_Continuous(4,8,2)
theta_center = net_center.state_dict()
print(theta_center)
net1 = copy.deepcopy(net_center)
net1.mutate(1)
theta = net1.state_dict()
print(theta)
'''






