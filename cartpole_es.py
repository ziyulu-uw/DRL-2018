import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import gym
import gym.spaces
import numpy as np
import argparse
import os
from torch import Tensor
from torch.distributions import Categorical
from torch.distributions import Normal

# Evolution strategy in CartPole-v0 environment

class ES_Model(nn.Module):
    def __init__(self):
        ## save the parameters for copying model
        super(ES_Model, self).__init__()
        self.input = nn.Linear(4,16,bias = False)
        self.output = nn.Linear(16,2,bias = False)

        ## In this implementation we just set weights to init as zero
        ## don't worry about bias to make things simpler
        self.input.weight.data.zero_()
        self.output.weight.data.zero_()

        ## these are the epsilon for theta_i
        self.epsilon_input = None
        self.epsilon_output = None

    def forward(self, x):
        out = F.relu(self.input(x))
        log_probs = F.log_softmax(self.output(out),dim=1)
        return log_probs

    def mutate(self, sigma):
        ## call this for theta_i, after sync with theta_center
        ## to add epsilon
        input_shape = self.input.weight.data.shape
        self.epsilon_input = Normal(torch.zeros(input_shape),sigma).sample()
        output_shape = self.output.weight.data.shape
        self.epsilon_output = Normal(torch.zeros(output_shape),sigma).sample()
        self.input.weight.data += self.epsilon_input
        self.output.weight.data += self.epsilon_output
    def update_theta_center(self,theta_i, alpha, fitness, pop_size, sigma):
        ## update theta center with a theta_i
        self.input.weight.data += alpha * fitness * theta_i.epsilon_input /(pop_size*sigma)
        self.output.weight.data += alpha * fitness * theta_i.epsilon_output /(pop_size*sigma)

def compute_fitness(env, model, n_episode=1):
    fitness = 0
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(200):
            # env.render()
            log_probs = model.forward(Tensor(observation).reshape(1,-1))

            ## from probs to action
            dist = Categorical(logits=log_probs)
            action = dist.sample()

            ## take action in env
            observation, reward, done, info = env.step(action.item())
            fitness += reward
            if done:
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
    return fitness/n_episode

sigma = 10
## theta here is the center theta

POP_SIZE = 500
fitness_eval_episodes = 1
N_GEN = 10

ALPHA = 10
# for i_gen in range(N_GENERATIONS):

fitness_list = [0 for _ in range(POP_SIZE)]

center_return_all = []
for i_experiment in range(5):
    random_seed = i_experiment+1
    env = gym.make('CartPole-v0')
#    print(gym)

    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    theta_center = ES_Model()
    theta_list = [ES_Model() for _ in range(POP_SIZE)]

    center_return_list = []
    center_return_list.append(compute_fitness(env,theta_center,10))
    for i_gen in range(N_GEN):
        ## first generate theta_i population
        for i in range(POP_SIZE):
            theta_i = theta_list[i]
            theta_i.load_state_dict(theta_center.state_dict()) # first sync with theta_center
            theta_i.mutate(sigma)
            fitness = compute_fitness(env,theta_list[i],fitness_eval_episodes)
            fitness_list[i] = fitness

        ## get mean and std of fitness for normalization
        ave_fit = np.mean(fitness_list)
        fit_std = np.std(fitness_list) + 1e-8

        for i_pop in range(POP_SIZE):
            ## normalize fitness
            fitness = fitness_list[i_pop]
            fitness -= ave_fit
            fitness /= fit_std
            theta_i = theta_list[i_pop]

            ## now update theta_center
            theta_center.update_theta_center(theta_i, ALPHA, fitness, POP_SIZE, sigma)
        
        center_return = compute_fitness(env,theta_center,10) #compute fitness for the updated center
        print('gen',i_gen,'center',center_return,'ave',ave_fit)
        center_return_list.append(center_return)

    center_return_all.append(center_return_list)

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.tsplot(data=np.array(center_return_all),color='blue')

plt.xlabel('Epoch')
plt.ylabel('Return')
plt.tight_layout()
plt.show()

