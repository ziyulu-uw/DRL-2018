#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:12:01 2018

@author: ZIYU
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
#import torch.optim as optim
import gym.spaces
#import sys
import numpy as np
#import argparse
#import os
import copy
#import time
#'HalfCheetah-v2'
'''
env = gym.make('HalfCheetah-v2')
print(env.action_space) #Box(6,) 
print(env.action_space.high) #[ 1.  1.  1.  1.  1.  1.]
#print(len(env.action_space.high))
print(env.action_space.low)  #[-1. -1. -1. -1. -1. -1.]
'''

# Combine vanilla PG and evolution strategy in HalfCheetah-v2 environment

class Net(nn.Module): 
    
    def __init__(self,n_input,n_hidden,n_output,initW):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_input,n_hidden)
#        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear2 = nn.Linear(n_hidden,n_output)
            
        if initW:
            self.init_weights()

    
    def forward(self,x):

        x = F.relu(self.linear1(x))
        mean = F.tanh(self.linear2(x))
        distr = D.Normal(loc=mean, scale=0.05)
        sampled_action = distr.rsample().detach() 
        sampled_log_prob = distr.log_prob(sampled_action).sum(1).view(-1, 1)
        return sampled_action,sampled_log_prob,distr
    
    def init_weights(self):
        layers = []
        layers.append(self.linear1)
        layers.append(self.linear2)
        for layer in layers:
            layer.weight.data.normal_(0,1e-12)
            layer.bias.data.normal_(0,1e-12)
            
def compute_fitness(env,n_input,n_hidden,n_output,theta, max_path_length,n_episode):
#    global total_time_steps
    fitness = 0
    net = Net(n_input,n_hidden,n_output,False)
    net.load_state_dict(theta)
        
    for i_episode in range(n_episode):
        ob_ = env.reset()
        steps = 0
        while True:
            env.render()
            ob = torch.from_numpy(ob_).float().unsqueeze(0)
            ac_, log_prob, _ = net(ob)
#            print(ac_)

            ac = ac_.squeeze(0).numpy()
#            print(ac)
            ob_, rew, done, _ = env.step(ac)
            fitness += rew
            steps += 1
            if done or steps >= max_path_length:
#                total_time_steps += steps
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
#    print(total_time_steps)
    return fitness/n_episode

def ES(env_name,N_EXP,N_GEN,POP_SIZE,fitness_eval_episodes,max_path_length,ALPHA,sigma, n_hidden,pg, batchSize, w_pg0):
#    global total_time_steps
    i_exp = 0
    center_return_all = []
    while i_exp < N_EXP:
        w_es = 1-w_pg0
        w_pg = w_pg0
        env = gym.make(env_name)
        max_path_length = max_path_length or env.spec.max_episode_steps
#        print(max_path_length)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        random_seed = (i_exp+1) * 1
#        random_seed = 1
#        random_seed = 8 + i_exp
        print(random_seed)
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        ob_dim = env.observation_space.shape[0]
#        print(ob_dim)
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
#        print(ac_dim)
        center_net = Net(ob_dim,n_hidden,ac_dim,False)
        theta_center = center_net.state_dict()
        
        center_return_list = []
        for i_gen in range(N_GEN):
            
             center_return = compute_fitness(env,ob_dim,n_hidden,ac_dim,theta_center,max_path_length,fitness_eval_episodes)
             center_return_list.append(center_return)
                
             epslion1_all = []#linear1.weight
             epslion2_all = []#linear1.bais
             epslion3_all = []#linear2.weight
             epslion4_all = []#linear2.bais             
             epslion_all = {}
             for i_pop in range(POP_SIZE):
    #            new_theta = copy.deepcopy(theta_center)
                 epslion1 = torch.normal(mean = torch.zeros(n_hidden, ob_dim),std = sigma)
                 epslion2 = torch.normal(mean = torch.zeros(1,n_hidden),std = sigma)
                 epslion3 = torch.normal(mean = torch.zeros(ac_dim,n_hidden),std = sigma)
                 epslion4 = torch.normal(mean = torch.zeros(1,ac_dim),std = sigma)

                 epslion1_all.append(epslion1)
                 epslion2_all.append(epslion2)
                 epslion3_all.append(epslion3)
                 epslion4_all.append(epslion4)

             epslion_all['linear1.weight'] = epslion1_all
             epslion_all['linear1.bias'] = epslion2_all
             epslion_all['linear2.weight'] = epslion3_all
             epslion_all['linear2.bias'] = epslion4_all

    #        print(type(epslion_all['linear1.weight']))
    #        print(epslion_all['linear1.weight'][0])
             theta_all = []
             for i_pop in range(POP_SIZE):
                 new_theta = copy.deepcopy(theta_center)
                 for key in new_theta.keys():
                     new_paras = torch.add(new_theta[key],sigma,epslion_all[key][i_pop]).squeeze()
                     new_theta[key] = new_paras
                 theta_all.append(new_theta)
                       
             fitness_list = [0 for _ in range(POP_SIZE)]
             for i_pop in range(POP_SIZE):
                 theta = theta_all[i_pop]
                 fitness = compute_fitness(env,ob_dim,n_hidden,ac_dim,theta,max_path_length,fitness_eval_episodes)
                 fitness_list[i_pop] = fitness
#                print(fitness)
             ##normalization
             ave_fit = np.mean(fitness_list)
             fit_std = np.std(fitness_list) + 1e-8
             
             if pg:
                batch_size = batchSize
                policy_net = Net(ob_dim,n_hidden,ac_dim,False)
#                print(w_es,w_pg)
                policy_net.load_state_dict(theta_center)
                grad_dic = train_PG(env_name,policy_net,batch_size)
#                total_time_steps += batch_size
#                print(total_time_steps)
#                print('!!!')
            ## update theta center
#            env.seed(random_seed)
#            torch.manual_seed(random_seed)
#            np.random.seed(random_seed)
             print(w_es,w_pg)
             for key in theta_center.keys():
                 move = torch.zeros_like(theta_center[key])
                 epslion_list = epslion_all[key]
                     
                 for i in range(POP_SIZE):
                     new = torch.add(move,(fitness_list[i]-ave_fit)/fit_std, epslion_list[i]) 
                     move = new
                 
                 new1 = torch.add(theta_center[key],w_es*ALPHA/(POP_SIZE*sigma),move).squeeze()
                 theta_center[key] = new1
                 if pg:   
                     grad = grad_dic[key]
                     new2 = torch.add(new1,w_pg,grad)
                     theta_center[key] = new2
                     
#             print(theta_center)
             print('gen',i_gen,'center',center_return,'ave',ave_fit)
             if pg:
                 w_pg = w_pg*10
                 w_es = 1 - w_pg
#                 print(w_es)
#                 print(w_pg)
                
        
        center_return_all.append(center_return_list)
            
        i_exp += 1
    
    return center_return_all
    
def policy_GD_loss(log_prob,adv,num_path):
    return -(log_prob.view(-1,1)*adv).sum()/num_path

def pathlength(path):
    return len(path["reward"])

#policy_net = Net(4,4,1,False)
    
def train_PG(env_name,policy_net,batch_size):
#    print(policy_net.state_dict())
    gamma = 0.9 #discount factor
#    learning_rate = 5e-3
#    n_iter = 1
#    animate = True
#    torch.manual_seed(seed)
#    np.random.seed(seed)
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    max_path_length = env.spec.max_episode_steps
    min_timesteps_per_batch = batch_size
#    from torch.optim import Adam
    policy_loss = policy_GD_loss
#    policy_optimizer = Adam(policy_net.parameters(), lr=learning_rate)
        # Collect paths until we have enough timesteps
    timesteps_this_batch = 0
    paths = []
    while True: #generate a batch 
        # Simulate one episode and get a path
        ob_ = env.reset()
        obs, acs, rews, log_probs = [], [], [], []
#        animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
        steps = 0
        while True: #generate a path
#            if animate_this_episode:
#                env.render()
#                time.sleep(0.05)
            ob = torch.from_numpy(ob_).float().unsqueeze(0)
            obs.append(ob)
            ac_, log_prob, _ = policy_net(ob)
            acs.append(ac_)
            log_probs.append(log_prob)
            if discrete:
                ac = int(ac_)
            else:
                ac = ac_.squeeze(0).numpy() 
#            print(ac)
#                if ac[0] > 3: #InvertedPendulum setting
#                    ac = [3]
#                elif ac[0] < -3:
#                    ac = [-3]
            # Simulate one time step
            ob_, rew, done, _ = env.step(ac)
            rews.append(rew)
            steps += 1
            if done or steps > max_path_length:
                break
        path = {"observation" : torch.cat(obs, 0), 
                "action" : torch.cat(acs, 0),
                "reward" : torch.Tensor(rews),
                "log_probs": torch.cat(log_probs, 0)}
        
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch > min_timesteps_per_batch:
            break

    # Build arrays for observation, action for the policy gradient update by concatenating 
    # across paths
#        ob_no = torch.cat([path["observation"] for path in paths],0)
#        ac_nac = torch.cat([path["action"] for path in paths],0)
    
    q_n_ = []
    for path in paths:
        rewards = path['reward']
        num_steps = pathlength(path)
        q_n_.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                               for t in range(num_steps)]))
    q_n = torch.cat(q_n_, 0)
    adv_n = q_n
    adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())
    log_probs = torch.cat([path["log_probs"] for path in paths],0)
#    policy_optimizer.zero_grad()
    loss = policy_loss(log_probs,adv_n,len(paths))

    loss.backward()
    
    grad = {}
    grad['linear1.weight'] = policy_net.linear1.weight.grad
    grad['linear1.bias'] = policy_net.linear1.bias.grad
    grad['linear2.weight'] = policy_net.linear2.weight.grad
    grad['linear2.bias'] = policy_net.linear2.bias.grad

#    print(grad)
#    print('!!!')
#    print(policy_net.state_dict())
    return grad
        
def plot(center_return_all):

    import seaborn as sns
    import matplotlib.pyplot as plt
    
#    for i in range(N_EXP):
    ax = sns.tsplot(data=np.array(center_return_all),color='blue')
        
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.show()
    
#total_time_steps = 0    
#print(total_time_steps)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--n_exp', type=int, default = 5)
    parser.add_argument('--n_gen', type=int)
    parser.add_argument('--pop_size', type=int, default = 100)
    parser.add_argument('--fitness_eval_episodes', type=int,default = 1)
    parser.add_argument('--max_path_length', type=int)
    parser.add_argument('--ALPHA', type=int, default = 0.1)
    parser.add_argument('--sigma', type=int, default = 0.05)
    parser.add_argument('--n_hidden', type=int, default = 32)
    parser.add_argument('--pg_flag','-pg',action = 'store_true')
    parser.add_argument('--batch', type=int, default = 1000)
    parser.add_argument('--w_pg0', type=float, default = 0.0)
#    parser.add_argument('--w_es0', type=float, default = 1.0)
    args = parser.parse_args()
    
    max_path_length = args.max_path_length if args.max_path_length > 0 else None
    
    center_return = ES(env_name = args.env_name, N_EXP = args.n_exp, N_GEN = args.n_gen,POP_SIZE = args.pop_size,\
                       fitness_eval_episodes =args.fitness_eval_episodes, max_path_length = max_path_length,\
                       ALPHA = args.ALPHA, sigma = args.sigma, n_hidden = args.n_hidden,\
                       pg = args.pg_flag, batchSize = args.batch, w_pg0 = args.w_pg0)
    rs_ = []
    for i in range(args.n_exp):
        rs_.append(str(i+1))
    rs = ''.join(rs_)
    import pickle
    filename = '{env}-{N_GEN}-{POP_SIZE}-{max_path_length}-{wPG}-{batch}-{seeds}.dat'.format(env =args.env_name,\
            N_GEN = str(args.n_gen),POP_SIZE = str(args.pop_size),max_path_length = str(max_path_length),\
            wPG = str(args.w_pg0),batch = str(args.batch),seeds = rs)
    outfile = open(filename,'wb')
    pickle.dump(center_return,outfile)
    outfile.close()
    
if __name__ == "__main__":
    main()
    
    
