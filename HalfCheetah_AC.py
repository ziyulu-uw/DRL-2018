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

# Combine actor-critic and evolution strategy in HalfCheetah-v2 environment

class Net(nn.Module): 
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_input,n_hidden)
#        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear2 = nn.Linear(n_hidden,n_output)
        self.layers = [self.linear1, self.linear2]
    
    def forward(self,x):

        x = F.relu(self.linear1(x))
        mean = F.tanh(self.linear2(x))
        distr = D.Normal(loc=mean, scale=0.05)
        sampled_action = distr.rsample().detach() 
        sampled_log_prob = distr.log_prob(sampled_action).sum(1).view(-1, 1)
        return sampled_action,sampled_log_prob,distr
    
class NNBaseline(nn.Module):
    
    def __init__(self,n_input,n_hidden):   
        super(NNBaseline,self).__init__()
        self.linear1 = nn.Linear(n_input,n_hidden)
        self.linear2 = nn.Linear(n_hidden,1)
        self.layers = [self.linear1, self.linear2]
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        out = F.tanh(self.linear2(x))
        return out
            
def compute_fitness(env,n_input,n_hidden,n_output,theta, max_path_length,n_episode):
    fitness = 0
    net = Net(n_input,n_hidden,n_output)
    net.load_state_dict(theta)
    episodes = []
    for i_episode in range(n_episode):
        obs, rews, log_probs = [], [], []
        ob_ = env.reset()
        steps = 0
        while True:
            # env.render()
            ob = torch.from_numpy(ob_).float().unsqueeze(0)
            obs.append(ob)
            ac_, log_prob, _ = net(ob)
            log_probs.append(log_prob)
#            print(ac_)
            ac = ac_.squeeze(0).numpy()
#            print(ac)
            ob_, rew, done, _ = env.step(ac)
            rews.append(rew)
            fitness += rew
            steps += 1
            if done or steps >= max_path_length:
                break
        episode = {"observation":torch.cat(obs,0),\
                   "reward":torch.Tensor(rews),\
                   "log_probs":torch.cat(log_probs,0)}
        episodes.append(episode)
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
#    print(total_time_steps)
    return fitness/n_episode,episodes

def ESPG(env_name,N_EXP,N_GEN,POP_SIZE,fitness_eval_episodes,max_path_length,ALPHA,sigma, n_hidden,\
         batchSize, MB, minibatch, Alr,Clr,goal):
#    global total_time_steps
    i_exp = 0
    center_return_all = []
    while i_exp < N_EXP:
        baseline_update = 0
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
        center_net = Net(ob_dim,n_hidden,ac_dim)
        theta_center = center_net.state_dict()
        
        center_return_list = []
        #build baseline
        gamma = 0.9 #discount factor
        learning_rate = Clr
        from torch.optim import Adam
        baseline = NNBaseline(ob_dim,32)
        baseline_loss = nn.MSELoss()
        baseline_optimizer = Adam(baseline.parameters(), lr=learning_rate)
        
        for i_gen in range(N_GEN):
            
             episodes_all = []
            
             center_return,center_episodes = compute_fitness(env,ob_dim,n_hidden,ac_dim,theta_center,max_path_length,fitness_eval_episodes)
             print('gen',i_gen,'center',center_return)
             center_return_list.append(center_return)
             episodes_all.extend(center_episodes)
             
             if center_return <= 0.8*goal:
                 print('now is es')   
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
                     fitness,pop_episodes = compute_fitness(env,ob_dim,n_hidden,ac_dim,theta,max_path_length,fitness_eval_episodes)
                     fitness_list[i_pop] = fitness
                     episodes_all.extend(pop_episodes)
                 ##update baseline
                 ob_all = torch.cat([episode["observation"] for episode in episodes_all],0)
                 q_n_ = []
                 for episode in episodes_all:
                     rewards = episode['reward']
                     num_steps = len(rewards)
                     q_n_.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                               for t in range(num_steps)]))
                 q_n = torch.cat(q_n_, 0)
                 q_n_std = q_n.std()
                 q_n_mean = q_n.mean()
                 target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
                 if MB:
                     from sklearn.utils import shuffle
                     ob_all_np = ob_all.numpy()
                     target_np = target.numpy()
                     data, label = shuffle(ob_all_np,target_np,random_state=0)
                     import math
                     steps = math.ceil(len(ob_all)/minibatch)
                     for i in range(steps):
                        pred = baseline(torch.tensor(data[i:i+minibatch]))
                        true = torch.tensor(label[i:i+minibatch])
                        baseline_optimizer.zero_grad()
                        b_loss = baseline_loss(pred, true)
                        b_loss.backward()
                        baseline_optimizer.step()
                        baseline_update += 1
                 else:
                     b_n = baseline(ob_all)
                     baseline_optimizer.zero_grad()
                     b_loss = baseline_loss(b_n, target)
                     b_loss.backward()
                     baseline_optimizer.step()
                     baseline_update += 1
                 ##normalization
                 ave_fit = np.mean(fitness_list)
                 fit_std = np.std(fitness_list) + 1e-8
                 ##update theta center
                 for key in theta_center.keys():
                     move = torch.zeros_like(theta_center[key])
                     epslion_list = epslion_all[key]
                     
                     for i in range(POP_SIZE):
                         new = torch.add(move,(fitness_list[i]-ave_fit)/fit_std, epslion_list[i]) 
                         move = new  
                     es_move = move
                     new = torch.add(theta_center[key],ALPHA/(POP_SIZE*sigma),es_move).squeeze()
                     theta_center[key] = new
             
             else:
                print('now is pg')
                batch_size = batchSize
                policy_net = Net(ob_dim,n_hidden,ac_dim)
                policy_net.load_state_dict(theta_center)
                basenet = baseline
                theta_center = train_PG(env_name,policy_net,basenet,batch_size,MB,minibatch,max_path_length,Alr,Clr,random_seed)

        center_return_all.append(center_return_list)
        print('total baseline update times',baseline_update)   
        i_exp += 1
    
    return center_return_all
    
def policy_GD_loss(log_prob,adv,num_path):
    return -(log_prob.view(-1,1)*adv).sum()/num_path

def pathlength(path):
    return len(path["reward"])
    
def train_PG(env_name,policy_net,basenet,batch_size,MB,minibatch,max_path_length,Alr,Clr,seed):
#    print(batch_size)
#    print(policy_net.state_dict())
    gamma = 0.9 #discount factor
#    learning_rate = lr
#    n_iter = 1
#    animate = True
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    min_timesteps_per_batch = batch_size
    from torch.optim import Adam
    policy_loss = policy_GD_loss
    policy_optimizer = Adam(policy_net.parameters(), lr=Alr)
    #baseline
    baseline = basenet
    baseline_loss = nn.MSELoss()
    baseline_optimizer = Adam(baseline.parameters(), lr=Clr)
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
    ob_all = torch.cat([path["observation"] for path in paths],0)
#    ob_all_np = ob_all.numpy()
#    print(ob_all_np.shape)
#    ac_nac = torch.cat([path["action"] for path in paths],0)
    # Build arrays for observation, action for the policy gradient update by concatenating 
    # across paths
    
    q_n_ = []
    for path in paths:
        rewards = path['reward']
        num_steps = pathlength(path)
        q_n_.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                               for t in range(num_steps)]))
#    for q in q_n_:
#        print(q)        
    q_n = torch.cat(q_n_, 0)
    q_n_std = q_n.std()
    q_n_mean = q_n.mean()
    target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
    
    b_n = baseline(ob_all)
    if MB:
        ob_all_np = ob_all.numpy()
        target_np = target.numpy()
        from sklearn.utils import shuffle
        data, label = shuffle(ob_all_np,target_np,random_state=0)
        #update baseline
        import math
        steps = math.ceil(len(ob_all)/minibatch)
        for i in range(steps):
            pred = baseline(torch.tensor(data[i:i+minibatch]))
            true = torch.tensor(label[i:i+minibatch])
            baseline_optimizer.zero_grad()
            b_loss = baseline_loss(pred, true)
            b_loss.backward()
            baseline_optimizer.step()
        
    else:
        baseline_optimizer.zero_grad()
        b_loss = baseline_loss(b_n, target)
        b_loss.backward()
        baseline_optimizer.step()

    b_n_scaled = b_n * q_n_std + q_n_mean
    adv_n = (q_n - b_n_scaled).detach()
    #normalize advantage       
    adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())
    #policy update
    log_probs = torch.cat([path["log_probs"] for path in paths],0)
    policy_optimizer.zero_grad()
    loss = policy_loss(log_probs,adv_n,len(paths))
    loss.backward()
    policy_optimizer.step()
    
#    grad = {}
#    grad['linear1.weight'] = policy_net.linear1.weight.grad
#    grad['linear1.bias'] = policy_net.linear1.bias.grad
#    grad['linear2.weight'] = policy_net.linear2.weight.grad
#    grad['linear2.bias'] = policy_net.linear2.bias.grad

    state_dict = policy_net.state_dict()
#    print(state_dict)
    return state_dict

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--n_exp', type=int, default = 3)
    parser.add_argument('--n_gen', type=int)
    parser.add_argument('--pop_size', type=int, default = 100)
    parser.add_argument('--fitness_eval_episodes', type=int,default = 1)
    parser.add_argument('--max_path_length', type=int)
    parser.add_argument('--ALPHA', type=int, default = 0.1)
    parser.add_argument('--sigma', type=int, default = 0.05)
    parser.add_argument('--n_hidden', type=int, default = 32)
#    parser.add_argument('--pg_flag','-pg',action = 'store_true')
    parser.add_argument('--batch', type=int, default = 1000)
    parser.add_argument('--MB', '-mb',action = 'store_true')
    parser.add_argument('--minibatch',type=int, default = 256)
    parser.add_argument('--Alr',type=float,default = 0.001)
    parser.add_argument('--Clr',type=float,default = 0.001)
    parser.add_argument('--goal', type=int)
#    parser.add_argument('--hold', type=float,default = 0.8)

    args = parser.parse_args()
    
    max_path_length = args.max_path_length if args.max_path_length > 0 else None
    
    center_return = ESPG(env_name = args.env_name, N_EXP = args.n_exp, N_GEN = args.n_gen,POP_SIZE = args.pop_size,\
                       fitness_eval_episodes =args.fitness_eval_episodes, max_path_length = max_path_length,\
                       ALPHA = args.ALPHA, sigma = args.sigma, n_hidden = args.n_hidden,\
                       batchSize = args.batch, MB = args.MB, minibatch = args.minibatch, \
                       Alr = args.Alr, Clr = args.Clr, goal = args.goal)
#    rs_ = []
#    for i in range(args.n_exp):
#        rs_.append(str(i+1))
#    rs = ''.join(rs_)
    import pickle
    filename = '{env}-{N_GEN}-{POP_SIZE}-{max_path_length}-{batch}-{Alr}-{Clr}-{goal}-{minibatch}-{mb}.dat'.format(env ='HC',\
            N_GEN = str(args.n_gen),POP_SIZE = str(args.pop_size),max_path_length = str(max_path_length),\
            batch = str(args.batch),Alr = str(args.Alr),Clr = str(args.Clr), goal = str(args.goal),\
            minibatch = str(args.minibatch),mb = str(args.MB))
    outfile = open(filename,'wb')
    pickle.dump(center_return,outfile)
    outfile.close()
    
if __name__ == "__main__":
    main()
    
    
