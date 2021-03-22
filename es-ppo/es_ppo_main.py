import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces
import sys
import numpy as np
import argparse
import os
from torch.distributions import Categorical
from torch.distributions import Normal
import time
from torch import Tensor
from mlp import Net_Continuous, Net_Baseline

LOG_FOLDER_PATH = 'logs'
if not os.path.isdir(LOG_FOLDER_PATH):
    os.mkdir(LOG_FOLDER_PATH)
    print("log directory not found, created folder.")

## add project path if using folders
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import get_reward_to_go, get_gae_advantage, get_trajectories, get_tensors_from_paths
from ppo import ppo_update

## PPO implementation

program_start_time = time.time()
parser = argparse.ArgumentParser()

#'InvertedPendulum-v2'
#'HalfCheetah-v2'
###PPO parameters
parser.add_argument('--envname', type=str, default='HalfCheetah-v2', help='name of the gym env')
#parser.add_argument('-n','--num_train_iter', type=int, default=100,
#                    help='Number of training iterations')
parser.add_argument('-ep','--episode_length', type=int, default=100,
                    help='Number of max episode length')
parser.add_argument('-seed','--random_seed', type=int, default=42,
                    help='random seed')
parser.add_argument('-nep', '--n_episode', type=int, default=21,
                    help='number of episode to run for each training iteration')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('-e','--n_experiment', type=int, default=3,
                    help='Number of experiment to run')
parser.add_argument('-hd','--hidden_dim', type=int, default=64,
                    help='number of hidden layer neurons')

parser.add_argument('-mb','--minibatch_size', type=int, default=64,
                    help='number of data in one minibatch')

parser.add_argument('--ppo_epoch', type=int, default=10,
                    help='number of ppo epoch in one iteration')

parser.add_argument('-lr','--learning_rate', type=float, default=3e-4,
                    help='model learning rate')
parser.add_argument('-wd','--weight_decay', type=float, default=1e-3,
                    help='model weight decay rate')
parser.add_argument('-lam','--lam', type=float, default=0.95,
                    help='lambda value in generalized advantage estimator')
###ES parameters
parser.add_argument('--n_gen', type=int, default=100)
parser.add_argument('--pop_size', type=int, default = 100)
parser.add_argument('--fitness_eval_episodes', type=int,default = 1)
parser.add_argument('--ALPHA', type=int, default = 0.1)
parser.add_argument('--sigma', type=int, default = 0.05)
parser.add_argument('--goal', type=int, default = 375)
###define file name
parser.add_argument('--exp_name', type=str, default='trial_run.txt', help='specify name of log file')

args = parser.parse_args()

#DEBUG = False
#if DEBUG:
#    args.num_train_iter = 100
#    print("DEBUG MODE!!!!!!!!!!!!!!!!!!!!!")

print(args)
sys.stdout.flush()

def ESPG(envname,seed,N_GEN,POP_SIZE,fitness_eval_episodes,episode_length,ALPHA,sigma,hidden_dim,goal,\
         learning_rate,gamma,n_episode,weight_decay,lam, ppo_epoch, minibatch_size):

    env = gym.make(envname)
    episode_length = episode_length or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    random_seed = seed
    print(random_seed)
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    theta_center = Net_Continuous(ob_dim,hidden_dim,ac_dim)
    policy_optimizer = optim.Adam(theta_center.parameters(), lr=learning_rate)
    baseline_model = Net_Baseline(ob_dim, hidden_dim)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    baseline_criterion = nn.MSELoss()
    
    center_return_list = []
    ES_gen = 0
    PG_gen = 0
    tts = 0
    
    for i_gen in range(N_GEN):
        
         episodes_all = []
        
         center_episodes, center_return = get_trajectories(env, theta_center, fitness_eval_episodes, episode_length)
         print('gen',i_gen,'center',center_return)
         center_return_list.append(center_return)
         episodes_all.extend(center_episodes)

         if center_return <= 0.8*goal-10:
             print('ES now')
             ES_gen += 1
             theta_all = [Net_Continuous(ob_dim,hidden_dim,ac_dim) for _ in range(POP_SIZE)]
             fitness_list = [0 for _ in range(POP_SIZE)]
             for i in range(POP_SIZE):
                 theta_i = theta_all[i]
                 theta_i.load_state_dict(theta_center.state_dict()) # first sync with theta_center
                 theta_i.mutate(sigma)
    #                 fitness,pop_episodes = compute_fitness(env,theta_all[i],max_path_length, fitness_eval_episodes)
                 pop_episodes,fitness = get_trajectories(env, theta_i, fitness_eval_episodes, episode_length)
                 fitness_list[i] = fitness
                 episodes_all.extend(pop_episodes)
             ## get mean and std of fitness for normalization
             ave_fit = np.mean(fitness_list)
             fit_std = np.std(fitness_list) + 1e-8
             for i in range(POP_SIZE):
                 ## normalize fitness
                 fitness = fitness_list[i]
                 fitness -= ave_fit
                 fitness /= fit_std
                 theta_i = theta_all[i]
                 ## now update theta_center
                 theta_center.update_theta_center(theta_i, ALPHA, fitness, POP_SIZE, sigma)

             ##update baseline
             n_data, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n = get_tensors_from_paths(episodes_all)
             ppo_update(theta_center, policy_optimizer, baseline_model, baseline_optimizer, baseline_criterion,
                   ppo_epoch, minibatch_size, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n,
                   gamma, lam,update_policy=False)
             
         else:
             print('PPO now')
             PG_gen += 1
             paths, average_episode_return = get_trajectories(env, theta_center, n_episode, episode_length)
             n_data, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n = get_tensors_from_paths(paths)
             ppo_update(theta_center, policy_optimizer, baseline_model, baseline_optimizer, baseline_criterion,
                   ppo_epoch, minibatch_size, obs_n_tensor, log_prob_n_old_tensor, action_n_tensor, rewards_n, mask_n,
                   gamma, lam)
    
    #Total time steps
    tts = (POP_SIZE+1)*fitness_eval_episodes*episode_length*ES_gen + episode_length*n_episode*PG_gen
    print('Total time steps this exp:',tts)
    return center_return_list,tts

center_return_all = []
n_experiment = args.n_experiment
tts_all = 0
for i in range(n_experiment):
    random_seed = (i+1) * args.random_seed
    center_return_list,tts = ESPG(envname = args.envname,\
                              seed = random_seed,\
                              N_GEN = args.n_gen,\
                              POP_SIZE = args.pop_size,\
                              fitness_eval_episodes = args.fitness_eval_episodes,\
                              episode_length = args.episode_length,\
                              ALPHA = args.ALPHA,\
                              sigma = args.sigma,\
                              hidden_dim=args.hidden_dim,\
                              goal = args.goal,\
                              learning_rate=args.learning_rate,\
                              gamma = args.gamma,\
                              n_episode = args.n_episode,\
                              weight_decay=args.weight_decay,\
                              lam=args.lam,ppo_epoch=args.ppo_epoch,minibatch_size=args.minibatch_size
                              )
    center_return_all.append(center_return_list)
    tts_all += tts
tts_avg = int(tts_all/n_experiment)

training_log_all_experiment = np.array(center_return_all)
save_path = os.path.join('logs',args.exp_name+'_'+str(tts_avg))
np.savetxt(save_path,training_log_all_experiment)

