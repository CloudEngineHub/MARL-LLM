import argparse
import torch
import time
import os
import sys

from torch._C import device
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_level_dir = os.path.join(current_dir, '..')
sys.path.append(upper_level_dir)
import numpy as np
import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg.flocking_mappo_cfg import gpsargs as args
from algorithm.algorithms.mappo import MAPPO
from pathlib import Path
from datetime import datetime
from algorithm.mappo_utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter

USE_CUDA = True 

def run(cfg):

    ## ======================================= record =======================================
    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run   
    log_dir = run_dir / 'logs'    
    os.makedirs(log_dir)    
    logger = SummaryWriter(str(log_dir)) 

    ## ======================================= Initialize =======================================
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # environment
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)

    start_stop_num=[slice(0, env.num_agents)]   

    # algorithm
    if USE_CUDA:
        device = torch.device("cuda:0")
        torch.set_num_threads(cfg.n_training_threads) 
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(cfg.n_training_threads) 

    mappo = MAPPO.init_from_env(env, cfg, device)

    # buffer
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cent_obs_dim = obs_dim + 4 * cfg.n_a if cfg.use_centralized_V else obs_dim
    agents_buffer = ReplayBuffer(cfg, env.n_a, obs_dim, cent_obs_dim, act_dim)      
    buffer_total=[agents_buffer]  

    # training index
    training_index = range(1)
    index_a = np.array([i for i in range(env.n_a)])

    p_store = []
    h_store = []

    ## ======================================= Training =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward_mean = 0
        episode_reward_std = 0

        obs, cent_obs = env.reset() 
        cent_obs = cent_obs if cfg.use_centralized_V else obs
        start_stop_num = [slice(0, env.n_a)]  
        agents_buffer.push_obs(start_stop_num[0], obs, cent_obs)
        rnn_states = np.zeros((env.n_a, cfg.recurrent_N, cfg.hidden_dim), dtype=np.float32)
        rnn_states_critic = np.zeros((env.n_a, cfg.recurrent_N, cfg.hidden_dim), dtype=np.float32)
        masks = np.ones((env.n_a, 1), dtype=np.float32)

        M_l, N_l = np.shape(env.p)     
        M_h, N_h = np.shape(env.heading)
        p_store = np.zeros((M_l, N_l, cfg.episode_length))       
        h_store = np.zeros((M_h, N_h, cfg.episode_length))

        if cfg.use_linear_lr_decay:
            for a in mappo.agents:
                a.lr_decay(ep_index, cfg.n_episodes)
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_i = 0
        mappo.prep_rollout()
        while et_i in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()
            
            # 获取 observation for per agent and convert to torch variable
            p_store[:, :, et_i] = env.p             
            h_store[:, :, et_i] = env.heading
 
            state_info = obs, cent_obs, rnn_states, rnn_states_critic, masks
            values, actions, action_log_probs, rnn_states, rnn_states_critic = mappo.step(state_info, start_stop_num, env.n_a, training_index, is_deterministic=False) 
            agent_actions = actions.T
  
            # obtain reward and next state
            next_obs, next_cent_obs, rewards, dones, infos = env.step(agent_actions)  
            next_cent_obs = next_cent_obs if cfg.use_centralized_V else next_obs
            start_stop_num = [slice(0, env.n_a)]

            if next_obs.shape[1] == agent_actions.shape[1]:
                store_info = next_obs, next_cent_obs, agent_actions, action_log_probs, rewards, dones, values, rnn_states, rnn_states_critic
                agents_buffer.push(store_info, start_stop_num[0], index_a, 0, 0)  

            obs = next_obs 
            cent_obs = next_cent_obs 
            masks = np.ones((env.n_a, 1), dtype=np.float32)

            episode_reward_mean += np.mean(rewards[:,start_stop_num[0]])
            episode_reward_std += np.std(rewards[:,start_stop_num[0]]) 

            et_i += 1
            # print(et_i, left_agents_buffer.step)

        end_time_1 = time.time()
        ########################### train ###########################
        start_time_2 = time.time()  
        for a_i in training_index:
            mappo.compute(buffer_total[a_i], a_i)
            train_info = mappo.train(buffer_total[a_i], a_i, True)
        
        buffer_total[0].reset()
        mappo.prep_rollout()
        end_time_2 = time.time()  

        ########################### process data ###########################
        if ep_index % 10 == 0:
            print("Episodes %i of %i, episode reward: %f, step time: %f, training time: %f" % (ep_index, cfg.n_episodes, episode_reward_mean/cfg.episode_length, end_time_1 - start_time_1, end_time_2 - start_time_2))

        if ep_index % cfg.save_interval == 0:
            logger.add_scalars('data', {'episode_reward_mean': episode_reward_mean/cfg.episode_length, 'episode_reward_std': episode_reward_std/cfg.episode_length}, ep_index)
            logger.add_scalars('train', {'value_loss': train_info['value_loss'], 'policy_loss': train_info['policy_loss'], 'ratio': train_info['ratio']}, ep_index)

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            mappo.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    mappo.save(run_dir / 'model.pt')

    # env.close()       
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    # for i in np.arange(1, 3, 0.3):
    #     args.k_1 = i
        run(args)  