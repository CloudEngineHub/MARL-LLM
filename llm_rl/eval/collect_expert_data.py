import torch
import time
import os
import sys
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils.buffer_expert import ReplayBufferExpert
import json
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime

def run(cfg):
 
    ## ======================================= Initialize =======================================
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)  
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed) 

    # make the directory
    model_dir = './' / Path('./eval/expert_data')
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run      
    os.makedirs(run_dir)

    if args.video:
        args.video_path = run_dir + '/video.mp4'

    # environment
    scenario_name = 'FormationSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FormationSwarmWrapper(base_env, args)
    start_stop_num = slice(0, env.num_agents)

    # expert buffer
    expert_buffer = ReplayBufferExpert(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                start_stop_index=start_stop_num)

    ## ======================================= Evaluation =======================================
    print('Step Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward = 0
        obs = env.reset() 
        agent_actions = np.zeros((2, env.n_a))
        start_stop_num = slice(0, env.n_a)
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            # if ep_index % 50 == 0:
            #     env.render()

            next_obs, rewards, dones, infos, agent_actions = env.step(agent_actions)
            expert_buffer.push(obs, agent_actions, next_obs, dones, start_stop_num) 
            obs = next_obs    

            episode_reward += np.mean(rewards) 

        coverage_rate = env.coverage_rate()
        uniformity_degree = env.distribution_uniformity()
        print('coverage rate: {:.4f}, distribution uniformity: {:.4f}'.format(coverage_rate, uniformity_degree))

        end_time_1 = time.time()

        ########################### process data ###########################
        print("Episodes %i / %i, episode reward: %f, filling_length %i / %i, step time: %f" % (ep_index, cfg.n_episodes, episode_reward/cfg.episode_length, 
                                                                                               expert_buffer.filled_i, expert_buffer.total_length, end_time_1 - start_time_1))
    # save expert data
    expert_buffer.save(run_dir)

if __name__ == '__main__':
    args.agent_strategy = 'rule'
    args.n_episodes = 500 # 300
    args.buffer_length = int(1e5) # total length = 50 x 3e4 = 1e6
    run(args) 

