import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..')) 
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils.buffer_expert import ReplayBuffer
import json
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime

USE_CUDA = False 

def process_shape(shape_index, env, l_cells_input, grid_center_origins_input, binary_images_input, shape_bound_points_origins_input):

    env.env.l_cell = l_cells_input[shape_index]
    env.env.grid_center_origin = grid_center_origins_input[shape_index].T
    env.env.target_shape = binary_images_input[shape_index]
    env.env.shape_bound_points_origin = shape_bound_points_origins_input[shape_index]

    # shape_scales = [1, 1, 1, 1, 1]
    shape_scales = [1, 1, 1, 1, 1, 1, 1]
    # shape_scales = [1.2, 1.2, 1.2, 1.2, 1.2]
    shape_scale = shape_scales[shape_index]
    env.env.l_cell = shape_scale * env.env.l_cell
    env.env.grid_center_origin = shape_scale * env.env.grid_center_origin
    env.env.shape_bound_points_origin = shape_scale * env.env.shape_bound_points_origin

    # rand_angle = np.pi * np.random.uniform(-1, 1)
    rand_angle = 0
    rotate_matrix = np.array([[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]])
    env.env.grid_center_origin = np.dot(rotate_matrix, env.env.grid_center_origin)

    env.env.n_g = env.env.grid_center_origin.shape[1]

    # compute the collision avoidance distance
    # env.env.r_avoid = np.sqrt(4*env.env.n_g/(env.env.n_a*np.pi)) * env.env.l_cell
    print(env.env.r_avoid)

    # env.env.d_sen = 0.5*shape_scale
    env.env.d_sen = 0.4

    # randomize target shape's position
    rand_target_offset = np.random.uniform(-1.0, 1.0, (2, 1))   ################## domain generalization 4
    # rand_target_offset = np.zeros((2,1))
    env.env.grid_center = env.env.grid_center_origin.copy() + rand_target_offset
    env.env.shape_bound_points = np.hstack((env.env.shape_bound_points_origin[:2] + rand_target_offset[0,0], env.env.shape_bound_points_origin[2:] + rand_target_offset[1,0]))

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
    expert_buffer = ReplayBuffer(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                start_stop_index=start_stop_num)

    ## ======================================= Evaluation =======================================
    print('Step Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward = 0
        obs = env.reset() 
        start_stop_num = slice(0, env.n_a)
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 50 == 0:
                env.render()

            agent_actions = np.zeros((2, env.n_a))
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
    run(args) 

