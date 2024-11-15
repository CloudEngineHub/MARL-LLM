import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..')) 
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_mappo_cfg import gpsargs as args
from pathlib import Path
from algorithm.algorithms.mappo import MAPPO
import json
import matplotlib.pyplot as plt
import random
import pickle

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
    
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads) 

    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = '2024-11-01-20-51-14'

    results_dir = os.path.join(model_dir, curr_run, 'results') 
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.video:
        args.video_path = results_dir + '/video.mp4'

    # environment
    scenario_name = 'FormationSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FormationSwarmWrapper(base_env, args)
    start_stop_num = [slice(0, env.num_agents)]   

    # algorithm
    run_dir = model_dir / curr_run / 'model.pt'
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep6801.pt'
    mappo = MAPPO.init_from_save(run_dir)
   
    # training index
    training_index = range(1)

    p_store = []
    dp_store = []
    torch_agent_actions=[]

    with open(args.results_file, 'rb') as f:
        loaded_results = pickle.load(f)

    l_cells = loaded_results['l_cell']
    grid_center_origins = loaded_results['grid_coords']
    binary_images = loaded_results['binary_image']
    shape_bound_points_origins = loaded_results['shape_bound_points']

    ## ======================================= Evaluation =======================================
    print('Step Starts...')
    for ep_index in range(0, 1, cfg.n_rollout_threads):

        episode_length = 2
        episode_reward_mean = 0

        obs = env.reset() 
        cent_obs = obs if cfg.use_centralized_V else obs  
        masks = np.ones((env.n_a, 1), dtype=np.float32)

        start_stop_num = [slice(0, env.n_a)]

        # M_l, N_l = np.shape(env.p)     
        # M_v, N_v =np.shape(env.dp)
        # p_store = np.zeros((M_l, N_l, episode_length))       
        # dp_store = np.zeros((M_v, N_v, episode_length))

        shape_count = 0
        delete_count = 0
        # time_points = [0, 120, 550, 800, 1200]
        time_points = [0, 120, 550, 800, 1200, 1500, 1800]
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_index = 0
        while et_index in range(episode_length):
            env.render()

            if et_index in time_points:
                process_shape(shape_count, env, l_cells, grid_center_origins, binary_images, shape_bound_points_origins)
                shape_count += 1
                coverage_rate = env.coverage_rate()
                uniformity_degree = env.distribution_uniformity()
                print('coverage rate: {:.4f}, distribution uniformity: {:.4f}'.format(coverage_rate, uniformity_degree))

            state_info = obs, cent_obs, masks
            _, actions, _ = mappo.step(state_info, start_stop_num, env.n_a, training_index, is_deterministic=True) 
            agent_actions = actions.T
  
            # obtain reward and next state
            next_obs, rewards, dones, infos = env.step(agent_actions)  
            next_cent_obs = next_obs if cfg.use_centralized_V else next_obs
                
            obs = next_obs 
            cent_obs = next_cent_obs 
            masks = np.ones((env.n_a, 1), dtype=np.float32)

            episode_reward_mean += np.mean(rewards[:,start_stop_num[0]])

            et_index += 1

        end_time_1 = time.time()
        env.close()

        ########################### process data ###########################
        print("Episode reward: %f, step time: %f" % (episode_reward_mean/episode_length, end_time_1 - start_time_1))

        ########################### plot ###########################
        log_dir = model_dir / curr_run / 'logs'
        with open(log_dir / 'summary.json', 'r') as f:
            data = json.load(f)

        # 提取 episode_reward 数据
        episode_rewards_mean = data[str(log_dir) + '/data/episode_reward_mean_bar']
        episode_rewards_std = data[str(log_dir) + '/data/episode_reward_std_bar']
        episode_value_loss = data[str(log_dir) + '/train/value_loss']
        episode_policy_loss = data[str(log_dir) + '/train/policy_loss']
        episode_policy_ratio = data[str(log_dir) + '/train/ratio']

        # 提取时间戳和奖励值
        timestamps = np.array([entry[1] for entry in episode_rewards_mean])
        rewards_mean = np.array([entry[2] for entry in episode_rewards_mean])
        rewards_std = np.array([entry[2] for entry in episode_rewards_std])

        value_loss = np.array([entry[2] for entry in episode_value_loss])
        policy_loss = np.array([entry[2] for entry in episode_policy_loss])
        policy_ratio = np.array([entry[2] for entry in episode_policy_ratio])

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps, rewards_mean, c=(0.0, 0.392, 0.0), label="Episode Reward")
        plt.fill_between(timestamps, rewards_mean - rewards_std, rewards_mean + rewards_std, color=(0.0, 0.392, 0.0), alpha=0.2)
        plt.xlabel('Episode',fontsize=12)
        plt.ylabel('Reward',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.title('Episode Reward Curve',fontsize=12)
        plt.legend(loc='lower right',fontsize=12)
        plt.savefig(os.path.join(results_dir, 'reward_curve.png'), format='png')

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps, value_loss, c=(0.0, 0.392, 0.0), label="Value loss")
        plt.plot(timestamps, policy_loss, c=(1, 0.388, 0.278), label="Policy loss")
        plt.plot(timestamps, policy_ratio, c=(0.5, 0.188, 0.278), label="Policy ratio")
        # plt.fill_between(timestamps, rewards_mean_l - rewards_std_l, rewards_mean_l + rewards_std_l, color=(0.0, 0.392, 0.0), alpha=0.2)
        # plt.fill_between(timestamps, rewards_mean_r - rewards_std_r, rewards_mean_r + rewards_std_r, color=(1, 0.388, 0.278), alpha=0.2)
        # plt.xlabel('Episode',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.title('Training loss',fontsize=12)
        plt.legend(loc='right',fontsize=12)
        plt.savefig(os.path.join(results_dir, 'loss_curve.png'), format='png')
        plt.show()

if __name__ == '__main__':

    # test_num = 50
    # statistics = np.zeros((test_num, 3))
    # for test_id in range(test_num):
    #     args.seed = test_id + 200
    run(args) 

    # mean_dist = np.sum(statistics[:,0]) / test_num
    # mean_dist_std = np.std(statistics[:,0])
    # mean_order = np.sum(statistics[:,1]) / test_num
    # mean_order_std = np.std(statistics[:,1])
    # print('mean dist metric: {:.1%}±{:.1%}, mean order metric: {:.1%}±{:.1%} mean time: {:.2f}±{:.2f} s'.format(mean_dist, mean_dist_std, mean_order, mean_order_std, 0,0)) 
