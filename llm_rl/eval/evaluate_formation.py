import torch
import time
import os
import sys
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_cfg import gpsargs as args
from pathlib import Path
from algorithm.algorithms.maddpg import MADDPG
import json
import matplotlib.pyplot as plt
import pickle
import random

USE_CUDA = False 

def process_shape(shape_index, env, l_cells_input, grid_center_origins_input, binary_images_input, shape_bound_points_origins_input, rand_target_offset_his):

    env.env.l_cell = l_cells_input[shape_index]
    env.env.grid_center_origin = grid_center_origins_input[shape_index].T
    env.env.target_shape = binary_images_input[shape_index]
    env.env.shape_bound_points_origin = shape_bound_points_origins_input[shape_index]

    rand_angle = 0
    rotate_matrix = np.array([[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]])
    env.env.grid_center_origin = np.dot(rotate_matrix, env.env.grid_center_origin)

    env.env.n_g = env.env.grid_center_origin.shape[1]

    # randomize target shape's position
    # rand_target_offset = np.random.uniform(-1.0, 1.0, (2, 1))   ################## domain generalization 4
    rand_target_offset = np.zeros((2, 1))

    if rand_target_offset_his == None:
        rand_target_offset_his = rand_target_offset.T
    else:
        rand_target_offset_his = np.concatenate((rand_target_offset_his, rand_target_offset.T))
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

    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = '2025-01-19-15-58-03'

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
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep2801.pt'
    maddpg = MADDPG.init_from_save(run_dir)

    ##
    rand_target_offset_his = None
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

        episode_length = 300
        episode_reward = 0
        obs = env.reset()     
        # maddpg.prep_rollouts(device='cpu') 

        maddpg.scale_noise(0, 0)
        maddpg.reset_noise()

        shape_count = 4
        time_interval = 300
        time_points = [0] + [i for i in range(300, (len(l_cells) - 1) * time_interval + 301, time_interval)]
        calculate_time_points = [x - 1 for x in time_points[1:]]
        results_list = []

        M_p, N_p = np.shape(env.p)     
        M_v, N_v =np.shape(env.dp)
        p_store = np.zeros((M_p, N_p, episode_length))       
        dp_store = np.zeros((M_v, N_v, episode_length))
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(episode_length):
            env.render()

            p_store[:, :, et_index] = env.p             
            dp_store[:, :, et_index] = env.dp

            if et_index in time_points:
                process_shape(shape_count, env, l_cells, grid_center_origins, binary_images, shape_bound_points_origins, rand_target_offset_his)
                shape_count += 1
            
            coverage_rate = env.coverage_rate()
            uniformity_degree = env.distribution_uniformity()
            voronoi_uniformity = env.voronoi_based_uniformity()
            print('coverage rate: {:.4f}, distribution uniformity: {:.4f}, voronoi_uniformity: {:.4f} of shape {:d}'.format(coverage_rate, uniformity_degree, voronoi_uniformity, shape_count))
            results_list.append({
                'coverage_rate': coverage_rate,
                'uniformity_degree': uniformity_degree,
                'voronoi_uniformity': voronoi_uniformity,
                'et_index': et_index,
                'shape_count': shape_count
            })

            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=False) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # obtain  reward and next state
            next_obs, rewards, dones, infos, _ = env.step(agent_actions)

            # print(rewards)    
            obs = next_obs    

            episode_reward += np.mean(rewards)

        end_time_1 = time.time()

        ########################### process data ###########################
        print("Episodes %i of %i, episode reward: %f, step time: %f" % (ep_index, cfg.n_episodes, episode_reward/episode_length, end_time_1 - start_time_1))

        ########################### store metric data ###########################
        # 保存列表到文件
        file_path = os.path.join(results_dir, 'metrics.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(results_list, f)

        # 保存到文件中
        rand_target_pos_dir = os.path.join(results_dir, 'rand_target_offset_log.npy')
        np.save(rand_target_pos_dir, rand_target_offset_his)

        # 保存状态数据
        np.savez(os.path.join(results_dir, 'state_data.npz'), pos = p_store, vel = dp_store, t_step = et_index)

        ########################### plot training data ###########################
        log_dir = model_dir / curr_run / 'logs'
        with open(log_dir / 'summary.json', 'r') as f:
            data = json.load(f)

        # 提取 episode_reward 数据
        episode_rewards_mean_bar = data[str(log_dir) + '/agent/data/episode_reward_mean_bar']
        episode_rewards_std_bar = data[str(log_dir) + '/agent/data/episode_reward_std_bar']
        loss_critic = data[str(log_dir) + '/agent0/losses/vf_loss']
        loss_actor = data[str(log_dir) + '/agent0/losses/pol_loss']
        loss_regular = data[str(log_dir) + '/agent0/losses/regularization_loss']
        if cfg.training_method == 'irl':
            loss_discriminator = data[str(log_dir) + '/agent0/losses/loss_discriminator']
            accuracy_pi = data[str(log_dir) + '/agent0/losses/accuracy_pi']
            accuracy_exp = data[str(log_dir) + '/agent0/losses/accuracy_exp']

        # 提取时间戳和奖励值
        timestamps1 = np.array([entry[1] for entry in episode_rewards_mean_bar])
        rewards_mean_bar = np.array([entry[2] for entry in episode_rewards_mean_bar])
        rewards_std_bar = np.array([entry[2] for entry in episode_rewards_std_bar])

        timestamps2 = np.array([entry[1] for entry in loss_critic])
        loss_critic = np.array([entry[2] for entry in loss_critic])
        loss_actor = np.array([entry[2] for entry in loss_actor])
        loss_regular = np.array([entry[2] for entry in loss_regular])

        if cfg.training_method == 'irl':
            timestamps3 = np.array([entry[1] for entry in loss_discriminator])
            loss_discriminator = np.array([entry[2] for entry in loss_discriminator])
            accuracy_pi = np.array([entry[2] for entry in accuracy_pi])
            accuracy_exp = np.array([entry[2] for entry in accuracy_exp])

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps1, rewards_mean_bar, label='Episode Reward (sparse)')
        plt.fill_between(timestamps1, rewards_mean_bar - rewards_std_bar, rewards_mean_bar + rewards_std_bar, color='blue', alpha=0.2, label='Std')
        plt.xlabel('Episode',fontsize=12)
        plt.ylabel('Reward',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.title('Episode Reward Curve',fontsize=12)
        plt.legend(loc='lower right',fontsize=12)
        plt.savefig(os.path.join(results_dir, 'reward_curve.pdf'), format='pdf')
        # plt.show()

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps2, loss_critic, label='critic loss')
        plt.plot(timestamps2, loss_actor, label='actor loss')
        plt.plot(timestamps2, loss_regular, label='regularization loss')
        plt.xlabel('Step',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.title('Step Loss Curve',fontsize=12)
        plt.legend(loc='lower right',fontsize=12)
        plt.savefig(os.path.join(results_dir, 'loss.pdf'), format='pdf')
        
        if cfg.training_method == 'irl':
            # 绘制曲线
            plt.figure(figsize=(8, 6))
            plt.plot(timestamps3, loss_discriminator, label='discriminator loss')
            plt.xlabel('Step',fontsize=12)
            plt.ylabel('Loss',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
            plt.title('Step Loss Curve',fontsize=12)
            plt.legend(loc='lower right',fontsize=12)
            plt.savefig(os.path.join(results_dir, 'loss_discriminator.pdf'), format='pdf')
            # plt.show()

            # 绘制曲线
            plt.figure(figsize=(8, 6))
            plt.plot(timestamps3, accuracy_pi, label='accuracy_pi')
            plt.plot(timestamps3, accuracy_exp, label='accuracy_exp')
            plt.xlabel('Step',fontsize=12)
            plt.ylabel('Accuracy',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
            plt.title('Discriminator Accuracy Curve',fontsize=12)
            plt.legend(loc='lower right',fontsize=12)
            plt.savefig(os.path.join(results_dir, 'accuracy.pdf'), format='pdf')
        
        plt.show()

if __name__ == '__main__':
    run(args) 

