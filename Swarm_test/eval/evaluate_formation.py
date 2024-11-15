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
from algorithm.algorithms.maddpg import MADDPG
import json
import matplotlib.pyplot as plt
import pickle
import random

from llm.modules.framework.actions.rl_critic import RLCritic
import asyncio
from modules.framework.context import WorkflowContext
import argparse
from modules.prompt.user_requirements import get_user_commands
from modules.utils import root_manager

USE_CUDA = False 

def process_shape(shape_index, env, l_cells_input, grid_center_origins_input, binary_images_input, shape_bound_points_origins_input):

    env.env.l_cell = l_cells_input[shape_index]
    env.env.grid_center_origin = grid_center_origins_input[shape_index].T
    env.env.target_shape = binary_images_input[shape_index]
    env.env.shape_bound_points_origin = shape_bound_points_origins_input[shape_index]

    # # shape_scales = [1, 1, 1, 1, 1]
    # shape_scales = [1, 1, 1, 1, 1, 1, 1, 1]
    # # shape_scales = [1.2, 1.2, 1.2, 1.2, 1.2]
    # shape_scale = shape_scales[shape_index]
    # env.env.l_cell = shape_scale * env.env.l_cell
    # env.env.grid_center_origin = shape_scale * env.env.grid_center_origin
    # env.env.shape_bound_points_origin = shape_scale * env.env.shape_bound_points_origin

    # rand_angle = np.pi * np.random.uniform(-1, 1)
    rand_angle = 0
    rotate_matrix = np.array([[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]])
    env.env.grid_center_origin = np.dot(rotate_matrix, env.env.grid_center_origin)

    env.env.n_g = env.env.grid_center_origin.shape[1]

    # compute the collision avoidance distance
    # env.env.r_avoid = np.sqrt(4*env.env.n_g/(env.env.n_a*np.pi)) * env.env.l_cell
    print(env.env.r_avoid, env.env.d_sen)

    # env.env.d_sen = 0.5*shape_scale
    # env.env.d_sen = 0.4

    # randomize target shape's position
    rand_target_offset = np.random.uniform(-1.0, 1.0, (2, 1))   ################## domain generalization 4
    # rand_target_offset = np.zeros((2,1))
    env.env.grid_center = env.env.grid_center_origin.copy() + rand_target_offset
    env.env.shape_bound_points = np.hstack((env.env.shape_bound_points_origin[:2] + rand_target_offset[0,0], env.env.shape_bound_points_origin[2:] + rand_target_offset[1,0]))

    # env.env.p = np.random.uniform(-2, 2, (2, env.env.n_a))


def eval_rew_policy(curr_obs, maddpg, start_stop_num):
    rew_rs = []
    for act_x_i in np.arange(-1, 1, 0.05):
        rew_rs_list = []
        for act_y_i in np.arange(-1, 1, 0.05):
            obs_rs = np.vstack((curr_obs[:,[0]], np.array([[act_x_i], [act_y_i]])))
            # obs_rs = obs.copy()
            torch_obs_rs = torch.Tensor(obs_rs).requires_grad_(False)
            torch_rewards_rs = maddpg.step_rew(torch_obs_rs, start_stop_num) # shaped reward
            agent_rewards_rs = np.column_stack([rew.data.numpy() for rew in torch_rewards_rs])
            rew_rs_list.append(agent_rewards_rs[0,0])

        rew_rs.append(rew_rs_list)

    X, Y = np.meshgrid(np.arange(-1, 1, 0.05), np.arange(-1, 1, 0.05))
    Z = np.array(rew_rs)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surface_plot = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Adding labels
    ax.set_xlabel("act_x_i")
    ax.set_ylabel("act_y_i")
    ax.set_zlabel("rew_rs")
    plt.show()

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
    curr_run = '2024-11-12-12-37-39_sparse'

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
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep4001.pt'
    maddpg = MADDPG.init_from_save(run_dir)

    # llm_reward
    root_manager.update_root("./workspace/test")

    parser = argparse.ArgumentParser(description="Run simulation with custom parameters.")
    parser.add_argument("--interaction_mode", type=bool, default=False, help="Whether to run in interaction mode in analyze constraints.")
    cfgs = parser.parse_args()

    context = WorkflowContext()
    task = get_user_commands("formation")[0]
    context.command = task
    context.args = cfgs

    rl_critic = RLCritic("rl critic")

    ##
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

        episode_length = 20
        episode_reward = 0
        obs = env.reset()     
        # maddpg.prep_rollouts(device='cpu') 

        maddpg.scale_noise(0, 0)
        maddpg.reset_noise()

        # M_p, N_p = np.shape(env.p)     
        # M_v, N_v =np.shape(env.dp)
        # p_store = np.zeros((M_p, N_p, episode_length))       
        # dp_store = np.zeros((M_v, N_v, episode_length))

        shape_count = 0
        delete_count = 0
        # time_points = [0, 250, 550, 800, 1200]
        time_points = [0, 250, 550, 850, 1150, 1450, 1750, 2050]
        # time_points = [0, 250, 550, 850, 1150, 1450, 1750]

        agent_reward_list = []
        agent_obs_cat = None
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(episode_length):
            env.render()

            if et_index in time_points:
                process_shape(shape_count, env, l_cells, grid_center_origins, binary_images, shape_bound_points_origins)
                shape_count += 1
                coverage_rate = env.coverage_rate()
                uniformity_degree = env.distribution_uniformity()
                print('coverage rate: {:.4f}, distribution uniformity: {:.4f}'.format(coverage_rate, uniformity_degree))

            # if et_index == 170:
            #     delete_range = [-0.6, 0.0, -0.8, 0.5]
            #     x_min, x_max, y_min, y_max = delete_range
            #     indices_not_in_range = np.where((env.p[0, :] < x_min) | (env.p[0, :] > x_max) | (env.p[1, :] < y_min) | (env.p[1, :] > y_max))[0]
            #     env.env.init_after_vari_num(indices_not_in_range)
            #     delete_count += 1
            
            # p_store[:, :, et_index] = env.p             
            # dp_store[:, :, et_index] = env.dp

            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=False) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            #####--------------------------#####
            # eval_rew_policy(obs, maddpg, start_stop_num)
            # obs_rs = np.vstack((obs, agent_actions))
            # # obs_rs = obs.copy()
            # torch_obs_rs = torch.Tensor(obs_rs).requires_grad_(False)
            # torch_rewards_rs = maddpg.step_rew(torch_obs_rs, start_stop_num) # shaped reward
            # agent_rewards_rs = np.column_stack([rew.data.numpy() for rew in torch_rewards_rs])
            #####--------------------------#####

            # if delete_count == 1:
            #     agent_actions = agent_actions[:, indices_not_in_range]
            #     delete_count = 0

            # obtain  reward and next state
            next_obs, rewards, dones, infos = env.step(agent_actions)

            ##################################################################
            agent_reward_list.append(rewards[0,0])
            agent_obs_cat = obs[:,[0]] if agent_obs_cat is None else np.concatenate((agent_obs_cat, obs[:,[0]]), axis=1)
            ##################################################################

            # print(rewards)    
            obs = next_obs    

            episode_reward += np.mean(rewards)

        end_time_1 = time.time()

        # obtain the reward of agent_0 from llm
        asyncio.run(rl_critic.run(agent_obs_cat))
        agent_reward_list_from_llm = rl_critic.reward_list
        for i, (item_a, item_b) in enumerate(zip(agent_reward_list, agent_reward_list_from_llm)):
            if item_a != item_b:
                print(f"Elements differ at index {i}: agent_reward_list[{i}] = {item_a}, agent_reward_list_from_llm[{i}] = {item_b}")


        ########################### process data ###########################
        print("Episodes %i of %i, episode reward: %f, step time: %f" % (ep_index, cfg.n_episodes, episode_reward/episode_length, end_time_1 - start_time_1))

        # np.savez(os.path.join(results_dir, 'state_data.npz'), pos = p_store, vel = dp_store, t_step = et_index)

        ########################### plot ###########################
        log_dir = model_dir / curr_run / 'logs'
        with open(log_dir / 'summary.json', 'r') as f:
            data = json.load(f)

        # 提取 episode_reward 数据
        # episode_rewards_mean = data[str(log_dir) + '/agent/data/episode_reward_mean']
        # episode_rewards_std = data[str(log_dir) + '/agent/data/episode_reward_std']
        episode_rewards_mean_bar = data[str(log_dir) + '/agent/data/episode_reward_mean_bar']
        episode_rewards_std_bar = data[str(log_dir) + '/agent/data/episode_reward_std_bar']
        episode_rewards_mean_hat = data[str(log_dir) + '/agent/data/episode_reward_mean_hat']
        episode_rewards_std_hat = data[str(log_dir) + '/agent/data/episode_reward_std_hat']
        # 提取时间戳和奖励值
        timestamps1 = np.array([entry[1] for entry in episode_rewards_mean_bar])
        rewards_mean_bar = np.array([entry[2] for entry in episode_rewards_mean_bar])
        rewards_std_bar = np.array([entry[2] for entry in episode_rewards_std_bar])
        rewards_mean_hat = np.array([entry[2] for entry in episode_rewards_mean_hat])
        rewards_std_hat = np.array([entry[2] for entry in episode_rewards_std_hat])

        ############################################
        # loss_critic_rew = data[str(log_dir) + '/agent0/losses/vf_loss_rew']
        # loss_actor_rew = data[str(log_dir) + '/agent0/losses/pol_loss_rew']
        loss_critic = data[str(log_dir) + '/agent0/losses/vf_loss']
        loss_actor = data[str(log_dir) + '/agent0/losses/pol_loss']
        # # 提取时间戳和损失值
        timestamps2 = np.array([entry[1] for entry in loss_critic])
        # loss_critic_rew = np.array([entry[2] for entry in loss_critic_rew])
        # loss_actor_rew = np.array([entry[2] for entry in loss_actor_rew])
        loss_critic = np.array([entry[2] for entry in loss_critic])
        loss_actor = np.array([entry[2] for entry in loss_actor])
        # loss_critic = loss_critic[::10]
        # loss_actor = loss_actor[::10]

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps1, rewards_mean_bar, label='Episode Reward (sparse)')
        plt.fill_between(timestamps1, rewards_mean_bar - rewards_std_bar, rewards_mean_bar + rewards_std_bar, color='blue', alpha=0.2, label='Std')
        plt.plot(timestamps1, rewards_mean_hat, label='Episode Reward (total)')
        plt.fill_between(timestamps1, rewards_mean_hat - rewards_std_hat, rewards_mean_hat + rewards_std_hat, color='green', alpha=0.2, label='Std')
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
        # plt.figure(figsize=(8, 6))
        # plt.plot(timestamps2, loss_critic_rew, label='rew_critic loss')
        # # plt.fill_between(timestamps2, rewards_mean_bar - rewards_std_bar, rewards_mean_bar + rewards_std_bar, color='blue', alpha=0.2, label='Std')
        # plt.plot(timestamps2, loss_actor_rew, label='rew_actor loss')
        # # plt.fill_between(timestamps2, rewards_mean_hat - rewards_std_hat, rewards_mean_hat + rewards_std_hat, color='green', alpha=0.2, label='Std')
        # plt.xlabel('Step',fontsize=12)
        # plt.ylabel('Loss_rew',fontsize=12)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(True)
        # plt.title('Step Loss_rew Curve',fontsize=12)
        # plt.legend(loc='lower right',fontsize=12)
        # plt.savefig(os.path.join(results_dir, 'loss_rew.pdf'), format='pdf')
        # # plt.show()

        # 绘制曲线
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps2, loss_critic, label='critic loss')
        # plt.fill_between(timestamps2, rewards_mean_bar - rewards_std_bar, rewards_mean_bar + rewards_std_bar, color='blue', alpha=0.2, label='Std')
        plt.plot(timestamps2, loss_actor, label='actor loss')
        # plt.fill_between(timestamps2, rewards_mean_hat - rewards_std_hat, rewards_mean_hat + rewards_std_hat, color='green', alpha=0.2, label='Std')
        plt.xlabel('Step',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.title('Step Loss Curve',fontsize=12)
        plt.legend(loc='lower right',fontsize=12)
        plt.savefig(os.path.join(results_dir, 'loss.pdf'), format='pdf')
        plt.show()

if __name__ == '__main__':

    run(args) 

