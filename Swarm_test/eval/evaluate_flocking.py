import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..')) 
import numpy as np
import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg.flocking_cfg import gpsargs as args
from pathlib import Path
from algorithm.algorithms.maddpg import MADDPG
import json
import matplotlib.pyplot as plt

USE_CUDA = False 

def plot_policy(env, start_stop_num, maddpg, current_obs):
    scale_step = 0.2
    obs_manual = np.zeros((env.obs_dim_agent, 1))
    obs_manual[2:] = current_obs[2:,[0]]
    x_pos_range = np.arange(env.boundary_pos[0], env.boundary_pos[2] + scale_step, scale_step)
    y_pos_range = np.arange(env.boundary_pos[1], env.boundary_pos[3] - scale_step, -scale_step)
    action_x = np.zeros(len(x_pos_range) * len(y_pos_range))
    action_y = np.zeros(len(x_pos_range) * len(y_pos_range))
    count = 0
    for y_pos in y_pos_range:
        for x_pos in x_pos_range:
            obs_manual[0] = x_pos
            obs_manual[1] = y_pos
            torch_obs = torch.Tensor(obs_manual).requires_grad_(False)  
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=False) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
            action_x[count] = agent_actions[0]
            action_y[count] = agent_actions[1]
            count += 1

    count = 0
    fig = plt.figure(figsize=(16, 16))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height],projection = None)
    for y_pos in y_pos_range:
        for x_pos in x_pos_range:
            ax.quiver(x_pos, y_pos, action_x[count], action_y[count], scale=30, color='blue', width = 0.003)
            count += 1
    ax.axis('equal')
    ax.set_xlabel('X',fontsize=24)
    ax.set_ylabel('Y',fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    ax.set_title('Action', fontsize=24)

    plt.show()

def run(cfg, sta, t_id, test_num):
 
    ## ======================================= Initialize =======================================
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed)   
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads) 

    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = '2024-09-07-21-05-47'

    results_dir = os.path.join(model_dir, curr_run, 'results') 
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.video:
        args.video_path = results_dir + '/video.mp4'

    # environment
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)
    start_stop_num = [slice(0,env.num_agents)]

    # algorithm
    run_dir = model_dir / curr_run / 'model.pt'
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep801.pt'
    maddpg = MADDPG.init_from_save(run_dir)

    p_store = []
    dp_store = []
    torch_agent_actions=[]

    ## ======================================= Evaluation =======================================
    print('Step Starts...')
    for ep_index in range(0, 1, cfg.n_rollout_threads):

        episode_length = 500
        episode_reward = 0
        obs=env.reset()     
        # maddpg.prep_rollouts(device='cpu') 

        maddpg.scale_noise(0, 0)
        maddpg.reset_noise()

        M_p, N_p = np.shape(env.p)     
        M_v, N_v =np.shape(env.dp)
        p_store = np.zeros((M_p, N_p, episode_length))       
        dp_store = np.zeros((M_v, N_v, episode_length))
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(episode_length):
            if test_num == 1:
                env.render()

            # plot_policy(env, [slice(0,1)], maddpg, obs)
            
            p_store[:, :, et_index] = env.p             
            dp_store[:, :, et_index] = env.dp

            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=False) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # obtain  reward and next state
            next_obs, rewards, dones, infos = env.step(agent_actions)    
            obs = next_obs    

            episode_reward += np.mean(rewards) 

        end_time_1 = time.time()

        ########################### process data ###########################
        print("Episodes %i of %i, episode reward: %f, step time: %f" % (ep_index, cfg.n_episodes, episode_reward/episode_length, 
                end_time_1 - start_time_1))
        dist_epi = env.dist_metric(p_store, episode_length, env.num_agents)
        order_epi = env.order_metric(p_store, dp_store, episode_length, env.num_agents)

        sta[t_id, 0] = dist_epi
        sta[t_id, 1] = order_epi

        np.savez(os.path.join(results_dir, 'state_data.npz'), pos = p_store, vel = dp_store, t_step = et_index)

        ########################### plot ###########################
        if test_num == 1:
            log_dir = model_dir / curr_run / 'logs'
            with open(log_dir / 'summary.json', 'r') as f:
                data = json.load(f)

            # 提取 episode_reward 数据
            episode_rewards_mean = data[str(log_dir) + '/agent/data/episode_reward_mean']
            episode_rewards_std = data[str(log_dir) + '/agent/data/episode_reward_std']

            # 提取时间戳和奖励值
            timestamps = np.array([entry[1] for entry in episode_rewards_mean])
            rewards_mean = np.array([entry[2] for entry in episode_rewards_mean])
            rewards_std = np.array([entry[2] for entry in episode_rewards_std])

            # 绘制曲线
            plt.figure(figsize=(8, 6))
            plt.plot(timestamps, rewards_mean, label='Episode Reward')
            plt.fill_between(timestamps, rewards_mean - rewards_std, rewards_mean + rewards_std, color='blue', alpha=0.3, label='Std')
            plt.xlabel('Episode',fontsize=12)
            plt.ylabel('Reward',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
            plt.title('Episode Reward Curve',fontsize=12)
            plt.legend(loc='lower right',fontsize=12)
            plt.savefig(os.path.join(results_dir, 'reward_curve.pdf'), format='pdf')
            plt.show()

if __name__ == '__main__':

    test_num = 1
    statistics = np.zeros((test_num, 3))
    for test_id in range(test_num):
        args.seed = test_id + 200
        run(args, statistics, test_id, test_num) 

    mean_dist = np.sum(statistics[:,0]) / test_num
    mean_dist_std = np.std(statistics[:,0])
    mean_order = np.sum(statistics[:,1]) / test_num
    mean_order_std = np.std(statistics[:,1])
    print('mean dist metric: {:.1%}±{:.1%}, mean order metric: {:.1%}±{:.1%} mean time: {:.2f}±{:.2f} s'.format(mean_dist, mean_dist_std, mean_order, mean_order_std, 0,0))
