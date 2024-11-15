import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..')) 
import numpy as np
import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg.flocking_mappo_cfg import gpsargs as args
from pathlib import Path
from algorithm.algorithms.mappo import MAPPO
import json
import matplotlib.pyplot as plt

USE_CUDA = False 

def run(cfg, sta, t_id, test_num):
 
    ## ======================================= Initialize =======================================
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed)   
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    if not USE_CUDA:
        torch.set_num_threads(cfg.n_training_threads) 

    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = '2024-07-16-21-10-33'

    results_dir = os.path.join(model_dir, curr_run, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.video:
        args.video_path = results_dir + '/video.mp4'

    # environment
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)
    start_stop_num=[slice(0, env.num_agents)]   

    # algorithm
    run_dir = model_dir / curr_run / 'model.pt'
    # run_dir = model_dir / curr_run / 'incremental' / 'model_ep1601.pt'
    mappo = MAPPO.init_from_save(run_dir)
   
    # training index
    training_index = range(1)

    p_store = []
    dp_store = []

    ## ======================================= Evaluation =======================================
    print('Step Starts...')
    for ep_index in range(0, 1, cfg.n_rollout_threads):

        episode_length = 500
        episode_reward_mean = 0

        obs, cent_obs = env.reset() 
        cent_obs = cent_obs if cfg.use_centralized_V else obs
        start_stop_num = [slice(0, env.n_a)]  
        rnn_states = np.zeros((env.n_a, cfg.recurrent_N, cfg.hidden_dim), dtype=np.float32)
        rnn_states_critic = np.zeros((env.n_a, cfg.recurrent_N, cfg.hidden_dim), dtype=np.float32)
        masks = np.ones((env.n_a, 1), dtype=np.float32)

        M_l, N_l = np.shape(env.p)     
        M_v, N_v =np.shape(env.dp)
        p_store = np.zeros((M_l, N_l, episode_length))       
        dp_store = np.zeros((M_v, N_v, episode_length))
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_index = 0
        while et_index in range(episode_length):
            if test_num == 1:
                env.render()

            p_store[:, :, et_index] = env.p             
            dp_store[:, :, et_index] = env.dp

            state_info = obs, cent_obs, rnn_states, rnn_states_critic, masks
            _, actions, _, rnn_states, rnn_states_critic = mappo.step(state_info, start_stop_num, env.n_a, training_index, is_deterministic=True) 
            agent_actions = actions.T
  
            # obtain reward and next state
            next_obs, next_cent_obs, rewards, dones, infos = env.step(agent_actions)  
            next_cent_obs = next_cent_obs if cfg.use_centralized_V else next_obs
            start_stop_num = [slice(0, env.n_a)]
                
            obs = next_obs 
            cent_obs = next_cent_obs 
            masks = np.ones((env.n_a, 1), dtype=np.float32)

            episode_reward_mean += np.mean(rewards[:,start_stop_num[0]])

            et_index += 1

        end_time_1 = time.time()
        env.close()

        ########################### process data ###########################
        print("Episode reward: %f, step time: %f" % (episode_reward_mean/episode_length, end_time_1 - start_time_1))
        dist_epi = env.dist_metric(p_store, episode_length, env.num_agents)
        order_epi = env.order_metric(p_store, dp_store, episode_length, env.num_agents)

        sta[t_id, 0] = dist_epi
        sta[t_id, 1] = order_epi

        ########################### plot ###########################
        if test_num == 1:
            log_dir = model_dir / curr_run / 'logs'
            with open(log_dir / 'summary.json', 'r') as f:
                data = json.load(f)

            # 提取 episode_reward 数据
            episode_rewards_mean = data[str(log_dir) + '/data/episode_reward_mean']
            episode_rewards_std = data[str(log_dir) + '/data/episode_reward_std']
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
            plt.figure(figsize=(18, 12))
            plt.plot(timestamps, rewards_mean, c=(0.0, 0.392, 0.0), label="Episode Reward")
            plt.fill_between(timestamps, rewards_mean - rewards_std, rewards_mean + rewards_std, color=(0.0, 0.392, 0.0), alpha=0.2)
            plt.xlabel('Episode',fontsize=25)
            plt.ylabel('Reward',fontsize=25)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.grid(True)
            plt.title('Episode Reward Curve',fontsize=25)
            plt.legend(loc='lower right',fontsize=25)
            plt.savefig(os.path.join(results_dir, 'reward_curve.png'), format='png')

            # 绘制曲线
            plt.figure(figsize=(18, 12))
            plt.plot(timestamps, value_loss, c=(0.0, 0.392, 0.0), label="Value loss")
            plt.plot(timestamps, policy_loss, c=(1, 0.388, 0.278), label="Policy loss")
            plt.plot(timestamps, policy_ratio, c=(0.5, 0.188, 0.278), label="Policy ratio")
            # plt.fill_between(timestamps, rewards_mean_l - rewards_std_l, rewards_mean_l + rewards_std_l, color=(0.0, 0.392, 0.0), alpha=0.2)
            # plt.fill_between(timestamps, rewards_mean_r - rewards_std_r, rewards_mean_r + rewards_std_r, color=(1, 0.388, 0.278), alpha=0.2)
            # plt.xlabel('Episode',fontsize=25)
            plt.ylabel('Loss',fontsize=25)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.grid(True)
            plt.title('Training loss',fontsize=25)
            plt.legend(loc='right',fontsize=25)
            plt.savefig(os.path.join(results_dir, 'loss_curve.png'), format='png')
            plt.show()

if __name__ == '__main__':

    test_num = 50
    statistics = np.zeros((test_num, 3))
    for test_id in range(test_num):
        args.seed = test_id + 200
        run(args, statistics, test_id, test_num) 

    mean_dist = np.sum(statistics[:,0]) / test_num
    mean_dist_std = np.std(statistics[:,0])
    mean_order = np.sum(statistics[:,1]) / test_num
    mean_order_std = np.std(statistics[:,1])
    print('mean dist metric: {:.1%}±{:.1%}, mean order metric: {:.1%}±{:.1%} mean time: {:.2f}±{:.2f} s'.format(mean_dist, mean_dist_std, mean_order, mean_order_std, 0,0)) 
