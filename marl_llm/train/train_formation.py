import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils.buffer_agent import ReplayBufferAgent
from tensorboardX import SummaryWriter
from datetime import datetime
from algorithm.algorithms.maddpg import MADDPG
import random

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
    random.seed(cfg.seed)  
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # environment
    scenario_name = 'FormationSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FormationSwarmWrapper(base_env, args)
    start_stop_num = [slice(0, env.num_agents)]  

    # algorithm
    adversary_alg = None
    maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg, tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
                                  hidden_dim=cfg.hidden_dim, device=cfg.device, epsilon=cfg.epsilon, noise=cfg.noise_scale, name=cfg.env_name)
    # last_run = '2025-01-08-19-57-46'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save(last_run_dir)

    # buffer
    agent_buffer = [ReplayBufferAgent(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                start_stop_index=start_stop_num[0])]  

    torch_agent_actions=[]

    ## ======================================= Training =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0
        obs = env.reset()  
        start_stop_num = [slice(0, env.n_a)]    
        maddpg.prep_rollouts(device='cpu') 
      
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()

            # obtain action
            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # environment step
            next_obs, rewards, dones, _, agent_actions_prior = env.step(agent_actions)

            # store experience
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones, start_stop_num[0], agent_actions_prior)
            obs = next_obs  

            # record rewards
            episode_reward_mean_bar += np.mean(rewards) # sparse rewards
            episode_reward_std_bar += np.std(rewards) # sparse rewards

        end_time_1 = time.time()
        ########################### train ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)
        for _ in range(20):      
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(cfg.batch_size, to_gpu=True if cfg.device == 'gpu' else False, is_prior=True if cfg.training_method == 'llm_rl' else False)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, a_i, acs_prior_sample, env.alpha, logger=logger)     # parameter update 
            maddpg.update_all_targets()
            
        maddpg.prep_rollouts(device='cpu')    
        maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)
        env.env.alpha = 0.1
        end_time_2 = time.time()

        ########################### process data ###########################
        if ep_index % 10 == 0:
            print("Episodes %i of %i, agent num: %i, episode reward: %f, step time: %f, training time: %f" % 
                  (ep_index, cfg.n_episodes, env.n_a, episode_reward_mean_bar/cfg.episode_length, end_time_1 - start_time_1, end_time_2 - start_time_2))
            
        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0
            logger.add_scalars('agent/data', {'episode_reward_mean_bar': episode_reward_mean_bar/cfg.episode_length, 
                                              'episode_reward_std_bar': episode_reward_std_bar/cfg.episode_length, 
                                              'ALIGN_epi': ALIGN_epi}, ep_index)

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    maddpg.prep_training(device=cfg.device)
    maddpg.save(run_dir / 'model.pt')
      
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    plt.close('all')

if __name__ == '__main__':
    run(args)  
