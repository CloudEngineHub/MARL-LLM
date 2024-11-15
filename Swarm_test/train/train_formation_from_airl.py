import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_level_dir = os.path.join(current_dir, '..')
sys.path.append(upper_level_dir)
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils.buffer_variable_num import ReplayBuffer as ReplayBufferAgent
from algorithm.utils.buffer_expert import ReplayBuffer as ReplayBufferExpert
from tensorboardX import SummaryWriter
from datetime import datetime
from algorithm.algorithms.maddpg import MADDPG
from algorithm.algorithms.airl import AIRL
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

    # expert buffer
    expert_buffer = ReplayBufferExpert(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                       start_stop_index=start_stop_num[0])
    # agent buffer
    agent_buffer = [ReplayBufferAgent(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                      start_stop_index=start_stop_num[0])]    
    
    # algorithm
    adversary_alg = None
    # maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg, tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
    #                               hidden_dim=cfg.hidden_dim, device=cfg.device, epsilon=cfg.epsilon, noise=cfg.noise_scale, name=cfg.env_name)
    last_run = '2024-11-05-20-59-07'
    last_run_dir = model_dir / last_run / 'model.pt'
    maddpg = MADDPG.init_from_save(last_run_dir)

    # load airl
    # model_dir = './' / Path('./models') / cfg.env_name 
    # curr_run = '2024-11-05-20-59-07'
    # run_dir_airl = model_dir / curr_run / 'discriminator.pt'
    # airl = AIRL(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=cfg.hidden_dim, hidden_num=3, lr_discriminator=cfg.lr_discriminator, 
    #             expert_buffer=expert_buffer, device=cfg.device)
    # airl.load(run_dir_airl)

    p_store = []
    dp_store = []
    torch_agent_actions = []
    expert_weight = 1

    ## ======================================= Training =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0
        episode_reward_mean_hat = 0
        episode_reward_std_hat = 0
        obs = env.reset()  
        start_stop_num = [slice(0, env.n_a)]    
        maddpg.prep_rollouts(device='cpu') 
      
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        agent_rewards_rs = np.zeros((1, env.n_a))

        # M_p, N_p = np.shape(env.p)     
        # M_v, N_v =np.shape(env.dp)
        # p_store = np.zeros((M_p, N_p, cfg.episode_length))       
        # dp_store = np.zeros((M_v, N_v, cfg.episode_length))
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()
            
            # p_store[:, :, et_index] = env.p             
            # dp_store[:, :, et_index] = env.dp

            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, torch_log_pis = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
            log_pis = np.column_stack([log_pi.data.numpy() for log_pi in torch_log_pis])

            next_obs, rewards, dones, infos = env.step(agent_actions) # sparse rewards
            agent_buffer[0].push(obs, agent_actions, log_pis, rewards, next_obs, dones, start_stop_num[0]) # all rewards, shape rewards, sparse rewards
            obs = next_obs  

            # record rewards
            episode_reward_mean_bar += np.mean(rewards) # sparse rewards
            episode_reward_std_bar += np.std(rewards) # sparse rewards
            episode_reward_mean_hat += np.mean(rewards + agent_rewards_rs) # total rewards
            episode_reward_std_hat += np.std(rewards + agent_rewards_rs) # total rewards

        end_time_1 = time.time()
        ########################### train ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)
        for _ in range(20):      
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(cfg.batch_size, to_gpu=True if cfg.device == 'gpu' else False)  
                    obs_sample, acs_sample, log_pis_sample, rew_sample, next_obs_sample, dones_sample = sample
                    # if cfg.use_shaping_reward:
                    #     rew_sample += expert_weight * airl.discriminator.calculate_reward(obs_sample, acs_sample, log_pis_sample, next_obs_sample, dones_sample)
                    maddpg.update(obs_sample, acs_sample, rew_sample, next_obs_sample, dones_sample, a_i, logger=logger)     # parameter update  
            maddpg.update_all_targets()
            
        maddpg.prep_rollouts(device='cpu')    
        maddpg.noise = max(0.4, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)
        # expert_weight = max(0, expert_weight - 3/cfg.n_episodes)
        end_time_2 = time.time()

        ########################### process data ###########################
        if ep_index % 10 == 0:
            print("Episodes %i of %i, agent num: %i, episode reward (sparse): %f, episode reward (total): %f, step time: %f, training time: %f" % 
                  (ep_index, cfg.n_episodes, env.n_a, episode_reward_mean_bar/cfg.episode_length, episode_reward_mean_hat/cfg.episode_length, 
                  end_time_1 - start_time_1, end_time_2 - start_time_2))
        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0
            logger.add_scalars('agent/data', {'episode_reward_mean_bar': episode_reward_mean_bar/cfg.episode_length, 'episode_reward_mean_hat': episode_reward_mean_hat/cfg.episode_length, 
                                              'episode_reward_std_bar': episode_reward_std_bar/cfg.episode_length, 'episode_reward_std_hat': episode_reward_std_hat/cfg.episode_length, 
                                              'ALIGN_epi': ALIGN_epi}, ep_index)

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    maddpg.prep_training(device=cfg.device)
    maddpg.save(run_dir / 'model.pt')
      
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    
    run(args)  
