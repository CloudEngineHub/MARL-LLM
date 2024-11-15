import torch
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_level_dir = os.path.join(current_dir, '..')
sys.path.append(upper_level_dir)
import numpy as np
import gym
from gym.wrappers import FlockingSwarmWrapper
from cfg.flocking_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from datetime import datetime
from algorithm.algorithms.maddpg import MADDPG


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
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 

    # environment
    scenario_name = 'FlockingSwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = FlockingSwarmWrapper(base_env, args)

    start_stop_num = [slice(0,env.num_agents)]  

    # algorithm
    if scenario_name == 'PredatorPreySwarm-v0':
        adversary_alg = cfg.adversary_alg
    else:
        adversary_alg = None

    maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg, tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
                                  lambda_s=cfg.lambda_s, epsilon_p=cfg.epsilon_p, hidden_dim=cfg.hidden_dim, device=cfg.device, epsilon=cfg.epsilon, noise=cfg.noise_scale)
    # last_run = '2024-07-27-16-33-51'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save(last_run_dir)

    # buffer
    agent_buffer = [ReplayBuffer(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                start_stop_index=start_stop_num[0])]    

    p_store = []
    dp_store = []
    torch_agent_actions=[]

    ## ======================================= Training =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward_mean = 0
        episode_reward_std = 0
        obs = env.reset()     
        maddpg.prep_rollouts(device='cpu') 
      
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        M_p, N_p = np.shape(env.p)     
        M_v, N_v =np.shape(env.dp)
        p_store = np.zeros((M_p, N_p, cfg.episode_length))       
        dp_store = np.zeros((M_v, N_v, cfg.episode_length))
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()
            
            p_store[:, :, et_index] = env.p             
            dp_store[:, :, et_index] = env.dp

            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions = maddpg.step(torch_obs, start_stop_num, explore=True) 
            # convert actions to numpy.arrays
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            next_obs, rewards, dones, infos = env.step(agent_actions)    
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs  

            episode_reward_mean += np.mean(rewards)
            episode_reward_std += np.std(rewards) 

        end_time_1 = time.time()
        ########################### train ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)
        for _ in range(20):      
            for a_i in range(maddpg.nagents):
                # if len(agent_buffer[a_i]) >= 900 * cfg.batch_size:
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                # if len(agent_buffer[a_i]) >= 10:
                    sample = agent_buffer[a_i].sample(cfg.batch_size, to_gpu=True if cfg.device == 'gpu' else False)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample = sample  
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, a_i, logger=logger)     # parameter update 
            maddpg.update_all_targets()
            
        maddpg.prep_rollouts(device='cpu')    
        # maddpg.noise = max(0.1, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)
        end_time_2 = time.time()

        ########################### process data ###########################
        if ep_index % 10 == 0:
            print("Episodes %i of %i, episode reward: %f, step time: %f, training time: %f" % (ep_index, cfg.n_episodes, episode_reward_mean/cfg.episode_length, 
                  end_time_1 - start_time_1, end_time_2 - start_time_2))
        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0
            logger.add_scalars('agent/data', {'episode_reward_mean': episode_reward_mean/cfg.episode_length, 'episode_reward_std': episode_reward_std/cfg.episode_length, 'ALIGN_epi': ALIGN_epi}, ep_index)

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    maddpg.save(run_dir / 'model.pt')
      
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    
    run(args)  
