import argparse
import torch
import time
import os
import sys
import random
from torch._C import device
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_level_dir = os.path.join(current_dir, '..')
sys.path.append(upper_level_dir)
import numpy as np
import gym
from gym.wrappers import FormationSwarmWrapper
from cfg.formation_mappo_cfg import gpsargs as args
from algorithm.algorithms.mappo import MAPPO
from pathlib import Path
from datetime import datetime
from algorithm.mappo_utils.buffer import ReplayBuffer as ReplayBufferAgent
from algorithm.utils.buffer_expert import ReplayBuffer as ReplayBufferExpert
from algorithm.algorithms.airl import AIRL
from tensorboardX import SummaryWriter
from algorithm.mappo_utils.util import check

USE_CUDA = True 

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
    if USE_CUDA:
        device = torch.device("cuda:0")
        torch.set_num_threads(cfg.n_training_threads) 
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(cfg.n_training_threads) 

    # buffer
    # load expert demonstration
    model_dir = './' / Path('./eval') / 'expert_data' 
    curr_run = '2024-11-02-10-42-58'
    expert_dir = os.path.join(model_dir, curr_run, 'expert_data.npz') 
    expert_buffer = ReplayBufferExpert(cfg.buffer_length, env.num_agents, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], 
                                       start_stop_index=start_stop_num[0])
    expert_buffer.load(expert_dir)
    # buffer
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cent_obs_dim = obs_dim + 4 * cfg.n_a if cfg.use_centralized_V else obs_dim
    agents_buffer = ReplayBufferAgent(cfg, env.n_a, obs_dim, cent_obs_dim, act_dim)      
    buffer_total=[agents_buffer]  

    # algorithm
    mappo = MAPPO.init_from_env(env, cfg, device)
    airl = AIRL(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=cfg.hidden_dim, hidden_num=3, lr_discriminator=cfg.lr_discriminator, expert_buffer=expert_buffer, device=cfg.device)


    # training index
    training_index = range(1)
    index_a = np.array([i for i in range(env.n_a)])

    p_store = []
    h_store = []

    ## ======================================= Training =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):

        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0

        obs = env.reset() 
        cent_obs = obs if cfg.use_centralized_V else obs
        masks = np.ones((env.n_a, 1), dtype=np.float32)

        # push the initial state
        start_stop_num = [slice(0, env.n_a)]  
        agents_buffer.push_obs(start_stop_num[0], obs, cent_obs)

        # M_l, N_l = np.shape(env.p)     
        # M_h, N_h = np.shape(env.heading)
        # p_store = np.zeros((M_l, N_l, cfg.episode_length))       
        # h_store = np.zeros((M_h, N_h, cfg.episode_length))

        if cfg.use_linear_lr_decay:
            for a in mappo.agents:
                a.lr_decay(ep_index, cfg.n_episodes)
        
        ########################### step one episode ###########################
        start_time_1 = time.time()
        et_i = 0
        mappo.prep_rollout()
        while et_i in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()
            
            # 获取 observation for per agent and convert to torch variable
            # p_store[:, :, et_i] = env.p             
            # h_store[:, :, et_i] = env.heading
 
            state_info = obs, cent_obs, masks
            values, actions, action_log_probs = mappo.step(state_info, start_stop_num, env.n_a, training_index, is_deterministic=False) 
            agent_actions = actions.T
  
            # obtain reward and next state
            next_obs, rewards, dones, infos = env.step(agent_actions)  
            next_cent_obs = next_obs if cfg.use_centralized_V else next_obs

            obs_t = check(obs.T).to(airl.device, dtype=torch.float32)
            actions_t = check(actions).to(airl.device, dtype=torch.float32)
            action_log_probs_t = check(action_log_probs).to(airl.device, dtype=torch.float32)
            next_obs_t = check(next_obs.T).to(airl.device, dtype=torch.float32)
            dones_t = check(dones.T).to(airl.device, dtype=torch.float32)
            rew = airl.discriminator.calculate_reward(obs_t, actions_t, action_log_probs_t, next_obs_t, dones_t)
            store_info = next_obs, next_cent_obs, agent_actions, action_log_probs, rew.cpu().detach().numpy().T, dones, values

            # store_info = next_obs, next_cent_obs, agent_actions, action_log_probs, rewards, dones, values
            agents_buffer.push(store_info, start_stop_num[0], index_a, 0, 0)  

            obs = next_obs 
            cent_obs = next_cent_obs 
            masks = np.ones((env.n_a, 1), dtype=np.float32)

            episode_reward_mean_bar += np.mean(rewards)
            episode_reward_std_bar += np.std(rewards) 

            et_i += 1

        end_time_1 = time.time()
        ########################### train ###########################
        start_time_2 = time.time()  
        if ep_index % 3 == 0:
            for _ in range(20):
                for a_i in range(mappo.num_agents):
                    sample = buffer_total[a_i].sample()
                    _, obs_sample, acs_sample, _, _, _, log_pis_sample, _, next_obs_sample = sample
                    airl.update(obs_sample, acs_sample, log_pis_sample, next_obs_sample, logger=logger)

        for a_i in training_index:
            mappo.compute(buffer_total[a_i], a_i)
            train_info = mappo.train(buffer_total[a_i], a_i, True)
        
        buffer_total[0].reset()
        mappo.prep_rollout()
        end_time_2 = time.time()  

        ########################### process data ###########################
        if ep_index % 10 == 0:
            print("Episodes %i of %i, agent num: %i, episode reward (sparse): %f, episode reward (total): %f, step time: %f, training time: %f" % 
                  (ep_index, cfg.n_episodes, env.n_a, episode_reward_mean_bar/cfg.episode_length, episode_reward_mean_bar/cfg.episode_length, 
                  end_time_1 - start_time_1, end_time_2 - start_time_2))

        if ep_index % cfg.save_interval == 0:
            logger.add_scalars('data', {'episode_reward_mean_bar': episode_reward_mean_bar/cfg.episode_length, 'episode_reward_std_bar': episode_reward_std_bar/cfg.episode_length}, ep_index)
            logger.add_scalars('train', {'value_loss': train_info['value_loss'], 'policy_loss': train_info['policy_loss'], 'ratio': train_info['ratio']}, ep_index)

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            mappo.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    mappo.save(run_dir / 'model.pt')
    print(mappo.agents[0].actor.base.mlp.fc1[0].weight)

    # env.close()       
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    # for i in np.arange(1, 3, 0.3):
    #     args.k_1 = i
        run(args)  