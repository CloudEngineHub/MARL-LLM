import numpy as np
from torch import Tensor
import torch

class ReplayBufferAgent(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.ac_prior_buffs = []
        self.log_pi_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.total_length = self.max_steps * self.num_agents

        self.obs_buffs = np.zeros((self.total_length, state_dim)) 
        self.ac_buffs = np.zeros((self.total_length, action_dim))
        self.ac_prior_buffs = np.zeros((self.total_length, action_dim))
        self.log_pi_buffs = np.zeros((self.total_length, 1))
        self.rew_buffs = np.zeros((self.total_length, 1))
        self.next_obs_buffs = np.zeros((self.total_length, state_dim))
        self.done_buffs = np.zeros((self.total_length, 1))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

        self.agent_index= start_stop_index

    def __len__(self):           
        return self.filled_i
                                            
    def push(self, observations_orig, actions_orig, rewards_orig, next_observations_orig, dones_orig, index, actions_prior_orig=None, log_pi_orig=None):

        start = index.start
        stop = index.stop
        span = range(start, stop)
        data_length = len(span)
        
        observations = observations_orig[:, index].T   
        actions = actions_orig[:,index].T
        rewards = rewards_orig[:, index].T                  
        next_observations = next_observations_orig[:, index].T
        dones = dones_orig[:, index].T        
        if actions_prior_orig is not None:
            actions_prior = actions_prior_orig[:,index].T
        if log_pi_orig is not None:
            log_pis = log_pi_orig[:, index].T 
     
        if self.curr_i + data_length > self.total_length:   
            rollover = data_length - (self.total_length - self.curr_i) # num of indices to roll over
            self.curr_i -= rollover

        # Add num_agents transitions at each step
        self.obs_buffs[self.curr_i:self.curr_i + data_length, :] = observations             
        self.ac_buffs[self.curr_i:self.curr_i + data_length, :] = actions
        self.rew_buffs[self.curr_i:self.curr_i + data_length, :] = rewards
        self.next_obs_buffs[self.curr_i:self.curr_i + data_length, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + data_length, :] = dones     
        if actions_prior_orig is not None:
            self.ac_prior_buffs[self.curr_i:self.curr_i + data_length, :] = actions_prior
        if log_pi_orig is not None:
            self.log_pi_buffs[self.curr_i:self.curr_i + data_length, :] = log_pis    

        self.curr_i += data_length

        if self.filled_i < self.total_length:
            self.filled_i += data_length         
        if self.curr_i == self.total_length: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, is_prior=False, is_log_pi=False):

        # formation
        # begin_index_range = 6e5 # 4.5e5
        begin_index_range = 7e4
        begin_index = np.random.randint(0, begin_index_range)
        inds = np.random.choice(np.arange(begin_index, self.total_length - begin_index_range + begin_index, dtype=np.int32), size=N, replace=False)

        obs_inds = self.obs_buffs[inds, :]
        act_inds = self.ac_buffs[inds, :]
        rew_inds = self.rew_buffs[inds, :]
        next_obs_inds = self.next_obs_buffs[inds, :]
        done_inds = self.done_buffs[inds, :]
        if is_prior:
            act_prior_inds = self.ac_prior_buffs[inds, :]
        if is_log_pi:
            log_pis_inds = self.log_pi_buffs[inds, :]
        
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        return (cast(obs_inds), cast(act_inds), cast(rew_inds), cast(next_obs_inds), cast(done_inds), cast(act_prior_inds) if is_prior else None, cast(log_pis_inds) if is_log_pi else None)

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)   
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
