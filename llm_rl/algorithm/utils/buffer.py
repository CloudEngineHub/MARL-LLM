import numpy as np
from torch import Tensor

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []

        self.obs_buffs = np.zeros((self.max_steps * self.num_agents, state_dim)) 
        self.ac_buffs = np.zeros((self.max_steps * self.num_agents, action_dim))
        self.rew_buffs = np.zeros((self.max_steps * self.num_agents, 1))
        self.next_obs_buffs = np.zeros((self.max_steps * self.num_agents, state_dim))
        self.done_buffs = np.zeros((self.max_steps * self.num_agents, 1))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

        self.agent_index= start_stop_index

    def __len__(self):           
        return self.filled_i
                                            
    def push(self, observations_original, actions_original, rewards_original, next_observations_original, dones_original):
        
        observations = observations_original[:, self.agent_index].T   
        actions = actions_original[:,self.agent_index].T
        rewards = rewards_original[:, self.agent_index].T                  
        next_observations = next_observations_original[:, self.agent_index].T
        dones = dones_original[:, self.agent_index].T          
   
        if self.curr_i + self.num_agents > self.max_steps * self.num_agents:   
            rollover = self.max_steps * self.num_agents - self.curr_i # num of indices to roll over

            self.obs_buffs = np.roll(self.obs_buffs, rollover, axis=0)     
            self.ac_buffs = np.roll(self.ac_buffs, rollover, axis=0)
            self.rew_buffs = np.roll(self.rew_buffs, rollover, axis=0)
            self.next_obs_buffs = np.roll(self.next_obs_buffs, rollover, axis=0)
            self.done_buffs = np.roll(self.done_buffs, rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_steps

        # Add num_agents transitions at each step
        self.obs_buffs[self.curr_i:self.curr_i + self.num_agents, :] = observations             
        self.ac_buffs[self.curr_i:self.curr_i + self.num_agents, :] = actions
        self.rew_buffs[self.curr_i:self.curr_i + self.num_agents, :] = rewards
        self.next_obs_buffs[self.curr_i:self.curr_i + self.num_agents, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + self.num_agents, :] = dones         

        self.curr_i += self.num_agents

        if self.filled_i < self.max_steps * self.num_agents:
            self.filled_i += self.num_agents         
        if self.curr_i == self.max_steps * self.num_agents: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, norm_rews=True, agent_index=0):

        obs_inds = np.zeros((N, self.obs_buffs.shape[1]))
        act_inds = np.zeros((N, self.ac_buffs.shape[1]))
        rew_inds = np.zeros((N, 1))
        next_obs_inds = np.zeros((N, self.next_obs_buffs.shape[1]))
        done_inds = np.zeros((N, 1))

        # flocking
        begin_index = np.random.randint(0, 2e5)
        inds = np.random.choice(np.arange(begin_index, 3e5 + begin_index, dtype=np.int32), size=N, replace=False)

        obs_inds = self.obs_buffs[inds, :]
        act_inds = self.ac_buffs[inds, :]
        rew_inds = self.rew_buffs[inds, :]
        next_obs_inds = self.next_obs_buffs[inds, :]
        done_inds = self.done_buffs[inds, :]
        
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        return (cast(obs_inds), cast(act_inds), cast(rew_inds), cast(next_obs_inds), cast(done_inds))

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)   
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
