import numpy as np
from torch import Tensor
import torch

class ReplayBufferEpisode(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_size):
         self.max_size = max_size
         self.data = []
         self.ptr = 0

    def append(self, state, act, rew, state_act, n_a, epi_len):
        for agent_i in range(n_a):
            state_list = []
            act_list = []
            rew_list = []
            state_act_list = []
            for time_i in range(epi_len):
                state_list.append(state[time_i][:, agent_i])
                act_list.append(act[time_i][:, agent_i])
                rew_list.append(rew[time_i][:, agent_i])
                state_act_list.append(state_act[time_i][:, agent_i])

            if self.full():
                self.data[self.ptr] = [state_list, act_list, rew_list, state_act_list]
            else:
                self.data.append([state_list, act_list, rew_list, state_act_list])

            self.ptr = (self.ptr + 1) % self.max_size
                           
    def sample(self, sample_size, to_gpu):
        idxes = np.random.choice(len(self.data), sample_size)
        mini_batch = []

        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        for idx in idxes:
            traj_data = [[cast(arr) for arr in row] for row in self.data[idx]]
            mini_batch.append(traj_data)

        return mini_batch

    def __len__(self): 
        return len(self.data)

    def full(self): 
        return len(self.data) == self.max_size  

