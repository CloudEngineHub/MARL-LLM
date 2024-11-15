from numpy.lib.function_base import insert
import torch
import numpy as np
from itertools import chain
import copy

from torch._C import dtype
from .util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

class ReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_dim, cent_obs_dim, act_dim):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.cent_obs_dim = cent_obs_dim
        self.act_dim = act_dim
        self.data_buffer_length = int(args.data_buffer_length)
        self.episode_length = args.episode_length
        self.batch_size = args.batch_size
        self.sample_index_start = args.sample_index_start
        self.hidden_dim = args.hidden_dim
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.advantage_method = args.advantage_method
        self._use_valuenorm = args.use_valuenorm

        self.cent_obs_buffer = np.zeros(((self.episode_length + 1), self.num_agents, self.cent_obs_dim), dtype=np.float32)
        self.obs_buffer = np.zeros(((self.episode_length + 1), self.num_agents, self.obs_dim), dtype=np.float32)
        self.value_preds_buffer = np.zeros(((self.episode_length + 1), self.num_agents, 1), dtype=np.float32)
        self.returns_buffer = np.zeros_like(self.value_preds_buffer)
        self.masks_buffer = np.zeros(((self.episode_length + 1), self.num_agents, 1), dtype=np.float32)
        self.actions_buffer = np.zeros((self.episode_length, self.num_agents, self.act_dim), dtype=np.float32)
        self.action_log_probs_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)
        self.rewards_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)
        self.advantages_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)

        self.next_obs_buffer = np.zeros((self.episode_length, self.num_agents, self.obs_dim), dtype=np.float32)

        self.stop_index = np.zeros(self.num_agents, dtype=np.int32)
        self.step = 0

        self.data_cent_obs_buffer = np.zeros((self.data_buffer_length, self.cent_obs_dim), dtype=np.float32)
        self.data_obs_buffer = np.zeros((self.data_buffer_length, self.obs_dim), dtype=np.float32)
        self.data_value_preds_buffer = np.zeros((self.data_buffer_length, 1), dtype=np.float32)
        self.data_returns_buffer = np.zeros_like(self.data_value_preds_buffer)
        self.data_masks_buffer = np.zeros((self.data_buffer_length, 1), dtype=np.float32)
        self.data_actions_buffer = np.zeros((self.data_buffer_length, self.act_dim), dtype=np.float32)
        self.data_action_log_probs_buffer = np.zeros((self.data_buffer_length, 1), dtype=np.float32)
        self.data_rewards_buffer = np.zeros((self.data_buffer_length, 1), dtype=np.float32)
        self.data_advantages_buffer = np.zeros((self.data_buffer_length, 1), dtype=np.float32)

        self.data_next_obs_buffer = np.zeros((self.data_buffer_length, self.obs_dim), dtype=np.float32)

        self.curr_i = 0
        self.filled_i = 0

        self.data_cent_obs_buffer_list = []
        self.data_obs_buffer_list = []
        self.data_actions_buffer_list = []
        self.data_action_log_probs_buffer_list = []
        self.data_value_preds_buffer_list = []
        self.data_returns_buffer_list = []
        self.data_masks_buffer_list = []
        self.data_advantages_buffer_list = []

        self.data_chunks_length = []

    def push(self, store_info, index, index_origin, agent_i, n_lr_init):
        obs_input, cent_obs_input, act_input, act_log_probs_input, rew_input, dones_input, values_input = store_info

        obs = obs_input[:,index].T   
        cent_obs = cent_obs_input[:,index].T     
        act = act_input[:,index].T  
        act_log_probs = act_log_probs_input[index] 
        rew = rew_input[:,index].T   
        dones = dones_input[:,index].T     
        values = values_input[index]      

        # Add num_agents transitions at each step
        if agent_i == 0:
            agent_index = index_origin
        else:
            agent_index = index_origin - n_lr_init
        self.obs_buffer[self.step + 1][agent_index] = obs 
        self.cent_obs_buffer[self.step + 1][agent_index] = cent_obs
        self.masks_buffer[self.step + 1][agent_index] = 1.0 - dones 

        self.next_obs_buffer[self.step][agent_index] = obs

        self.actions_buffer[self.step][agent_index] = act 
        self.action_log_probs_buffer[self.step][agent_index] = act_log_probs
        self.rewards_buffer[self.step][agent_index] = rew 
        self.value_preds_buffer[self.step][agent_index] = values    

        # for i in range(self.num_agents):
        #     # if i not in agent_index and self.stop_index[i] == 0:
        #     #     self.stop_index[i] = self.step
        #     if i in agent_index:
        #         self.stop_index[i] = self.step

        self.stop_index = np.ones_like(self.stop_index) * self.step

        self.step += 1

    def push_obs(self, index, obs, cent_obs, rnn_state=None, rnn_state_critic=None):
        observations = obs[:, index].T   
        cent_observations = cent_obs[:,index].T        

        # Add num_agents transitions at each step
        self.obs_buffer[0] = observations 
        self.cent_obs_buffer[0] = cent_observations   
        self.masks_buffer[0] = 1        

    def reset(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.cent_obs_buffer = np.zeros(((self.episode_length + 1), self.num_agents, self.cent_obs_dim), dtype=np.float32)
        self.obs_buffer = np.zeros(((self.episode_length + 1), self.num_agents, self.obs_dim), dtype=np.float32)
        self.value_preds_buffer = np.zeros(((self.episode_length + 1), self.num_agents, 1), dtype=np.float32)
        self.returns_buffer = np.zeros_like(self.value_preds_buffer)
        self.masks_buffer = np.zeros(((self.episode_length + 1), self.num_agents, 1), dtype=np.float32)
        self.actions_buffer = np.zeros((self.episode_length, self.num_agents, self.act_dim), dtype=np.float32)
        self.action_log_probs_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)
        self.rewards_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)
        self.advantages_buffer = np.zeros((self.episode_length, self.num_agents, 1), dtype=np.float32)

        self.next_obs_buffer = np.zeros((self.episode_length, self.num_agents, self.obs_dim), dtype=np.float32)

        self.stop_index = np.zeros(self.num_agents, dtype=np.int32)
        self.step = 0

    @torch.no_grad()
    def compute_returns(self, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        # self.value_preds_buffer[self.step] = next_value
        # min_val = self.rewards_buffer.min()
        # max_val = self.rewards_buffer.max()
        # self.rewards_buffer = (self.rewards_buffer - min_val) / (max_val - min_val + 1e-6)
        # rewards_copy = copy.deepcopy(self.rewards_buffer[:self.step])
        # rewards_copy[self.masks_buffer[1:(self.step + 1)] == 0.0] = np.nan
        # mean_rewards = np.nanmean(rewards_copy)
        # std_rewards = np.nanstd(rewards_copy)
        # self.rewards_buffer = (self.rewards_buffer - mean_rewards) / (std_rewards + 1e-6)

        # for i in range(self.num_agents):
        #     reward_i_copy = copy.deepcopy(self.rewards_buffer[:(self.stop_index[i] + 1),i,:])
        #     mean_reward_i = np.mean(reward_i_copy)
        #     std_reward_i = np.std(reward_i_copy)
        #     reward_i_copy = (reward_i_copy - mean_reward_i) / (std_reward_i + 1e-6)
        #     # min_val = reward_i_copy.min()
        #     # max_val = reward_i_copy.max()
        #     # reward_i_copy = (reward_i_copy - min_val) / (max_val - min_val + 1e-6)
        #     self.rewards_buffer[:(self.stop_index[i] + 1),i,:] = copy.deepcopy(reward_i_copy)
        #     # print(mean_reward_i, std_reward_i)

        # min_val = self.value_preds_buffer.min()
        # max_val = self.value_preds_buffer.max()
        # self.value_preds_buffer = (self.value_preds_buffer - min_val) / (max_val - min_val + 1e-6)

        if self.advantage_method == 'GAE':
            gae = 0
            for step in reversed(range(self.step)):
                if self._use_valuenorm:
                    delta = self.rewards_buffer[step] + self.gamma * value_normalizer.denormalize(self.value_preds_buffer[step + 1]) * self.masks_buffer[step + 1] \
                            - value_normalizer.denormalize(self.value_preds_buffer[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks_buffer[step + 1] * gae
                    self.advantages_buffer[step] = gae
                    self.returns_buffer[step] = gae + value_normalizer.denormalize(self.value_preds_buffer[step])
                else:
                    delta = self.rewards_buffer[step] + self.gamma * self.value_preds_buffer[step + 1] * self.masks_buffer[step + 1] - self.value_preds_buffer[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks_buffer[step + 1] * gae
                    self.advantages_buffer[step] = gae
                    self.returns_buffer[step] = gae + self.value_preds_buffer[step]
        elif self.advantage_method == 'TD':
            for step in reversed(range(self.step)):
                self.returns_buffer[step] = self.rewards_buffer[step] + self.gamma * value_normalizer.denormalize(self.value_preds_buffer[step + 1]) * self.masks_buffer[step + 1]
                self.advantages_buffer[step] = self.returns_buffer[step] - value_normalizer.denormalize(self.value_preds_buffer[step])
        elif self.advantage_method == 'n_step_TD':
            n_step = 20
            if self._use_valuenorm:
                for step in range(self.step - n_step):
                    cumulative_reward = 0
                    for i in range(n_step):
                        cumulative_reward += self.gamma ** i * self.rewards_buffer[step + i]
                    self.returns_buffer[step] = cumulative_reward + self.gamma ** n_step * value_normalizer.denormalize(self.value_preds_buffer[step + n_step])
                    self.advantages_buffer[step] = self.returns_buffer[step] - value_normalizer.denormalize(self.value_preds_buffer[step])

                for step in range(self.step - n_step, self.step):
                    cumulative_reward = 0
                    n_step_2 = self.step - step
                    for i in range(n_step_2):
                        cumulative_reward += self.gamma ** i * self.rewards_buffer[step + i]
                    self.returns_buffer[step] = cumulative_reward + self.gamma ** n_step_2 * value_normalizer.denormalize(self.value_preds_buffer[step + n_step_2])
                    self.advantages_buffer[step] = self.returns_buffer[step] - value_normalizer.denormalize(self.value_preds_buffer[step])
            else:
                for step in range(self.step - n_step):
                    cumulative_reward = 0
                    for i in range(n_step):
                        cumulative_reward += self.gamma ** i * self.rewards_buffer[step + i]
                    self.returns_buffer[step] = cumulative_reward + self.gamma ** n_step * self.value_preds_buffer[step + n_step]
                    self.advantages_buffer[step] = self.returns_buffer[step] - self.value_preds_buffer[step]

                for step in range(self.step - n_step, self.step):
                    cumulative_reward = 0
                    n_step_2 = self.step - step
                    for i in range(n_step_2):
                        cumulative_reward += self.gamma ** i * self.rewards_buffer[step + i]
                    self.returns_buffer[step] = cumulative_reward + self.gamma ** n_step_2 * self.value_preds_buffer[step + n_step_2]
                    self.advantages_buffer[step] = self.returns_buffer[step] - self.value_preds_buffer[step]

    def feed_forward_generator(self, advantages_input, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        # episode_length, num_agents = self.rewards_buffer.shape[0:2]
        # batch_size = episode_length * num_agents
        # batch_size = self.step * num_agents
        total_step_num = np.sum(self.stop_index) + self.num_agents
        batch_size = total_step_num
        # batch_size = 2000
        mini_batch_size = batch_size // num_mini_batch

        cent_obs = np.zeros((total_step_num, self.cent_obs_dim), dtype=np.float32)
        obs = np.zeros((total_step_num, self.obs_dim), dtype=np.float32)
        actions = np.zeros((total_step_num, self.act_dim), dtype=np.float32)
        action_log_probs = np.zeros((total_step_num, 1), dtype=np.float32)
        value_preds = np.zeros((total_step_num, 1), dtype=np.float32)
        returns = np.zeros((total_step_num, 1), dtype=np.float32)
        masks = np.zeros((total_step_num, 1), dtype=np.float32)
        advantages = np.zeros((total_step_num, 1), dtype=np.float32)

        start_insert_index = 0
        for i in range(self.num_agents):
            step_num = self.stop_index[i] + 1

            cent_obs[start_insert_index:start_insert_index + step_num] = self.cent_obs_buffer[:step_num,i,:]
            obs[start_insert_index:start_insert_index + step_num] = self.obs_buffer[:step_num,i,:]
            actions[start_insert_index:start_insert_index + step_num] = self.actions_buffer[:step_num,i,:]
            action_log_probs[start_insert_index:start_insert_index + step_num] = self.action_log_probs_buffer[:step_num,i,:]
            value_preds[start_insert_index:start_insert_index + step_num] = self.value_preds_buffer[:step_num,i,:]
            returns[start_insert_index:start_insert_index + step_num] = self.returns_buffer[:step_num,i,:]
            masks[start_insert_index:start_insert_index + step_num] = self.masks_buffer[:step_num,i,:]
            advantages[start_insert_index:start_insert_index + step_num] = advantages_input[:step_num,i,:]

            start_insert_index += step_num

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            cent_obs_batch = cent_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            yield cent_obs_batch, obs_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

        # inds = np.random.choice(np.arange(0, total_step_num, dtype=np.int32), size=batch_size, replace=False)
        # # print(inds)
        # cent_obs_batch = cent_obs[inds]
        # obs_batch = obs[inds]
        # actions_batch = actions[inds]
        # value_preds_batch = value_preds[inds]
        # return_batch = returns[inds]
        # masks_batch = masks[inds]
        # old_action_log_probs_batch = action_log_probs[inds]
        # adv_targ = advantages[inds]

        # yield cent_obs_batch, obs_batch, actions_batch,\
        #         value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages_input, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        # episode_length, num_agents = self.rewards_buffer.shape[0:2]
        # batch_size = episode_length * num_agents
        total_step_num = np.sum(self.stop_index) + self.num_agents
        batch_size = total_step_num
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        cent_obs = np.zeros((total_step_num, self.cent_obs_dim), dtype=np.float32)
        obs = np.zeros((total_step_num, self.obs_dim), dtype=np.float32)
        actions = np.zeros((total_step_num, self.act_dim), dtype=np.float32)
        action_log_probs = np.zeros((total_step_num, 1), dtype=np.float32)
        value_preds = np.zeros((total_step_num, 1), dtype=np.float32)
        returns = np.zeros((total_step_num, 1), dtype=np.float32)
        masks = np.zeros((total_step_num, 1), dtype=np.float32)
        advantages = np.zeros((total_step_num, 1), dtype=np.float32)

        start_insert_index = 0
        for i in range(self.num_agents):
            step_num = self.stop_index[i] + 1

            cent_obs[start_insert_index:start_insert_index + step_num] = self.cent_obs_buffer[:step_num,i,:]
            obs[start_insert_index:start_insert_index + step_num] = self.obs_buffer[:step_num,i,:]
            actions[start_insert_index:start_insert_index + step_num] = self.actions_buffer[:step_num,i,:]
            action_log_probs[start_insert_index:start_insert_index + step_num] = self.action_log_probs_buffer[:step_num,i,:]
            value_preds[start_insert_index:start_insert_index + step_num] = self.value_preds_buffer[:step_num,i,:]
            returns[start_insert_index:start_insert_index + step_num] = self.returns_buffer[:step_num,i,:]
            masks[start_insert_index:start_insert_index + step_num] = self.masks_buffer[:step_num,i,:]
            advantages[start_insert_index:start_insert_index + step_num] = advantages_input[:step_num,i,:]

            start_insert_index += step_num

        for indices in sampler:
            cent_obs_batch = []
            obs_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                cent_obs_batch.append(cent_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])

            # These are all from_numpys of size (L, N, Dim)           
            cent_obs_batch = np.concatenate(np.stack(cent_obs_batch))
            obs_batch = np.concatenate(np.stack(obs_batch))
            actions_batch = np.concatenate(np.stack(actions_batch))
            value_preds_batch = np.concatenate(np.stack(value_preds_batch))
            return_batch = np.concatenate(np.stack(return_batch))
            masks_batch = np.concatenate(np.stack(masks_batch))
            old_action_log_probs_batch = np.concatenate(np.stack(old_action_log_probs_batch))
            adv_targ = np.concatenate(np.stack(adv_targ))

            yield cent_obs_batch, obs_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def insert_data_buffer(self, advantages_input, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """

        total_step_num = np.sum(self.stop_index) + self.num_agents

        cent_obs = np.zeros((total_step_num, self.cent_obs_dim), dtype=np.float32)
        obs = np.zeros((total_step_num, self.obs_dim), dtype=np.float32)
        actions = np.zeros((total_step_num, self.act_dim), dtype=np.float32)
        action_log_probs = np.zeros((total_step_num, 1), dtype=np.float32)
        value_preds = np.zeros((total_step_num, 1), dtype=np.float32)
        returns = np.zeros((total_step_num, 1), dtype=np.float32)
        masks = np.zeros((total_step_num, 1), dtype=np.float32)
        advantages = np.zeros((total_step_num, 1), dtype=np.float32)

        next_obs = np.zeros((total_step_num, self.obs_dim), dtype=np.float32)

        start_insert_index = 0
        for i in range(self.num_agents):
            step_num = self.stop_index[i] + 1

            cent_obs[start_insert_index:start_insert_index + step_num] = self.cent_obs_buffer[:step_num,i,:]
            obs[start_insert_index:start_insert_index + step_num] = self.obs_buffer[:step_num,i,:]
            actions[start_insert_index:start_insert_index + step_num] = self.actions_buffer[:step_num,i,:]
            action_log_probs[start_insert_index:start_insert_index + step_num] = self.action_log_probs_buffer[:step_num,i,:]
            value_preds[start_insert_index:start_insert_index + step_num] = self.value_preds_buffer[:step_num,i,:]
            returns[start_insert_index:start_insert_index + step_num] = self.returns_buffer[:step_num,i,:]
            masks[start_insert_index:start_insert_index + step_num] = self.masks_buffer[:step_num,i,:]
            advantages[start_insert_index:start_insert_index + step_num] = advantages_input[:step_num,i,:]

            next_obs[start_insert_index:start_insert_index + step_num] = self.next_obs_buffer[:step_num,i,:]

            start_insert_index += step_num
        
        if self.curr_i + total_step_num > self.data_buffer_length:
            rollover = total_step_num - (self.data_buffer_length - self.curr_i) # num of indices to roll over
            self.curr_i -= rollover

        self.data_cent_obs_buffer[self.curr_i:self.curr_i + total_step_num] = cent_obs
        self.data_obs_buffer[self.curr_i:self.curr_i + total_step_num] = obs
        self.data_actions_buffer[self.curr_i:self.curr_i + total_step_num] = actions
        self.data_action_log_probs_buffer[self.curr_i:self.curr_i + total_step_num] = action_log_probs
        self.data_value_preds_buffer[self.curr_i:self.curr_i + total_step_num] = value_preds
        self.data_returns_buffer[self.curr_i:self.curr_i + total_step_num] = returns
        self.data_masks_buffer[self.curr_i:self.curr_i + total_step_num] = masks
        self.data_advantages_buffer[self.curr_i:self.curr_i + total_step_num] = advantages

        self.data_next_obs_buffer[self.curr_i:self.curr_i + total_step_num] = next_obs

        self.curr_i += total_step_num
        self.filled_i += total_step_num

        if self.curr_i >= self.data_buffer_length: 
            self.curr_i = 0 
        if self.filled_i >= self.data_buffer_length:
            self.filled_i = self.data_buffer_length
    
    def insert_data_buffer_recurrent(self, advantages_input, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """

        # total_step_num = np.sum(self.stop_index) + self.num_agents

        for i in range(self.num_agents):
            step_num = self.stop_index[i] + 1

            while (self.curr_i + step_num) > self.data_buffer_length:
                del self.data_cent_obs_buffer_list[0]
                del self.data_obs_buffer_list[0]
                del self.data_actions_buffer_list[0]
                del self.data_action_log_probs_buffer_list[0]
                del self.data_value_preds_buffer_list[0]
                del self.data_returns_buffer_list[0]
                del self.data_masks_buffer_list[0]
                del self.data_advantages_buffer_list[0]

                self.curr_i -= self.data_chunks_length[0]
                del self.data_chunks_length[0]

            self.data_cent_obs_buffer_list.append(self.cent_obs_buffer[:step_num,i,:])
            self.data_obs_buffer_list.append(self.obs_buffer[:step_num,i,:])
            self.data_actions_buffer_list.append(self.actions_buffer[:step_num,i,:])
            self.data_action_log_probs_buffer_list.append(self.action_log_probs_buffer[:step_num,i,:])
            self.data_value_preds_buffer_list.append(self.value_preds_buffer[:step_num,i,:])
            self.data_returns_buffer_list.append(self.returns_buffer[:step_num,i,:])
            self.data_masks_buffer_list.append(self.masks_buffer[:step_num,i,:])
            self.data_advantages_buffer_list.append(advantages_input[:step_num,i,:])

            self.data_chunks_length.append(step_num)
            self.curr_i += step_num
            self.filled_i += step_num

        if self.filled_i >= self.data_buffer_length:
            self.filled_i = self.data_buffer_length
        
    def sample(self):
        begin_index = np.random.randint(0, self.sample_index_start)
        inds = np.random.choice(np.arange(begin_index, self.data_buffer_length - self.sample_index_start + begin_index, dtype=np.int32), size=self.batch_size, replace=False)
        # print(inds)
        cent_obs_batch = self.data_cent_obs_buffer[inds]
        obs_batch = self.data_obs_buffer[inds]
        actions_batch = self.data_actions_buffer[inds]
        value_preds_batch = self.data_value_preds_buffer[inds]
        return_batch = self.data_returns_buffer[inds]
        masks_batch = self.data_masks_buffer[inds]
        old_action_log_probs_batch = self.data_action_log_probs_buffer[inds]
        adv_targ = self.data_advantages_buffer[inds]

        next_obs_batch = self.data_next_obs_buffer[inds]

        return cent_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, next_obs_batch

    def sample_recurrent(self):

        data_chunks_num = len(self.data_chunks_length)
        # inds = np.random.choice(np.arange(0, data_chunks_num, dtype=np.int32), size=6, replace=False)

        # cent_obs_batch = np.vstack([self.data_cent_obs_buffer_list[i] for i in inds])
        # obs_batch = np.vstack([self.data_obs_buffer_list[i] for i in inds])
        # actions_batch = np.vstack([self.data_actions_buffer_list[i] for i in inds])
        # value_preds_batch = np.vstack([self.data_value_preds_buffer_list[i] for i in inds])
        # return_batch = np.vstack([self.data_returns_buffer_list[i] for i in inds])
        # masks_batch = np.vstack([self.data_masks_buffer_list[i] for i in inds])
        # old_action_log_probs_batch = np.vstack([self.data_action_log_probs_buffer_list[i] for i in inds])
        # adv_targ = np.vstack([self.data_advantages_buffer_list[i] for i in inds])

        inds = np.random.choice(np.arange(0, self.curr_i - self.batch_size, dtype=np.int32), size=1, replace=False)

        cent_obs_batch_all = np.vstack([self.data_cent_obs_buffer_list[i] for i in range(data_chunks_num)])
        obs_batch_all = np.vstack([self.data_obs_buffer_list[i] for i in range(data_chunks_num)])
        actions_batch_all = np.vstack([self.data_actions_buffer_list[i] for i in range(data_chunks_num)])
        value_preds_batch_all = np.vstack([self.data_value_preds_buffer_list[i] for i in range(data_chunks_num)])
        return_batch_all = np.vstack([self.data_returns_buffer_list[i] for i in range(data_chunks_num)])
        masks_batch_all = np.vstack([self.data_masks_buffer_list[i] for i in range(data_chunks_num)])
        old_action_log_probs_batch_all = np.vstack([self.data_action_log_probs_buffer_list[i] for i in range(data_chunks_num)])
        adv_targ_all = np.vstack([self.data_advantages_buffer_list[i] for i in range(data_chunks_num)])

        cent_obs_batch = cent_obs_batch_all[inds[0]:inds[0] + self.batch_size]
        obs_batch = obs_batch_all[inds[0]:inds[0] + self.batch_size]
        actions_batch = actions_batch_all[inds[0]:inds[0] + self.batch_size]
        value_preds_batch = value_preds_batch_all[inds[0]:inds[0] + self.batch_size]
        return_batch = return_batch_all[inds[0]:inds[0] + self.batch_size]
        masks_batch = masks_batch_all[inds[0]:inds[0] + self.batch_size]
        old_action_log_probs_batch = old_action_log_probs_batch_all[inds[0]:inds[0] + self.batch_size]
        adv_targ = adv_targ_all[inds[0]:inds[0] + self.batch_size]

        return cent_obs_batch, obs_batch, actions_batch,\
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
