import numpy as np
import torch
import torch.nn as nn
from algorithm.mappo_utils.util import get_gard_norm, huber_loss, mse_loss, check
from algorithm.mappo_utils.value_norm import ValueNorm
from algorithm.mappo_utils.agents import PPOAgent
import copy

def _t2n(x):
    return x.detach().cpu().numpy()

class MAPPO():
    """
    Trainer class for MAPPO to update policies.
    """

    def __init__(self, args, agent_init_params, device=torch.device("cpu")):
        self.num_agents = len(agent_init_params)
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.agents = [PPOAgent(args, device=self.device, **params) for params in agent_init_params]

        self.action_space_class = args.action_space_class
        self.hidden_dim = args.hidden_dim
        self._recurrent_N = args.recurrent_N

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
            # error_clipped = self.value_normalizer.normalize(return_batch) - self.value_normalizer.normalize(value_pred_clipped)
            # error_original = self.value_normalizer.normalize(return_batch) - self.value_normalizer.normalize(values)
            # dd = error_clipped.mean()
            # ff = error_original.mean()
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        dd = value_loss_original.mean()
        ff = value_loss_clipped.mean()
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss, dd, ff

    def update(self, sample, agent_i, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.
        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        current_agent = self.agents[agent_i]
        values, action_log_probs, dist_entropy = current_agent.evaluate_action_value(share_obs_batch,
                                                                                    obs_batch,
                                                                                    actions_batch,
                                                                                    masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_total_loss = policy_loss - dist_entropy * self.entropy_coef

        current_agent.actor_optimizer.zero_grad()
        policy_total_loss.backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(current_agent.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(current_agent.actor.parameters())

        current_agent.actor_optimizer.step()

        # critic update
        value_loss, dd, ff = self.cal_value_loss(values, value_preds_batch, return_batch)
        value_loss = value_loss * self.value_loss_coef

        current_agent.critic_optimizer.zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(current_agent.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(current_agent.critic.parameters())

        current_agent.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, dd, ff

    def train(self, buffer, agent_index, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.
        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        self.prep_training()
        # index = -(buffer.episode_length + 1 - buffer.step)
        # if self._use_valuenorm:
        #     advantages = buffer.returns_buffer[:index] - self.value_normalizer.denormalize(buffer.value_preds_buffer[:index])
        # else:
        #     advantages = buffer.returns_buffer[:index] - buffer.value_preds_buffer[:index]
        advantages = buffer.advantages_buffer[:buffer.step]
        # min_val = advantages.min()
        # max_val = advantages.max()
        # advantages = (advantages - min_val) / (max_val - min_val + 1e-6)

        advantages_copy = copy.deepcopy(advantages)
        advantages_copy[buffer.masks_buffer[1:(buffer.step + 1)] == 0.0] = np.nan
        # mean_advantages = np.nanmean(advantages_copy)
        # std_advantages = np.nanstd(advantages_copy)
        # advantages = (advantages - mean_advantages) / (std_advantages + 1e-6)
        mean_advantages = np.nanmean(advantages_copy, axis=1)
        std_advantages = np.nanstd(advantages_copy, axis=1)
        advantages = (advantages - np.repeat(mean_advantages[:, np.newaxis], 30, axis=1)) / (np.repeat(std_advantages[:, np.newaxis], 30, axis=1) + 1e-6)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        # train_info['dist_entropy'] = 0
        # train_info['actor_grad_norm'] = 0
        # train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['return_norm_mean'] = 0
        train_info['value_mean'] = 0

        # for _ in range(self.ppo_epoch):
        #     if self._use_recurrent_policy:
        #         data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        #     else:
        #         data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        #     for sample in data_generator:
        #         value_loss, _, policy_loss, _, _, imp_weights, return_norm_mean, value_mean = self.update(sample, agent_index, update_actor)

        #         train_info['value_loss'] += value_loss
        #         train_info['policy_loss'] += policy_loss
        #         train_info['ratio'] += _t2n(torch.mean(imp_weights))
        #         train_info['return_norm_mean'] += return_norm_mean
        #         train_info['value_mean'] += value_mean

        # num_updates = self.ppo_epoch * self.num_mini_batch

        # for k in train_info.keys():
        #     train_info[k] /= num_updates

        if self._use_recurrent_policy:
            buffer.insert_data_buffer_recurrent(advantages)
        else:
            buffer.insert_data_buffer(advantages)

        if buffer.filled_i >= buffer.data_buffer_length / 2:
            for _ in range(self.ppo_epoch):
                if self._use_recurrent_policy:
                    sample = buffer.sample_recurrent()
                else:
                    sample = buffer.sample()
                value_loss, _, policy_loss, _, _, imp_weights, return_norm_mean, value_mean = self.update(sample, agent_index, update_actor)

                train_info['value_loss'] += value_loss
                train_info['policy_loss'] += policy_loss
                train_info['ratio'] += _t2n(torch.mean(imp_weights))
                train_info['return_norm_mean'] += return_norm_mean
                train_info['value_mean'] += value_mean

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        for a in self.agents:
            a.actor.train()
            a.critic.train()

    def prep_rollout(self):
        for a in self.agents:
            a.actor.eval()
            a.critic.eval()

    @torch.no_grad()
    def step(self, state_info, indexs, num_lr, training_index, is_deterministic):
        obs_input, cent_obs_input, masks_input = state_info

        values = np.zeros((num_lr, 1), dtype=np.float32)
        act_dim = 2 if self.action_space_class == 'Continuous' else 1
        actions = np.zeros((num_lr, act_dim), dtype=np.float32)
        action_log_probs = np.zeros((num_lr, 1), dtype=np.float32)
        for agent_i in training_index:
            agent = self.agents[agent_i]
            (value, action, action_log_prob) = agent.get_action_value(cent_obs_input[:,indexs[agent_i]].T,
                                                                    obs_input[:,indexs[agent_i]].T,
                                                                    masks_input[indexs[agent_i]],
                                                                    is_deterministic)
            values[indexs[agent_i]] = _t2n(value)
            actions[indexs[agent_i]] = _t2n(action)
            action_log_probs[indexs[agent_i]] = _t2n(action_log_prob)

        return values, actions, action_log_probs

    @torch.no_grad()
    def compute(self, agent_buffer, agent_i):
        """Calculate returns for the collected data."""
        self.prep_rollout()
        # for i in range(agent_buffer.num_agents):
        #     next_values = self.agents[agent_i].get_values(agent_buffer.cent_obs_buffer[agent_buffer.stop_index[i] + 1,[i],:],
        #                                                   agent_buffer.masks_buffer[agent_buffer.stop_index[i] + 1,[i],:])
        #     next_values = _t2n(next_values)
        #     agent_buffer.value_preds_buffer[agent_buffer.stop_index[i] + 1,[i],:] = next_values

        next_values = self.agents[agent_i].get_values(agent_buffer.cent_obs_buffer[-1],agent_buffer.masks_buffer[-1])
        next_values = _t2n(next_values)
        agent_buffer.value_preds_buffer[-1] = next_values

        agent_buffer.compute_returns(self.value_normalizer)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training()  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)
        
    @classmethod      
    def init_from_env(cls, env, args, device):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        dim_input_policy = env.observation_space.shape[0]
        dim_output_policy = env.action_space.shape[0]
        if args.env_name == 'adversarial':
            dim_input_critic = env.observation_space.shape[0] + 4 * (args.n_l + args.n_r) if args.use_centralized_V else env.observation_space.shape[0]
        else:
            dim_input_critic = env.observation_space.shape[0] + 4 * (args.n_a) if args.use_centralized_V else env.observation_space.shape[0]

        alg_types = [args.adversary_alg if atype == 'adversary' else args.agent_alg for atype in env.agent_types]   
                     
        for algtype in alg_types:  
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_input_critic': dim_input_critic,
                                      'dim_output_policy': dim_output_policy,})

        init_dict = {'args': args, 'agent_init_params': agent_init_params, 'device': device}
        instance = cls(**init_dict)    
        instance.init_dict = init_dict
        return instance

    @classmethod      
    def init_from_high_env(cls, env, args, device):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        dim_input_policy = env.observation_space_high.shape[0]
        dim_output_policy = env.action_space_high.shape[0] if args.action_space_class == 'Continuous' else env.action_space_high.n
        dim_input_critic = env.observation_space_high.shape[0]

        alg_types = [args.adversary_alg if atype == 'adversary' else args.agent_alg for atype in env.agent_types]   
                     
        for algtype in alg_types:  
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_input_critic': dim_input_critic,
                                      'dim_output_policy': dim_output_policy,})

        init_dict = {'args': args, 'agent_init_params': agent_init_params, 'device': device}
        instance = cls(**init_dict)    
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):    
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

    @classmethod
    def init_from_save_with_id(cls, filename, list_id):    
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for i in range(len(instance.agents)):
            a = instance.agents[i]
            policy_id = list_id[i]
            if policy_id == 2:
                continue
            params = save_dict['agent_params'][policy_id]
            a.load_params(params)
        return instance