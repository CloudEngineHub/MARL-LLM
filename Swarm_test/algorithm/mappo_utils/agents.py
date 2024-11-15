import torch
from .actor_critic import Actor, Critic
from .util import update_linear_schedule

class PPOAgent:
    """
    MAPPO Agent  class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, args, dim_input_policy, dim_input_critic, dim_output_policy, device=torch.device("cpu")):
        self.device = device
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_dim = dim_input_policy
        self.cent_obs_dim = dim_input_critic
        self.act_dim = dim_output_policy

        self.actor = Actor(args, self.obs_dim, self.act_dim, self.device)
        self.critic = Critic(args, self.cent_obs_dim, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr_actor, 
                                                eps=self.opti_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr_critic,
                                                 eps=self.opti_eps)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        #                                         lr=self.lr_actor, eps=self.opti_eps,
        #                                         weight_decay=self.weight_decay)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
        #                                          lr=self.lr_critic,
        #                                          eps=self.opti_eps,
        #                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr_actor)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.lr_critic)

    def get_action_value(self, cent_obs, obs, masks, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        """
        actions, action_log_probs = self.actor(obs, masks, deterministic)
        values = self.critic(cent_obs, masks)

        return values, actions, action_log_probs

    def get_values(self, cent_obs, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :return values: (torch.Tensor) value function predictions.
        """
        values = self.critic(cent_obs, masks)

        return values

    def evaluate_action_value(self, cent_obs, obs, action, masks):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, masks)
        values = self.critic(cent_obs, masks)

        return values, action_log_probs, dist_entropy

    def step(self, obs, masks, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _ = self.actor(obs, masks, deterministic)

        return actions

    def get_params(self):
        return {'policy': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'policy_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.actor_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])