import torch
import torch.nn as nn
from .tool_networks import MLPBase, RNNLayer, ACTLayer
from .util import init, check, get_shape_from_obs_space

class Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.action_space_class = args.action_space_class
        self.hidden_dim = args.hidden_dim
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPBase(args, obs_dim)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_dim, self.hidden_dim, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(self.action_space_class, self.hidden_dim, action_dim, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, masks, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_recurrent_policy:
            actor_features = self.rnn(actor_features, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action, masks):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_recurrent_policy:
            actor_features = self.rnn(actor_features, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action)

        return action_log_probs, dist_entropy

class Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_dim, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_dim = args.hidden_dim
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.base = MLPBase(args, cent_obs_dim)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_dim, self.hidden_dim, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # self.v_out = init_(nn.Linear(self.hidden_dim, 1))
        self.v_out = nn.Linear(self.hidden_dim, 1)

        self.to(device)

    def forward(self, cent_obs, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_recurrent_policy:
            critic_features = self.rnn(critic_features, masks)
        values = self.v_out(critic_features)

        return values