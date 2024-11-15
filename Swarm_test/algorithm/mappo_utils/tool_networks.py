import torch
import torch.nn as nn
from .util import init, get_clones
from .distributions import DiagGaussian, Categorical

"""MLP modules."""
class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_N, use_orthogonal, active_func_index):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU()][active_func_index]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu'][active_func_index])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_dim)), active_func)
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_dim, hidden_dim)), active_func)
        # self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func)
        # self.fc_h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), active_func)
        # self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim))
        # self.fc_h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class MLPBase(nn.Module):
    def __init__(self, args, obs_dim):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self.activate_func_index = args.activate_func_index
        self._layer_N = args.layer_N
        self.hidden_dim = args.hidden_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_dim, self._layer_N, self._use_orthogonal, self.activate_func_index)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
            # x = self.normalize_0_1(x)

        x = self.mlp(x)

        return x

    def normalize_0_1(self, x):
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)

"""RNN modules."""
class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            # if 'bias' in name:
            #     nn.init.constant_(param, 0)
            if 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        # self.norm = nn.LayerNorm(outputs_dim)
        self.norm = outputs_dim

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        # x = self.norm(x)
        return x, hxs

"""ACT modules."""
class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space_class, inputs_dim, action_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()

        if action_space_class == "Discrete":
            self.continuous_action = False
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space_class == "Continuous":
            self.continuous_action = True
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
    
    def forward(self, x, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.continuous_action:
            action_logit = self.action_out(x)
            actions = action_logit.mode() if deterministic else action_logit.sample()
            # actions = actions.clamp(-1, 1)
            action_log_probs = action_logit.log_probs(actions)
            # actions = actions.clamp(-1, 1)
        else:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :return action_probs: (torch.Tensor)
        """
        action_logits = self.action_out(x)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.continuous_action:
            action_log_probs = []
            dist_entropy = []
            action_logit = self.action_out(x)
            action_log_probs.append(action_logit.log_probs(action))
            dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0]
        else:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy