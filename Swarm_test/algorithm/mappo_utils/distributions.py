import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def sample_uniform(self):
        uniform_probs = torch.ones(self._num_events, device='cuda:0') / self._num_events
        probs_2d = uniform_probs.unsqueeze(0)
        samples_2d = torch.multinomial(probs_2d, 1, replacement=True).T
        return samples_2d
    
    def sample_topk(self):
        k = 1
        topk_probs, topk_indices = self.probs.topk(k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Normalize
        sampled_idx = torch.multinomial(topk_probs, 1, True).squeeze(-1)
        return topk_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)

    def log_probs(self, actions):
        return (super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1))

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        # self.linear = nn.Linear(num_inputs, num_outputs)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        # x = self.activate(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        # self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_mean = nn.Sequential(init_(nn.Linear(num_inputs, num_outputs)), nn.Tanh())
        # self.fc_mean = nn.Sequential(nn.Linear(num_inputs, num_outputs), nn.Tanh())
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # action_mean = torch.tanh(action_mean)  # 添加 tanh 激活函数

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
