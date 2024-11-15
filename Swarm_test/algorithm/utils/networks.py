import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)    
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out
    
class MLPNetworkRew(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu, constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetworkRew, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) 
        self.hidden_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(1)])
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        x = self.nonlin(self.fc1(X))
        for block in self.hidden_blocks:
            x = block(x)
        out = self.out_fn(self.fc4(x))

        # h1 = self.nonlin(self.fc1(X))
        # h2 = self.nonlin(self.fc2(h1))
        # h3 = self.nonlin(self.fc3(h2))
        # out = self.out_fn(self.fc4(h3))
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, nonlin=F.leaky_relu):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = nonlin

    def forward(self, x):
        residual = x
        x = self.nonlin(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.nonlin(x)
        return x
    
class Discriminator(nn.Module):

    def __init__(self, state_dim, action_dim, gamma, hidden_dim, hidden_num):
        super(Discriminator, self).__init__()

        self.g = MLPUnit(
           input_dim=state_dim + action_dim,
           out_dim=1,
           hidden_dim=hidden_dim,
           hidden_num=hidden_num
        )
        self.h = MLPUnit(
           input_dim=state_dim,
           out_dim=1,
           hidden_dim=hidden_dim,
           hidden_num=hidden_num
        )

        self.gamma = gamma

    def f(self, states, actions, next_states, dones = 0):
        # rs = self.g(states)
        rs = self.g(torch.cat((states, actions), dim=1))
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, actions, log_pis, next_states, dones):
        # Discriminator's output is sigmoid(f - log_pi).
        # return self.f(states, next_states, dones) - log_pis
        return self.f(states, actions, next_states, dones) - log_pis

    def calculate_reward(self, states, actions, log_pis, next_states, dones):
        with torch.no_grad():
            # logits = self.forward(states, actions, log_pis, next_states, dones)
            logits = self.f(states, actions, next_states, dones)
            # logits = self.g(states)
            # logits = self.g(torch.cat((states, actions), dim=1))
            # return -F.logsigmoid(-logits)
            # return F.logsigmoid(logits) - F.logsigmoid(-logits)
            return logits
        
class MLPUnit(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, hidden_num, hidden_activation=nn.LeakyReLU(), output_activation=None):
        """
        Parameters:
            input_dim (int): 输入层的维度
            hidden_dim (int): 每个隐藏层的神经元数量
            out_dim (int): 输出层的维度
            hidden_activation (torch.nn.Module): 隐藏层的激活函数，默认是ReLU
            output_activation (torch.nn.Module or None): 输出层的激活函数，默认为无激活
        """
        super(MLPUnit, self).__init__()
        
        layers = []
        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if hidden_activation:
            layers.append(hidden_activation)
        # hidden layer
        for i in range(1, hidden_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if hidden_activation:
                layers.append(hidden_activation)
        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation:
            layers.append(output_activation)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class MultiHeadMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim_1, input_dim_2, out_dim, hidden_dim_1=64, hidden_dim_2=128, nonlin=F.leaky_relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MultiHeadMLPNetwork, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1_1 = nn.Linear(input_dim_1, hidden_dim_1)
        self.fc2_1 = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.fc3_1 = nn.Linear(hidden_dim_1, hidden_dim_1)   

        self.fc1_2 = nn.Linear(input_dim_2, hidden_dim_2)
        self.fc2_2 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.fc3_2 = nn.Linear(hidden_dim_2, hidden_dim_2)

        self.fc4 = nn.Linear(hidden_dim_1 + hidden_dim_2, out_dim)

        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1_1 = self.nonlin(self.fc1_1(X[:,:self.input_dim_1]))
        h2_1 = self.nonlin(self.fc2_1(h1_1))
        h3_1 = self.nonlin(self.fc3_1(h2_1))

        h1_2 = self.nonlin(self.fc1_2(X[:,self.input_dim_1:]))
        h2_2 = self.nonlin(self.fc2_2(h1_2))
        h3_2 = self.nonlin(self.fc3_2(h2_2))

        h3_concat = torch.cat((h3_1, h3_2), dim=1)
        out = self.out_fn(self.fc4(h3_concat))
        return out

class SymmetricNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SymmetricNetwork, self).__init__() 

        # self.in_fn = lambda x: x
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin

        self.fc1_1 = nn.Linear(4, self.hidden_dim)
        self.fc2_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc1_2 = nn.Linear(4, self.hidden_dim)
        self.fc2_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc1_3 = nn.Linear(2, self.hidden_dim)
        self.fc2_3 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc3 = nn.Linear(3*self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, out_dim)

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        out = torch.zeros((X.shape[0], 2)).to(X.device)
        for agent_i in range(50):
            h_split = X[agent_i,:4*7].view(-1, 4)
            non_zero_segments = torch.sum(torch.any(h_split != 0, dim=1))
            sum_neigh = torch.zeros(self.hidden_dim).to(X.device)
            for nei_index in range(non_zero_segments):
                h1_1 = self.nonlin(self.fc1_1(X[agent_i,4 * nei_index:4 * (nei_index + 1)]))
                h2_1 = self.nonlin(self.fc2_1(h1_1))
                sum_neigh += h2_1

            h1_2 = self.nonlin(self.fc1_2(X[agent_i, 28:32]))
            h2_2 = self.nonlin(self.fc2_2(h1_2))

            h_split = X[agent_i,4*8:].view(-1, 2)
            non_zero_segments = torch.sum(torch.any(h_split != 0, dim=1))
            sum_grid = torch.zeros(self.hidden_dim).to(X.device)
            for grid_index in range(non_zero_segments):
                h1_3 = self.nonlin(self.fc1_3(X[agent_i,2 * grid_index + 32:2 * (grid_index + 1) + 32]))
                h2_3 = self.nonlin(self.fc2_3(h1_3))
                sum_grid += h2_3

            h_cat = torch.cat((sum_neigh, h2_2, sum_grid))
            h3 = self.nonlin(self.fc3(h_cat))
            h4 = self.nonlin(self.fc4(h3))
            out[agent_i] = self.out_fn(self.fc5(h4))

        return out