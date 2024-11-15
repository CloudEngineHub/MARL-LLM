import torch
from torch.optim import Adam
from algorithm.utils.networks import Discriminator
import torch.nn.functional as F
import numpy as np

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (0.6 * epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AIRL(object):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_num, lr_discriminator, expert_buffer, batch_size=512, gamma=0.95, device='cpu'):
        device = 'cuda' if device == 'gpu' else 'cpu'
        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            hidden_dim=hidden_dim,
            hidden_num=hidden_num
        ).to(device)

        self.device = device
        self.lr_discriminator = lr_discriminator
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.expert_buffer = expert_buffer
        self.batch_size = batch_size
        self.n_iter = 0

    def update(self, states, actions, log_pis, next_states, dones = 0, logger=None):
        # Samples from current policy's trajectories.
        # states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
        
        # states = check(states).to(self.device)
        # actions = check(actions).to(self.device)
        # log_pis = check(log_pis).to(self.device)
        # next_states = check(next_states).to(self.device)

        # Samples from expert's demonstrations.
        states_exp, actions_exp, next_states_exp, dones_exp = self.expert_buffer.sample(6*self.batch_size, to_gpu=True if self.device == 'cuda' else False)

        # Calculate log probabilities of expert actions.
        with torch.no_grad():
            # log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)
            # log_pis_exp = torch.zeros((states_exp.size(0), 1), device=states_exp.device)
            log_pis_exp = torch.full((states_exp.size(0), 1), -actions_exp.size(1) * np.log(1), device=states_exp.device)

        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.discriminator(states, actions, log_pis, next_states, dones)
        logits_exp = self.discriminator(states_exp, actions_exp, log_pis_exp, next_states_exp, dones_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_discriminator = loss_pi + loss_exp

        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        # Discriminator's accuracies.
        with torch.no_grad():
            acc_pi = (logits_pi < 0).float().mean().item()
            acc_exp = (logits_exp > 0).float().mean().item()
        logger.add_scalars('agent0/losses', {'loss_discriminator': loss_discriminator, 'accuracy_pi': acc_pi, 'accuracy_exp': acc_exp}, self.n_iter)
        self.n_iter += 1

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.discriminator_optimizer, episode, episodes, self.lr_discriminator)

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename)

    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename))