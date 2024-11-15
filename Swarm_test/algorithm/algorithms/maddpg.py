import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from algorithm.utils.networks import MLPNetwork
from algorithm.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from algorithm.utils.agents import DDPGAgent
import math

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise, gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, lambda_s=500, epsilon_p=0.06,  
                 hidden_dim=64, device='cpu', discrete_action=False, topo_nei_max=6, is_con_self=False, is_con_remark_leader=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                dim_input_policy (int): Input dimensions to policy
                dim_output_policy (int): Output dimensions to policy
                dim_input_policy (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise
        # 每个 agent 除了 agent inital parameters 不一样之外，都是同构的
        self.agents = [DDPGAgent(lr_actor=lr_actor, 
                                 lr_critic=lr_critic, 
                                 discrete_action=discrete_action, 
                                 hidden_dim=hidden_dim, 
                                 epsilon=self.epsilon, 
                                 noise=self.noise,
                                 topo_nei_max=topo_nei_max,
                                 is_con_self=is_con_self,
                                 is_con_remark_leader=is_con_remark_leader,
                                 **params) for params in agent_init_params]   
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lambda_s = lambda_s
        self.epsilon_p = epsilon_p
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  
        self.critic_dev = 'cpu' 
        self.trgt_pol_dev = 'cpu' 
        self.trgt_critic_dev = 'cpu'

        self.rew_pol_dev = 'cpu'  
        self.rew_critic_dev = 'cpu' 
        self.rew_trgt_pol_dev = 'cpu' 
        self.rew_trgt_critic_dev = 'cpu'

        self.spatial_loss = False
        self.temporal_loss = False 
        self.niter = 0

    @property           
    def policies(self):
        return [a.policy for a in self.agents]

    def target_policies(self, agent_i, obs):
        return self.agents[agent_i].target_policy(obs)

    def scale_noise(self, scale, new_epsilon):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)       
            a.epsilon = new_epsilon

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """  
        actions = []      
        log_pis = []          
        for i in range(len(start_stop_num)):
            action, log_pi = self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore)
            actions.append(action)
            log_pis.append(log_pi)
        return actions, log_pis
        # actions = []             
        # for i in range(len(start_stop_num)):
        #     action = self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore)
        #     actions.append(action)
        # return actions
    
    def step_rew(self, observations, start_stop_num):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            rewards: List of intrinsic rewards for each agent
        """                                                           
        return [self.agents[i].step_rew(observations[:, start_stop_num[i]].t()) for i in range(len(start_stop_num))]

    # def update(self, obs, acs, log_pis, rews, next_obs, dones, agent_i, airl, parallel=False, logger=None):
    def update(self, obs, acs, rews, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # obs, acs, rews, next_obs, dones = sample            
        curr_agent = self.agents[agent_i]    

        ######################### update critic #########################       
        curr_agent.critic_optimizer.zero_grad()     
        all_trgt_acs = self.target_policies(agent_i, next_obs)  
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1) 

        # no shaping reward 
        target_value = (rews + self.gamma * curr_agent.target_critic(trgt_vf_in) *  (1 - dones))                                           
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())

        # # shaping reward 1
        # vf_in = torch.cat((obs, acs), dim=1)
        # rew_rs = curr_agent.reward_rs(vf_in)
        # target_value = (rews + 0.2 * rew_rs + self.gamma * curr_agent.target_critic(trgt_vf_in) *  (1 - dones))    
        # actual_value = curr_agent.critic(vf_in)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        # vf_in = torch.cat((obs, acs), dim=1)
        # rew_rs = curr_agent.reward_rs(vf_in)
        # next_rew_rs = curr_agent.reward_rs(trgt_vf_in)
        # target_value = (rews + self.gamma * next_rew_rs - rew_rs + self.gamma * curr_agent.target_critic(trgt_vf_in) *  (1 - dones))    
        # actual_value = curr_agent.critic(vf_in)
        # vf_loss = MSELoss(actual_value, target_value.detach())

        # shaping reward 2
        # vf_in = torch.cat((obs, acs), dim=1)
        # rew_rs = curr_agent.reward_rs(vf_in)
        # target_value = (rew_rs + self.gamma * curr_agent.target_critic(trgt_vf_in) *  (1 - dones))    
        # actual_value = curr_agent.critic(vf_in)
        # vf_loss = MSELoss(actual_value, target_value.detach())

        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        # torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        ######################### update actor #########################
        curr_agent.policy_optimizer.zero_grad()  

        if not self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = curr_pol_vf_in  
        vf_in = torch.cat((obs, all_pol_acs), dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()

        # logits_pi = airl.discriminator(obs, acs, log_pis, next_obs, dones)
        # loss_pi = -F.logsigmoid(logits_pi).mean()

        # pol_loss += loss_pi

        # # Add Lipschitz regularization
        # lipschitz_loss = 0
        # epsilon = 0.01  # small perturbation magnitude
        # obs_perturbed = obs + epsilon * torch.randn_like(obs)  # small random perturbation
        # pol_out_perturbed = curr_agent.policy(obs_perturbed)
        # lipschitz_loss = ((curr_pol_out - pol_out_perturbed).norm(p=2) / epsilon).mean()
        # lipschitz_lambda = 0.1  # weight for the Lipschitz regularization term

        # pol_loss += lipschitz_lambda * lipschitz_loss
        
        spat_act_loss = 0
        # if self.spatial_loss:
            # obs_pert = (2 *torch.rand_like(obs) - 1) * self.epsilon_p + obs
            # # obs_pert = torch.randn_like(obs) * self.epsilon_p + obs
            # curr_pol_out_pert = curr_agent.policy(obs_pert)
            # spat_act_loss = self.lambda_s * MSELoss(all_pol_acs, curr_pol_out_pert)
        
        temp_act_loss = 0
        # if self.temporal_loss:
        #     next_pol_out = curr_agent.policy(next_obs)
        #     temp_act_loss = 5 * MSELoss(all_pol_acs, next_pol_out)

        all_pol_loss = pol_loss + spat_act_loss + temp_act_loss
        all_pol_loss.backward()     
                                
        if parallel:
            average_gradients(curr_agent.policy)
        # torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)   
        curr_agent.policy_optimizer.step() 
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {'vf_loss': vf_loss, 'pol_loss': pol_loss}, self.niter)

    def update_rew(self, obs, rew_rs, rew_bar, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        """            
        curr_agent = self.agents[agent_i]    

        ######################### update critic #########################       
        curr_agent.reward_rs_critic_optimizer.zero_grad()     
        all_trgt_rews = curr_agent.target_reward_rs(next_obs)  
        trgt_vf_in = torch.cat((next_obs, all_trgt_rews), dim=1)  
        target_value = (rew_bar + 0.99 * curr_agent.target_reward_rs_critic(trgt_vf_in) *  (1 - dones))                                               
        vf_in = torch.cat((obs, rew_rs), dim=1)
        actual_value = curr_agent.reward_rs_critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach()) 

        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.reward_rs_critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.reward_rs_critic.parameters(), 0.5)
        curr_agent.reward_rs_critic_optimizer.step()

        ######################### update actor #########################
        curr_agent.reward_rs_optimizer.zero_grad()  

        all_pol_acs = curr_agent.reward_rs(obs) 
        vf_in = torch.cat((obs, all_pol_acs), dim=1)
        pol_loss = -curr_agent.reward_rs_critic(vf_in).mean()
        pol_loss.backward()     
                                
        if parallel:
            average_gradients(curr_agent.reward_rs)
        torch.nn.utils.clip_grad_norm_(curr_agent.reward_rs.parameters(), 0.5)   
        curr_agent.reward_rs_optimizer.step() 

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {'vf_loss_rew': vf_loss, 'pol_loss_rew': pol_loss}, self.niter)


    def update_rew2(self, trj_batch, agent_i, logger=None):
        curr_agent = self.agents[agent_i]
        curr_agent.reward_rs_optimizer.zero_grad()

        loss = 0
        for i, trj1 in enumerate(trj_batch):
            for j, trj2 in enumerate(trj_batch):
                if self.distance(ter1=trj1, ter2=trj2, gamma=self.gamma) > 0:  # I(T1 > T2)
                    transition1_tensor = torch.stack(trj1[3], dim=0)  # already a tensor
                    transition2_tensor = torch.stack(trj2[3], dim=0)  # already a tensor
                else:  # 1 - I(T1 > T2)
                    transition1_tensor = torch.stack(trj2[3], dim=0)
                    transition2_tensor = torch.stack(trj1[3], dim=0)

                # Forward pass through reward network
                reward1 = curr_agent.reward_rs(transition1_tensor).squeeze()
                reward2 = curr_agent.reward_rs(transition2_tensor).squeeze()

                # Compute loss
                # G1 = torch.sum(torch.tensor([self.gamma ** i for i in range(len(reward1))], device=reward1.device) * reward1)
                # G2 = torch.sum(torch.tensor([self.gamma ** i for i in range(len(reward2))], device=reward2.device) * reward2)
                G1 = torch.sum(reward1)
                G2 = torch.sum(reward2)
                temp = -1 / (1 + torch.exp(G2 - G1))
                loss += temp

        # loss /= len(trj_batch) # Normalizing loss?
        loss.backward()
        curr_agent.reward_rs_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {'loss_rew': loss}, self.niter)

    def distance(self, ter1, ter2, gamma):
        # Assuming ter1[2] and ter2[2] are tensors
        # R1 = torch.sum(torch.tensor([gamma ** i * ter1[2][i] for i in range(len(ter1[2]))], device=ter1[2][0].device))
        # R2 = torch.sum(torch.tensor([gamma ** i * ter2[2][i] for i in range(len(ter2[2]))], device=ter2[2][0].device))
        R1 = torch.sum(torch.stack(ter1[2]))
        R2 = torch.sum(torch.stack(ter2[2]))
        return R1 >= R2

    def update_all_targets(self):    
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)   
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def update_all_targets_rew(self):    
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_reward_rs_critic, a.reward_rs_critic, self.tau)   
            soft_update(a.target_reward_rs, a.reward_rs, self.tau)
        # self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()  
            a.target_policy.train()
            a.target_critic.train()

            a.reward_rs.train()  
            a.target_reward_rs.train()
            a.target_reward_rs_critic.train()

        # device transform
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

        if not self.rew_pol_dev == device:
            for a in self.agents:
                a.reward_rs = fn(a.reward_rs)
            self.rew_pol_dev = device
        if not self.rew_critic_dev == device:
            for a in self.agents:
                a.reward_rs_critic = fn(a.reward_rs_critic)
            self.rew_critic_dev = device
        if not self.rew_trgt_pol_dev == device:
            for a in self.agents:
                a.target_reward_rs = fn(a.target_reward_rs)
            self.rew_trgt_pol_dev = device
        if not self.rew_trgt_critic_dev == device:
            for a in self.agents:
                a.target_reward_rs_critic = fn(a.target_reward_rs_critic)
            self.rew_trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()   
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

        if not self.rew_pol_dev == device:
            for a in self.agents:
                a.reward_rs = fn(a.reward_rs)
            self.rew_pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod      
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG", gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, lambda_s=500, epsilon_p=0.06,  
                        hidden_dim=64, name='flocking', device='cpu', epsilon=0.1, noise=0.1):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        dim_input_policy=env.observation_space.shape[0]
        dim_output_policy=env.action_space.shape[0]
        dim_input_critic=env.observation_space.shape[0] + env.action_space.shape[0]

        # print("num in pol", dim_input_policy, "num out pol", dim_output_policy, "num in critic", dim_input_critic)

        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.agent_types]   
                     
        for algtype in alg_types:  
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_output_policy': dim_output_policy,
                                      'dim_input_critic': dim_input_critic})

        if name == 'flocking':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'lambda_s': lambda_s, 'epsilon_p': epsilon_p, 'epsilon': epsilon, 'noise': noise, 
                         'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params, 'topo_nei_max': env.topo_nei_max, 
                         'is_con_self': env.is_con_self_state, 'is_con_remark_leader': env.is_remarkable_leader}
        elif name == 'formation':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'lambda_s': lambda_s, 'epsilon_p': epsilon_p, 'epsilon': epsilon, 'noise': noise, 
                         'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params, 'topo_nei_max': env.topo_nei_max, 
                         'is_con_self': env.is_con_self_state}
        elif name == 'predator_prey':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'lambda_s': lambda_s, 'epsilon_p': epsilon_p, 'epsilon': epsilon, 'noise': noise, 
                         'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params}
        elif name == 'adversarial':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'lambda_s': lambda_s, 'epsilon_p': epsilon_p, 'epsilon': epsilon, 'noise': noise, 
                         'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params}
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