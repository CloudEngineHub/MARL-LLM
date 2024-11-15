import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class FormationSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(FormationSwarmWrapper, self).__init__(env)
        env.__reinit__(args)
        self.num_agents = self.env.n_a
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.agent_types = ['agent']
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('Formation environment initialized successfully.')

    def coverage_rate(self):
        num_occupied_grid = 0
        for grid_i in range(self.n_g):
            grid_pos_rel = self.p - self.grid_center[:,[grid_i]]
            grid_pos_rel_norm = np.linalg.norm(grid_pos_rel, axis=0)
            if (grid_pos_rel_norm < self.r_avoid/2).any():
                num_occupied_grid += 1

        metric_1 = num_occupied_grid / self.n_g
        return metric_1

    def distribution_uniformity(self):
        min_dist = []
        for agent_i in range(self.n_a):
            agent_pos_rel = self.p - self.p[:,[agent_i]]
            agent_pos_rel_norm = np.linalg.norm(agent_pos_rel, axis=0)
            non_zero_elements = agent_pos_rel_norm[agent_pos_rel_norm != 0]
            min_dist_i = np.min(non_zero_elements)
            min_dist.append(min_dist_i)
        
        min_val = np.min(min_dist)
        max_val = np.max(min_dist)
        min_dist = (min_dist - min_val) / (max_val - min_val)
        metric_2 = np.var(min_dist)
        return metric_2

