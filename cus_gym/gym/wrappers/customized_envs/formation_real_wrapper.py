import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class FormationRealSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(FormationRealSwarmWrapper, self).__init__(env)
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
        
        # min_val = np.min(min_dist)
        # max_val = np.max(min_dist)
        # min_dist = 2 * (min_dist - min_val) / (max_val - min_val) - 1
        # metric_2 = np.var(min_dist)

        uniform = np.var(min_dist)
        metric_2 = (uniform - np.min(min_dist)) / (np.max(min_dist) - np.min(min_dist))

        return metric_2
    
    def voronoi_based_uniformity(self):
        num_grid_in_voronoi = np.zeros(self.n_a)
        for cell_index in range(self.grid_center.shape[1]):
            rel_pos_cell_nei = self.p - self.grid_center[:,[cell_index]]
            rel_pos_cell_nei_norm = np.linalg.norm(rel_pos_cell_nei, axis=0)
            min_index = np.argmin(rel_pos_cell_nei_norm)
            num_grid_in_voronoi[min_index] += 1

        # min_val = np.min(num_grid_in_voronoi)
        # max_val = np.max(num_grid_in_voronoi)
        # uniform = 2 * (num_grid_in_voronoi - min_val) / (max_val - min_val) - 1
        # metric_3 = np.var(uniform)

        uniform = np.var(num_grid_in_voronoi)
        metric_3 = (uniform - np.min(num_grid_in_voronoi)) / (np.max(num_grid_in_voronoi) - np.min(num_grid_in_voronoi))

        return metric_3

        

