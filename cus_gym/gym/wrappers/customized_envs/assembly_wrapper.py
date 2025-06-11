import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Agent:
    """
    Simple agent class for multi-agent environments.
    """
    def __init__(self, adversary=False):
        """
        Initialize agent with optional adversary flag.
        
        Args:
            adversary (bool): Whether this agent is adversarial
        """
        self.adversary = adversary

class AssemblySwarmWrapper(gym.Wrapper):
    """
    Gym wrapper for assembly swarm environments with multi-agent support.
    
    Provides metrics for evaluating swarm coverage, distribution uniformity,
    and spatial arrangement quality.
    """
    
    def __init__(self, env, args):
        """
        Initialize the assembly swarm wrapper.
        
        Args:
            env: Base gym environment to wrap
            args: Configuration arguments for environment
        """
        super(AssemblySwarmWrapper, self).__init__(env)
        env.__reinit__(args)
        
        # Initialize multi-agent components
        self.num_agents = self.env.n_a
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.agent_types = ['agent']
        
        # Set up spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        print('Assembly environment initialized successfully.')

    def coverage_rate(self):
        """
        Calculate the coverage rate of agents over the grid.
        
        Measures what fraction of grid cells are occupied by agents
        within the avoidance radius.
        
        Returns:
            float: Coverage rate (0 to 1)
        """
        num_occupied_grid = 0
        
        # Check each grid cell for agent occupation
        for grid_i in range(self.n_g):
            # Calculate relative positions of agents to current grid cell
            grid_pos_rel = self.p - self.grid_center[:, [grid_i]]
            grid_pos_rel_norm = np.linalg.norm(grid_pos_rel, axis=0)
            
            # Check if any agent is within avoidance radius of this grid cell
            if (grid_pos_rel_norm < self.r_avoid / 2).any():
                num_occupied_grid += 1

        # Calculate coverage as fraction of occupied grid cells
        metric_1 = num_occupied_grid / self.n_g
        return metric_1

    def distribution_uniformity(self):
        """
        Calculate distribution uniformity based on minimum inter-agent distances.
        
        Measures how uniformly agents are distributed by analyzing variance
        in minimum distances between agents.
        
        Returns:
            float: Normalized uniformity metric (0 to 1)
        """
        min_dist = []
        
        # Calculate minimum distance for each agent to its nearest neighbor
        for agent_i in range(self.n_a):
            # Get relative positions to current agent
            agent_pos_rel = self.p - self.p[:, [agent_i]]
            agent_pos_rel_norm = np.linalg.norm(agent_pos_rel, axis=0)
            
            # Remove zero distance (agent to itself)
            non_zero_elements = agent_pos_rel_norm[agent_pos_rel_norm != 0]
            min_dist_i = np.min(non_zero_elements)
            min_dist.append(min_dist_i)

        # Calculate normalized variance of minimum distances
        uniform = np.var(min_dist)
        metric_2 = (uniform - np.min(min_dist)) / (np.max(min_dist) - np.min(min_dist))

        return metric_2

    def voronoi_based_uniformity(self):
        """
        Calculate uniformity based on Voronoi cell distribution.
        
        Assigns each grid cell to the nearest agent and measures uniformity
        based on the variance in number of cells per agent.
        
        Returns:
            float: Normalized Voronoi uniformity metric (0 to 1)
        """
        num_grid_in_voronoi = np.zeros(self.n_a)
        
        # Assign each grid cell to nearest agent (Voronoi partitioning)
        for cell_index in range(self.grid_center.shape[1]):
            # Calculate distances from current grid cell to all agents
            rel_pos_cell_nei = self.p - self.grid_center[:, [cell_index]]
            rel_pos_cell_nei_norm = np.linalg.norm(rel_pos_cell_nei, axis=0)
            
            # Assign cell to nearest agent
            min_index = np.argmin(rel_pos_cell_nei_norm)
            num_grid_in_voronoi[min_index] += 1

        # Calculate normalized variance of grid cells per agent
        uniform = np.var(num_grid_in_voronoi)
        metric_3 = (uniform - np.min(num_grid_in_voronoi)) / (np.max(num_grid_in_voronoi) - np.min(num_grid_in_voronoi))

        return metric_3