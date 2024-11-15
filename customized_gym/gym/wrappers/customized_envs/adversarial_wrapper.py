import gym
import numpy as np

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class AdversarialSwarmWrapper(gym.Wrapper):

    def __init__(self, env, args):
        super(AdversarialSwarmWrapper, self).__init__(env)
        self.env.n_l = args.n_l
        self.env.n_r = args.n_r
        self.env.l_strategy = args.l_strategy
        self.env.r_strategy = args.r_strategy
        self.env.dynamics_mode = args.dynamics_mode
        self.env.render_traj = args.render_traj
        self.env.traj_len = args.traj_len
        self.env.billiards_mode = args.billiards_mode

        self.num_r = args.n_r
        self.num_l = args.n_l

        self.agents = [Agent() for _ in range(self.num_r)] + [Agent(adversary=True) for _ in range(self.num_l)]
        self.agent_types = ['adversary', 'agent']
        env.__reinit__(args)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('Adversarial environment initialized successfully.')

    def dos_and_doa(self, x, h, T, N, D):    
        """
        Args:
            x: agent的位置列表。
            T: 回合数。
            N: 代理总数。
            D: 环境大小。
        """
        k = [0] * (self.num_prey)
        k_h = [0] * (self.num_prey)
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]
        for t in range(np.shape(x)[2]):
            for j in range(np.shape(x)[1]):
                k[j] = self._find_nearest_neighbors_DOS(x[:, :, t], j)   # k[j] 表明的是第j个agent的 nearest neighbor
                k_h[j] = self._find_nearest_neighbors_DOA(h[:, :, t], j)
                distances.append(k[j]) 
                distances_h.append(k_h[j])

        DOS = np.sum(distances) / (T * N * D)
        DOA = np.sum(distances_h) / (2 * T * N)
        return DOS, DOA
    
    def dos_and_doa_one_episode(self, x, h, N, D):
        k = [0] * (self.num_prey)
        k_h = [0] * (self.num_prey)
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]                                               
        for i in range(np.shape(x)[1]):  
            k[i] = self._find_nearest_neighbors_DOS(x, i)   
            k_h[i] = self._find_nearest_neighbors_DOA(h, i)
            distances.append(k[i]) 
            distances_h.append(k_h[i])

        DOS = np.sum(distances) / (N * D)
        DOA = np.sum(distances_h) / (2 * N)
        return DOS, DOA
    
    def _find_nearest_neighbors_DOS(self, x, i):

        distances = []
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] - x[:, j]))

        return np.min(distances)
    
    def _find_nearest_neighbors_DOA(self, x, i):

        distances = []
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] + x[:, j]))

        return np.min(distances)
