__credits__ = ["zhugb@buaa.edu.cn"]

import gym
from gym import spaces
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from .VideoWriter import VideoWriter
import ctypes
import pickle
from .envs_cplus.c_lib import as_double_c_array, as_bool_c_array, as_int32_c_array, _load_lib

_LIB = _load_lib(env_name='Formation')

class FormationSwarmEnv(gym.Env):
    ''' A kind of MPE env.
    ball1 ball2 ball3 ...     (in order of pursuers, escapers and obstacles)
    sf = spring force,  df = damping force
    the forces include: u, ball(spring forces), aerodynamic forces
    x, dx: (2,n), position and vel of agent i  
    '''

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 45}
    def __init__(self):
        
        self.reward_sharing_mode = 'individual'   # select one from ['sharing_mean', 'sharing_max', 'individual'] 
        
        self.penalize_entering = True 
        self.penalize_interaction = True
        self.penalize_exploration = True 
        self.penalize_control_effort = False      
        self.penalize_collide_obstacles = False   
        self.penalize_collide_walls = False

        # dimension
        self.dim = 2

        # Numbers of agents
        self.n_a = 10   # number of agents
        self.n_o = 0     # number of obstacles

        # Observation 
        self.topo_nei_max = 6   # agent to agent 
        
        # Action
        self.act_dim_agent = self.dim
        
        # Mass
        self.m_a = 1     
        self.m_o = 10
        
        # Size  
        self.size_a = 0.035 
        self.size_o = 0.2
        
        # radius 
        self.d_sen = 3
        self.r_avoid = 0.15

        # physical constraint
        self.Vel_max = 0.8
        self.Vel_min = 0.0
        self.Acc_max = 1

        # Properties of obstacles
        self.obstacles_cannot_move = True 
        self.obstacles_is_constant = False
        if self.obstacles_is_constant:   # then specify their locations:
            self.p_o = np.array([[-0.5, 0.5],[0, 0]])
        ## ======================================== end ========================================

        # Half boundary length
        self.boundary_L_half = 2.4
        self.bound_center = np.zeros(2)

        ## Venue
        self.L = self.boundary_L_half
        self.k_ball = 30       # sphere-sphere contact stiffness  N/m 
        # self.c_ball = 5      # sphere-sphere contact damping N/m/s
        self.k_wall = 100      # sphere-wall contact stiffness  N/m
        self.c_wall = 5        # sphere-wall contact damping N/m/s
        self.c_aero = 1.2      # sphere aerodynamic drag coefficient N/m/s

        ## Simulation Steps
        self.simulation_time = 0
        self.dt = 0.1
        self.n_frames = 1  
        self.sensitivity = 1

        ## Rendering
        self.traj_len = 15
        self.plot_initialized = 0
        self.center_view_on_swarm = False
        self.fontsize = 20
        width = 12
        height = 12
        self.figure_handle = plt.figure(figsize = (width,height))

    def __reinit__(self, args):
        self.n_a = args.n_a
        self.render_traj = args.render_traj
        self.traj_len = args.traj_len
        self.video = args.video

        self.is_boundary = args.is_boundary
        if self.is_boundary:
            self.is_periodic = False
        else:
            self.is_periodic = True

        self.dynamics_mode = args.dynamics_mode
        self.agent_strategy = args.agent_strategy
        self.is_con_self_state = args.is_con_self_state

        self.results_file = args.results_file
        with open(self.results_file, 'rb') as f:
            loaded_results = pickle.load(f)

        self.l_cells = loaded_results['l_cell']
        self.grid_center_origins = loaded_results['grid_coords']
        self.binary_images = loaded_results['binary_image']
        self.shape_bound_points_origins = loaded_results['shape_bound_points']
        self.num_train_shape = len(self.l_cells)

        # compute the collision avoidance distance
        self.n_gs = [grid_center.shape[0] for grid_center in self.grid_center_origins]

        self.r_avoid = np.sqrt(4*np.min(self.n_gs)/(self.n_a*np.pi)) * np.min(self.l_cells)
        # self.r_avoid = 0.1
        # self.num_voronoi_mean_g = np.mean(self.n_gs) / 25

        # self.l_cell = args.l_cell
        # self.grid_center_origin = args.grid_center_pos.T
        # self.n_g = self.grid_center_origin.shape[1]
        # self.target_shape = args.shape_image
        # self.shape_bound_points_origin = args.shape_bound_points
        self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        self.num_obs_grid_max = 80
        self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        self.num_occupied_grid_max = 200
        self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)

        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_a, self.n_a))
        self.is_collide_b2w = np.zeros((4, self.n_a), dtype=bool)
        self.d_b2w = np.ones((4, self.n_a))

        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  
        if self.dynamics_mode == 'Cartesian':
            self.is_Cartesian = True
            self.Acc_min = -1    
            assert (self.Acc_min, self.Acc_max) == (-1, 1)
        elif self.dynamics_mode == 'Polar':
            self.is_Cartesian = False
            self.Acc_min = 0    
            self.angle_max = 0.5   
            assert self.Acc_min >= 0
        else:
            print('Wrong in linAcc_p_min')
        
        self.s_text = np.char.mod('%d',np.arange(self.n_a))
        self.color = np.tile(np.array([1, 0.5, 0]), (self.n_a, 1))
        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=40)
            self.video.video.setup(self.figure_handle, args.video_path)

    def init_after_vari_num(self, agent_indices):
        self.n_a = len(agent_indices)

        self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)

        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_a, self.n_a))
        self.is_collide_b2w = np.zeros((4, self.n_a), dtype=bool)
        self.d_b2w = np.ones((4, self.n_a))

        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size() 

        self.s_text = np.char.mod('%d',np.arange(self.n_a))
        self.color = np.tile(np.array([1, 0.5, 0]), (self.n_a, 1))

        # self.r_avoid = np.sqrt(4*self.n_g/(self.n_a*np.pi)) * self.l_cell

        self.heading = np.zeros((self.dim, self.n_a))
        self.p = self.p[:,agent_indices]
        self.dp = self.dp[:,agent_indices]
        self.ddp = self.ddp[:,agent_indices]
        self.p_traj = self.p_traj[:,:,agent_indices]

    def init_reset_num(self, agent_indices):
        self.n_a = len(agent_indices)

        # self.r_avoid = np.sqrt(4*self.n_g/(self.n_a*np.pi)) * self.l_cell

        self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)

        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_a, self.n_a))
        self.is_collide_b2w = np.zeros((4, self.n_a), dtype=bool)
        self.d_b2w = np.ones((4, self.n_a))

        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size() 

        self.s_text = np.char.mod('%d',np.arange(self.n_a))
        self.color = np.tile(np.array([1, 0.5, 0]), (self.n_a, 1))

    def reset(self):
        self.simulation_time = 0

        # change the number of the agent team
        # n_a = np.random.randint(30, 50)
        n_a = 30
        remain_agent_indices = np.arange(n_a)
        self.init_reset_num(remain_agent_indices)

        # the parameters of the current shape
        shape_index = np.random.randint(0, self.num_train_shape)  ################## domain generalization 1
        # shape_index = 0
        self.l_cell = self.l_cells[shape_index]
        self.grid_center_origin = self.grid_center_origins[shape_index].T
        self.target_shape = self.binary_images[shape_index]
        self.shape_bound_points_origin = self.shape_bound_points_origins[shape_index]

        # rectify the shape scale
        # shape_scale = np.random.uniform(1, 1.3)                   ################## domain generalization 2
        shape_scale = 1  # 1.1
        self.l_cell = shape_scale * self.l_cell
        self.grid_center_origin = shape_scale * self.grid_center_origin
        self.shape_bound_points_origin = shape_scale * self.shape_bound_points_origin 

        rand_angle = np.pi * np.random.uniform(-1, 1)               ################## domain generalization 3
        # rand_angle = 0
        rotate_matrix = np.array([[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]])
        self.grid_center_origin = np.dot(rotate_matrix, self.grid_center_origin)
        self.n_g = self.grid_center_origin.shape[1]

        # self.r_avoid = np.sqrt(4*self.n_g/(self.n_a*np.pi)) * self.l_cell

        # 
        # self.d_sen = 0.5 * shape_scale
        self.d_sen = 0.4
        # self.d_sen = 1.6 * self.r_avoid
        self.heading = np.zeros((self.dim, self.n_a))
        self.num_in_voronoi_last = np.zeros(self.n_a, dtype=np.float64)
        self.num_in_voronoi_last_c = np.zeros(self.n_a, dtype=np.float64)

        # randomize target shape's position
        rand_target_offset = np.random.uniform(-1.2, 1.2, (2, 1))   ################## domain generalization 4
        # rand_target_offset = np.zeros((2,1))
        self.grid_center = self.grid_center_origin.copy() + rand_target_offset
        self.shape_bound_points = np.hstack((self.shape_bound_points_origin[:2] + rand_target_offset[0,0], self.shape_bound_points_origin[2:] + rand_target_offset[1,0]))

        # bound position
        self.bound_center = np.array([0, 0])
        # x_min, y_max, x_max, y_min
        self.boundary_pos = np.array([self.bound_center[0] - self.boundary_L_half,
                                      self.bound_center[1] + self.boundary_L_half,
                                      self.bound_center[0] + self.boundary_L_half,
                                      self.bound_center[1] - self.boundary_L_half], dtype=np.float64) 

        # initialize position
        # random_int = np.random.randint(1, 4)
        random_int = 2.4
        if np.random.uniform(-1, 1) > 0:
            self.p = np.random.uniform(-random_int, random_int, (2, self.n_a))   # Initialize self.p
        else:
            self.p = np.random.uniform(-1, 1, (2, self.n_a)) +  np.random.uniform(-2, 2, (2, 1))  # Initialize self.p

        if self.render_traj == True:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_a))
            self.p_traj[-1,:,:] = self.p
         
        # initilize velocity
        if self.dynamics_mode == 'Cartesian':
            # self.dp = np.zeros((self.dim, self.n_a))
            self.dp = np.random.uniform(-0.5, 0.5, (self.dim, self.n_a))
        else:
            self.dp = np.zeros((self.dim, self.n_a))
            # self.dp = np.random.uniform(-1.5, 1.5, (self.dim, self.n_a))

        # if self.obstacles_cannot_move:
        #     self.dp[:, self.n_a:self.n_a] = 0

        # initilize acceleration
        self.ddp = np.zeros((2, self.n_a))         

        if self.dynamics_mode == 'Polar': 
            self.theta = np.pi * np.random.uniform(-1, 1, (1, self.n_a))
            self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0)                                 

        obs = self._get_obs()
        # cent_obs = self._get_cent_obs()

        return obs
        # return obs, cent_obs

    def _get_obs(self):

        self.obs = np.zeros(self.observation_space.shape) 
        self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)
        conditions = np.array([self.is_periodic, self.is_Cartesian, self.is_con_self_state])

        _LIB._get_observation(as_double_c_array(self.p), 
                              as_double_c_array(self.dp), 
                              as_double_c_array(self.heading),
                              as_double_c_array(self.obs),
                              as_double_c_array(self.boundary_pos),
                              as_double_c_array(self.grid_center),
                              as_int32_c_array(self.neighbor_index),
                              as_int32_c_array(self.in_flags),
                              as_int32_c_array(self.sensed_index),
                              as_int32_c_array(self.occupied_index),
                              ctypes.c_double(self.d_sen), 
                              ctypes.c_double(self.r_avoid), 
                              ctypes.c_double(self.l_cell), 
                              ctypes.c_int(self.topo_nei_max),
                              ctypes.c_int(self.num_obs_grid_max),
                              ctypes.c_int(self.num_occupied_grid_max), 
                              ctypes.c_int(self.n_a), 
                              ctypes.c_int(self.n_g),
                              ctypes.c_int(self.obs_dim_agent), 
                              ctypes.c_int(self.dim), 
                              as_bool_c_array(conditions))
        
        # bb = self.obs
        # self.obs = np.zeros(self.observation_space.shape) 
        # self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        # self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        # self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        # self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)

        # for i in range(self.n_a):
        #     '''
        #     The agent's observation, including the state of the neighbors and target grid
        #     '''
        #     relPos_a2a = self.p[:, :self.n_a] - self.p[:,[i]]
        #     if self.is_periodic:
        #         relPos_a2a = self._make_periodic(relPos_a2a, is_rel=True)
        #     relVel_a2a = self.dp[:,:self.n_a] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:, :self.n_a] - self.heading[:, [i]]
        #     relPos_a2a, relVel_a2a, neigh_index = self._get_focused(relPos_a2a, relVel_a2a, self.d_sen, self.topo_nei_max, True)  
        #     nei_num = len(neigh_index)
        #     if nei_num > 0:
        #         self.neighbor_index[i, :nei_num] = neigh_index

        #     if self.is_con_self_state:
        #         obs_agent_pos = np.concatenate((self.p[:, [i]], relPos_a2a), axis=1)
        #         obs_agent_vel = np.concatenate((self.dp[:, [i]], relVel_a2a), axis=1)
        #         obs_agent = np.concatenate((obs_agent_pos, obs_agent_vel), axis=0)
        #     else:
        #         obs_agent = np.concatenate((relPos_a2a, relVel_a2a), axis=0)

        #     # get the target position and velocity
        #     in_flag, target_grid_pos, target_grid_vel, sensed_indices = self._get_trgt_grid_state(i)
        #     self.in_flags[i] = in_flag
        #     target_grid_pos_rel = target_grid_pos - self.p[:,i]
        #     target_grid_vel_rel = target_grid_vel - self.dp[:,i]

        #     # filter the sensed grids
        #     occupied_indices = sensed_indices.copy()
        #     num_sensed_grid_origin = len(sensed_indices)
        #     if num_sensed_grid_origin > 0:
        #         sensed_grid = self.grid_center[:,sensed_indices]
        #         if self.in_flags[i] == 1: # need to remove the occupied grids
        #             # get the nearby agents
        #             agent_pos_rel = self.p - self.p[:,[i]]
        #             agent_pos_rel_norm = np.linalg.norm(agent_pos_rel, axis=0)
        #             nearby_agents = np.where(agent_pos_rel_norm < (self.d_sen + self.r_avoid/2))[0]
        #             # nearby_agents = nearby_agents[nearby_agents != i]
        #             for nearby_i in nearby_agents: 
        #                 grid_neigh_pos_relative = sensed_grid - self.p[:,[nearby_i]]
        #                 grid_neigh_pos_relative_norm = np.linalg.norm(grid_neigh_pos_relative, axis=0)
        #                 mask = np.where(grid_neigh_pos_relative_norm > self.r_avoid / 2)[0]
        #                 sensed_grid = sensed_grid[:, mask]
        #                 sensed_indices = sensed_indices[mask]

        #     occupied_indices = np.setdiff1d(occupied_indices, sensed_indices)
        #     num_occupied_grid = len(occupied_indices)
        #     if num_occupied_grid > self.num_occupied_grid_max:
        #         step = (num_occupied_grid - 1) / (self.num_occupied_grid_max - 1)
        #         final_indices = np.round(np.arange(0, self.num_occupied_grid_max) * step).astype(int)
        #         final_indices = np.array(occupied_indices)[final_indices]
        #         self.occupied_index[i] = final_indices
        #     elif num_occupied_grid > 0 and num_occupied_grid <= self.num_occupied_grid_max:
        #         self.occupied_index[i,:num_occupied_grid] = occupied_indices
        #         # pass

        #     # get the final sensed grid
        #     num_sensed_grid = len(sensed_indices)
        #     if num_sensed_grid > self.num_obs_grid_max:
        #         step = (num_sensed_grid - 1) / (self.num_obs_grid_max - 1)
        #         final_indices = np.round(np.arange(0, self.num_obs_grid_max) * step).astype(int)
        #         final_indices = np.array(sensed_indices)[final_indices]
        #         sensed_grid_pos = self.grid_center[:, final_indices]
        #         self.sensed_index[i] = final_indices
        #     elif num_sensed_grid > 0 and num_sensed_grid <= self.num_obs_grid_max:
        #         sensed_grid_pos = self.grid_center[:, sensed_indices]
        #         self.sensed_index[i,:num_sensed_grid] = sensed_indices
        #     else:
        #         sensed_grid_pos = None
        
        #     sensed_grid_pos_rel = np.zeros((self.dim, self.num_obs_grid_max))
        #     if sensed_grid_pos is None:
        #         pass
        #     else:
        #         num_obs_grid = len(sensed_grid_pos[0])
        #         sensed_grid_pos_rel[:,:num_obs_grid] = sensed_grid_pos - self.p[:,[i]]

        #     if self.dynamics_mode == 'Cartesian':
        #         self.obs[:self.obs_dim_agent - (2+self.num_obs_grid_max)*self.dim, i] = obs_agent.T.reshape(-1)       
        #         self.obs[self.obs_dim_agent - (2+self.num_obs_grid_max)*self.dim:self.obs_dim_agent - (1+self.num_obs_grid_max)*self.dim, i] = target_grid_pos_rel
        #         self.obs[self.obs_dim_agent - (1+self.num_obs_grid_max)*self.dim:self.obs_dim_agent - self.num_obs_grid_max*self.dim, i] = target_grid_vel_rel
        #         self.obs[self.obs_dim_agent - self.num_obs_grid_max*self.dim:, i] = sensed_grid_pos_rel.T.reshape(-1)
        #     elif self.dynamics_mode == 'Polar':
        #         self.obs[:self.obs_dim_agent - 3*self.dim, i] = obs_agent.T.reshape(-1)
        #         self.obs[self.obs_dim_agent - 3*self.dim:self.obs_dim_agent - 2 * self.dim, i] = target_grid_pos_rel
        #         self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = target_grid_vel_rel
        #         self.obs[self.obs_dim_agent - self.dim:, i] = self.heading[:,i]

        # ss = np.sum(bb - self.obs)
        # if ss != 0:
        #     print(ss)
                
        return self.obs

    # def _get_obs(self):

    #     self.obs = np.zeros(self.observation_space.shape) 
    #     self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
    #     self.in_flags = np.zeros(self.n_a, dtype=np.int32)
    #     self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
    #     self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)
    #     conditions = np.array([self.is_periodic, self.is_Cartesian, self.is_con_self_state])

    #     _LIB._get_observation(as_double_c_array(self.p), 
    #                           as_double_c_array(self.dp), 
    #                           as_double_c_array(self.heading),
    #                           as_double_c_array(self.obs),
    #                           as_double_c_array(self.boundary_pos),
    #                           as_double_c_array(self.grid_center),
    #                           as_int32_c_array(self.neighbor_index),
    #                           as_int32_c_array(self.in_flags),
    #                           as_int32_c_array(self.sensed_index),
    #                           as_int32_c_array(self.occupied_index),
    #                           ctypes.c_double(self.d_sen), 
    #                           ctypes.c_double(self.r_avoid), 
    #                           ctypes.c_double(self.l_cell), 
    #                           ctypes.c_int(self.topo_nei_max),
    #                           ctypes.c_int(self.num_obs_grid_max),
    #                           ctypes.c_int(self.num_occupied_grid_max), 
    #                           ctypes.c_int(self.n_a), 
    #                           ctypes.c_int(self.n_g),
    #                           ctypes.c_int(self.obs_dim_agent), 
    #                           ctypes.c_int(self.dim), 
    #                           as_bool_c_array(conditions))
        
        # bb = self.obs
        # self.obs = np.zeros(self.observation_space.shape) 
        # self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        # self.in_flags = np.zeros(self.n_a, dtype=np.int32)
        # self.sensed_index = -1 * np.ones((self.n_a, self.num_obs_grid_max), dtype=np.int32)
        # self.occupied_index = -1 * np.ones((self.n_a, self.num_occupied_grid_max), dtype=np.int32)

        # for i in range(self.n_a):
        #     '''
        #     The agent's observation, including the state of the neighbors and target grid
        #     '''
        #     relPos_a2a = self.p[:, :self.n_a] - self.p[:,[i]]
        #     if self.is_periodic:
        #         relPos_a2a = self._make_periodic(relPos_a2a, is_rel=True)
        #     relVel_a2a = self.dp[:,:self.n_a] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:, :self.n_a] - self.heading[:, [i]]
        #     relPos_a2a, relVel_a2a, neigh_index = self._get_focused(relPos_a2a, relVel_a2a, self.d_sen, self.topo_nei_max, True)  
        #     nei_num = len(neigh_index)
        #     if nei_num > 0:
        #         self.neighbor_index[i, :nei_num] = neigh_index

        #     if self.is_con_self_state:
        #         obs_agent_pos = np.concatenate((self.p[:, [i]], relPos_a2a), axis=1)
        #         obs_agent_vel = np.concatenate((self.dp[:, [i]], relVel_a2a), axis=1)
        #         obs_agent = np.concatenate((obs_agent_pos, obs_agent_vel), axis=0)
        #     else:
        #         obs_agent = np.concatenate((relPos_a2a, relVel_a2a), axis=0)

        #     # get the target position and velocity
        #     in_flag, target_grid_pos, target_grid_vel, sensed_indices = self._get_trgt_grid_state(i)
        #     self.in_flags[i] = in_flag
        #     target_grid_pos_rel = target_grid_pos - self.p[:,i]
        #     target_grid_vel_rel = target_grid_vel - self.dp[:,i]

        #     # filter the sensed grids
        #     occupied_indices = sensed_indices.copy()
        #     num_sensed_grid_origin = len(sensed_indices)
        #     # num_grid_in_voronoi = np.zeros(len(neigh_index) + 1)
        #     if num_sensed_grid_origin > 0:
        #         if self.in_flags[i] == 1: # need to remove the occupied grids
        #             neigh_index = np.insert(neigh_index, 0, i)
        #             index_grid_in_voronoi = []
        #             # index_grid_in_voronoi = [[] for _ in neigh_index]
        #             for cell_index in sensed_indices:
        #                 rel_pos_cell_nei = self.p[:,neigh_index] - self.grid_center[:,[cell_index]]
        #                 rel_pos_cell_nei_norm = np.linalg.norm(rel_pos_cell_nei, axis=0)
        #                 min_index = np.argmin(rel_pos_cell_nei_norm)
        #                 # num_grid_in_voronoi[min_index] += 1
        #                 if min_index == 0:
        #                     index_grid_in_voronoi.append(cell_index)

        #             sensed_indices = np.array(index_grid_in_voronoi)
        #             # sensed_indices = np.array(index_grid_in_voronoi[0])
        #             # num_grid_in_voronoi = num_grid_in_voronoi / num_sensed_grid_origin

        #     # num_grid_in_voronoi = np.pad(num_grid_in_voronoi, (0, self.topo_nei_max + 1 - len(num_grid_in_voronoi)), 'constant')

        #     occupied_indices = np.setdiff1d(occupied_indices, sensed_indices)
        #     num_occupied_grid = len(occupied_indices)
        #     if num_occupied_grid > self.num_occupied_grid_max:
        #         step = (num_occupied_grid - 1) / (self.num_occupied_grid_max - 1)
        #         final_indices = np.round(np.arange(0, self.num_occupied_grid_max) * step).astype(int)
        #         final_indices = np.array(occupied_indices)[final_indices]
        #         self.occupied_index[i] = final_indices
        #     elif num_occupied_grid > 0 and num_occupied_grid <= self.num_occupied_grid_max:
        #         self.occupied_index[i,:num_occupied_grid] = occupied_indices
        #         # pass

        #     # get the final sensed grid
        #     num_sensed_grid = len(sensed_indices)
        #     if num_sensed_grid > self.num_obs_grid_max:
        #         step = (num_sensed_grid - 1) / (self.num_obs_grid_max - 1)
        #         final_indices = np.round(np.arange(0, self.num_obs_grid_max) * step).astype(int)
        #         final_indices = np.array(sensed_indices)[final_indices]
        #         sensed_grid_pos = self.grid_center[:, final_indices]
        #         self.sensed_index[i] = final_indices
        #     elif num_sensed_grid > 0 and num_sensed_grid <= self.num_obs_grid_max:
        #         sensed_grid_pos = self.grid_center[:, sensed_indices]
        #         self.sensed_index[i,:num_sensed_grid] = sensed_indices
        #     else:
        #         sensed_grid_pos = None
        
        #     sensed_grid_pos_rel = np.zeros((self.dim, self.num_obs_grid_max))
        #     if sensed_grid_pos is None:
        #         pass
        #     else:
        #         num_obs_grid = len(sensed_grid_pos[0])
        #         sensed_grid_pos_rel[:,:num_obs_grid] = sensed_grid_pos - self.p[:,[i]]

        #     index_1 = (2 + self.num_obs_grid_max) * self.dim
        #     index_2 = self.num_obs_grid_max * self.dim
        #     # index_1 = (2 + self.num_obs_grid_max) * self.dim + self.topo_nei_max + 1
        #     # index_2 = self.num_obs_grid_max * self.dim + self.topo_nei_max + 1
        #     # index_3 = self.topo_nei_max + 1
        #     if self.dynamics_mode == 'Cartesian':
        #         self.obs[:self.obs_dim_agent - index_1, i] = obs_agent.T.reshape(-1)       
        #         self.obs[self.obs_dim_agent - index_1:self.obs_dim_agent - index_2, i] = np.concatenate((target_grid_pos_rel, target_grid_vel_rel))
        #         self.obs[self.obs_dim_agent - index_2:, i] = sensed_grid_pos_rel.T.reshape(-1)
        #         # self.obs[self.obs_dim_agent - index_3:, i] = num_grid_in_voronoi
        #     # if self.dynamics_mode == 'Cartesian':
        #     #     self.obs[:self.obs_dim_agent - index_1, i] = obs_agent.T.reshape(-1)       
        #     #     self.obs[self.obs_dim_agent - index_1:self.obs_dim_agent - index_2, i] = target_grid_pos_rel
        #     #     self.obs[self.obs_dim_agent - index_2:self.obs_dim_agent - index_3, i] = target_grid_vel_rel
        #     #     self.obs[self.obs_dim_agent - index_3:, i] = sensed_grid_pos_rel.T.reshape(-1)
        #     # elif self.dynamics_mode == 'Polar':
        #     #     self.obs[:self.obs_dim_agent - 3*self.dim, i] = obs_agent.T.reshape(-1)
        #     #     self.obs[self.obs_dim_agent - 3*self.dim:self.obs_dim_agent - 2 * self.dim, i] = target_grid_pos_rel
        #     #     self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = target_grid_vel_rel
        #     #     self.obs[self.obs_dim_agent - self.dim:, i] = self.heading[:,i]

        # ss = np.sum(bb - self.obs)
        # if ss != 0:
        #     print(ss)
                
        # return self.obs

    # def _get_obs(self):

    #     self.obs = np.zeros(self.observation_space.shape) 
    #     self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
    #     self.in_flags = np.zeros(self.n_a, dtype=np.int32)
    #     conditions = np.array([self.is_periodic, self.is_Cartesian, self.is_con_self_state])

    #     _LIB._get_observation(as_double_c_array(self.p), 
    #                           as_double_c_array(self.dp), 
    #                           as_double_c_array(self.heading),
    #                           as_double_c_array(self.obs),
    #                           as_double_c_array(self.boundary_pos),
    #                           as_double_c_array(self.grid_center),
    #                           as_int32_c_array(self.neighbor_index),
    #                           as_int32_c_array(self.in_flags),
    #                           ctypes.c_double(self.d_sen), 
    #                           ctypes.c_double(self.r_avoid), 
    #                           ctypes.c_double(self.l_cell), 
    #                           ctypes.c_int(self.topo_nei_max), 
    #                           ctypes.c_int(self.n_a), 
    #                           ctypes.c_int(self.n_g),
    #                           ctypes.c_int(self.obs_dim_agent), 
    #                           ctypes.c_int(self.dim), 
    #                           as_bool_c_array(conditions))
        
    #     # bb = self.obs
    #     # self.obs = np.zeros(self.observation_space.shape) 
    #     # self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
    #     # self.in_flags = np.zeros(self.n_a, dtype=np.int32)

    #     # for i in range(self.n_a):
    #     #     '''
    #     #     The agent's observation, including the state of the neighbors and target grid
    #     #     '''
    #     #     relPos_a2a = self.p[:, :self.n_a] - self.p[:,[i]]
    #     #     if self.is_periodic:
    #     #         relPos_a2a = self._make_periodic(relPos_a2a, is_rel=True)
    #     #     relVel_a2a = self.dp[:,:self.n_a] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:, :self.n_a] - self.heading[:, [i]]
    #     #     relPos_a2a, relVel_a2a, neigh_index = self._get_focused(relPos_a2a, relVel_a2a, self.d_sen, self.topo_nei_max, True)  
    #     #     nei_num = len(neigh_index)
    #     #     if nei_num > 0:
    #     #         self.neighbor_index[i, :nei_num] = neigh_index

    #     #     if self.is_con_self_state:
    #     #         obs_agent_pos = np.concatenate((self.p[:, [i]], relPos_a2a), axis=1)
    #     #         obs_agent_vel = np.concatenate((self.dp[:, [i]], relVel_a2a), axis=1)
    #     #         obs_agent = np.concatenate((obs_agent_pos, obs_agent_vel), axis=0)
    #     #     else:
    #     #         obs_agent = np.concatenate((relPos_a2a, relVel_a2a), axis=0)

    #     #     # get the target position and velocity
    #     #     in_flag, target_grid_pos, target_grid_vel, sensed_indices = self._get_trgt_grid_state(i)
    #     #     self.in_flags[i] = in_flag
    #     #     target_grid_pos_rel = target_grid_pos - self.p[:,i]
    #     #     target_grid_vel_rel = target_grid_vel - self.dp[:,i]

    #     #     if self.dynamics_mode == 'Cartesian':
    #     #         self.obs[:self.obs_dim_agent - 2*self.dim, i] = obs_agent.T.reshape(-1)       
    #     #         self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = target_grid_pos_rel
    #     #         self.obs[self.obs_dim_agent - self.dim:, i] = target_grid_vel_rel
    #     #     elif self.dynamics_mode == 'Polar':
    #     #         self.obs[:self.obs_dim_agent - 3*self.dim, i] = obs_agent.T.reshape(-1)
    #     #         self.obs[self.obs_dim_agent - 3*self.dim:self.obs_dim_agent - 2 * self.dim, i] = target_grid_pos_rel
    #     #         self.obs[self.obs_dim_agent - 2*self.dim:self.obs_dim_agent - self.dim, i] = target_grid_vel_rel
    #     #         self.obs[self.obs_dim_agent - self.dim:, i] = self.heading[:,i]

    #     # ss = np.sum(bb - self.obs)
    #     # if ss != 0:
    #     #     print(ss)
                
    #     return self.obs

    def _get_cent_obs(self):
        local_obs = self.obs
        state = np.concatenate((self.p, self.dp), axis=0)
        flatten_state = state.T.reshape(-1)
        global_obs = np.tile(flatten_state, (self.n_a, 1)).T
        cent_obs = np.concatenate((local_obs, global_obs), axis=0)
        
        return cent_obs
      
    def _get_reward(self, a):

        reward_a = np.zeros((1, self.n_a))
        coefficients = np.array([1] + [2, 1] + [15, 3] + [1, 1, 1] + [5] + [12], dtype=np.float64) # reward #2 dense 5 10 1 15
        conditions = np.array([self.is_periodic, self.is_Cartesian, self.penalize_entering, self.penalize_interaction, self.penalize_exploration,
                               self.penalize_control_effort, self.penalize_collide_obstacles, self.penalize_collide_walls], dtype=bool)

        _LIB._get_reward(as_double_c_array(self.p), 
                         as_double_c_array(self.dp), 
                         as_double_c_array(self.heading),
                         as_double_c_array(a.astype(np.float64)), 
                         as_double_c_array(reward_a), 
                         as_double_c_array(self.boundary_pos),
                         as_double_c_array(self.grid_center),
                         as_double_c_array(self.num_in_voronoi_last_c),
                         as_int32_c_array(self.neighbor_index),
                         as_int32_c_array(self.in_flags),
                         as_int32_c_array(self.sensed_index),
                         as_int32_c_array(self.occupied_index),
                         ctypes.c_double(self.d_sen), 
                         ctypes.c_double(self.r_avoid),
                         ctypes.c_int(self.topo_nei_max), 
                         ctypes.c_int(self.num_obs_grid_max),
                         ctypes.c_int(self.num_occupied_grid_max), 
                         ctypes.c_int(self.n_a), 
                         ctypes.c_int(self.n_g),
                         ctypes.c_int(self.dim), 
                         as_bool_c_array(conditions),
                         as_bool_c_array(self.is_collide_b2b),
                         as_bool_c_array(self.is_collide_b2w),
                         as_double_c_array(coefficients))

        # bb = reward_a
        # reward_a = np.zeros((1, self.n_a))
        # is_has_collisions = np.zeros(self.n_a, dtype=bool)
        # is_uniforms = np.zeros(self.n_a, dtype=bool)

        # # shape-entering reward
        # # if self.penalize_entering:
        # #     # flags = coefficients[0] * (self.in_flags - 1)
        # #     # reward_a += np.expand_dims(flags, axis=0)
        # #     contains_zero = np.any(self.in_flags == 0)
        # #     if contains_zero == False:
        # #         is_all_in_shape = True
            
        # # interaction reward
        # if self.penalize_interaction:
        #     for agent_i in range(self.n_a):
        #         # get the neighbors
        #         list_nei = self.neighbor_index[agent_i]
        #         list_nei = list_nei[list_nei != -1]

        #         # is_has_collision = False

        #         if len(list_nei) > 0:
        #             for neigh_i in list_nei:
        #                 # repulsion
        #                 pos_relative = self.p[:,neigh_i] - self.p[:,agent_i]
        #                 if self.is_periodic:
        #                     pos_relative = self._make_periodic(pos_relative, is_rel=True)

        #                 dist_b2b = np.linalg.norm(pos_relative)
        #                 if self.r_avoid > dist_b2b:
        #                     # reward_a[0, agent_i] -= coefficients[1]
        #                     is_has_collisions[agent_i] = True
        #                     break
        #                 # reward_a[0, agent_i] -= np.max([coefficients[1] * (self.r_avoid - dist_b2b), 0.0])
                    
        #             # alignment
        #             avg_neigh_vel = np.mean(self.dp[:,list_nei], axis=1)
        #             # reward_a[0,agent_i] -= coefficients[2] * np.linalg.norm(avg_neigh_vel - self.dp[:,agent_i])
        #             if np.linalg.norm(avg_neigh_vel - self.dp[:,agent_i]) > 0.2:
        #                 is_has_collisions[agent_i] = True
                
        #         # if self.in_flags[agent_i] == 1 and is_has_collision == False:
        #         #     reward_a[0, agent_i] += 1      

        # # exploration reward
        # if self.penalize_exploration:
        #     for agent_i in range(self.n_a):
        #         if self.in_flags[agent_i] == 1:
        #             ################################# reward form 1 #################################
        #             # # get the neighbors
        #             # list_nei = self.neighbor_index[agent_i]
        #             # list_nei = list_nei[list_nei != -1]

        #             # # get the sensed grids
        #             # list_grid = self.sensed_index[agent_i]
        #             # list_grid = list_grid[list_grid != -1]

        #             # if len(list_nei) > 0:
        #             #     # num_list_grid_norm = len(list_grid) / self.num_obs_grid_max
        #             #     # num_list_grid_neigh_avg = 0
        #             #     # for neigh_i in list_nei:
        #             #     #     # get the sensed grids
        #             #     #     list_grid_neigh = self.sensed_index[neigh_i]
        #             #     #     list_grid_neigh = list_grid_neigh[list_grid_neigh != -1]
        #             #     #     num_list_grid_neigh_avg += len(list_grid_neigh) / self.num_obs_grid_max

        #             #     # num_list_grid_neigh_avg = num_list_grid_neigh_avg / len(list_nei)
        #             #     num_list_grid_norm = self.num_neigh_grid[agent_i]
        #             #     num_list_grid_neigh_avg = np.sum(self.num_neigh_grid[list_nei]) / len(list_nei)
        #             #     reward_a[0,agent_i] -= coefficients[3] * np.abs(num_list_grid_norm - num_list_grid_neigh_avg)

        #             ################################# reward form 2 #################################
        #             # # get the sensed grids
        #             # list_grid = self.sensed_index[agent_i]
        #             # list_grid = list_grid[list_grid != -1]

        #             # # get the occupied grids
        #             # list_occupied_grid = self.occupied_index[agent_i]
        #             # list_occupied_grid = list_occupied_grid[list_occupied_grid != -1]

        #             # list_grid = np.concatenate((list_grid, list_occupied_grid))

        #             # sensed_grid_pos = self.grid_center[:, list_grid]
        #             # occupied_grid_pos = self.grid_center[:, list_occupied_grid]
        #             # num_list_grid = len(list_grid)
        #             # num_list_occupied_grid = len(list_occupied_grid)
        #             # if num_list_grid > 0 and num_list_occupied_grid > 0:
        #             #     mean_sensed_grid_pos = np.mean(sensed_grid_pos, axis=1)
        #             #     mean_occupied_grid_pos = np.mean(occupied_grid_pos, axis=1)
        #             #     # reward_a[0,agent_i] -= coefficients[3] * np.linalg.norm(mean_sensed_grid_pos - mean_occupied_grid_pos)
        #             #     print(np.linalg.norm(mean_sensed_grid_pos - mean_occupied_grid_pos))
        #             #     if np.linalg.norm(mean_sensed_grid_pos - mean_occupied_grid_pos) < 0.03:
        #             #         is_uniforms[agent_i] = True

        #             ################################# reward form 3 #################################
        #             # get the sensed grids
        #             list_grid = self.sensed_index[agent_i]
        #             list_grid = list_grid[list_grid != -1]

        #             if len(list_grid) > 0:
        #                 # print(len(list_grid))
        #                 sensed_grid = self.grid_center[:,list_grid]
        #                 if sensed_grid.size > 0:
        #                     sensed_grid_pos_rel = sensed_grid - self.p[:,[agent_i]]
        #                     numerator = np.sum(sensed_grid_pos_rel, axis=1) # shape: [dim]
        #                     denominator = list_grid.size  # scalar  
        #                     # sensed_grid_pos_rel_norm = np.linalg.norm(sensed_grid_pos_rel, axis=0)
        #                     # psi_values = self._rho_cos_dec(sensed_grid_pos_rel_norm, 0, self.d_sen)
        #                     # weighted_diff = psi_values * sensed_grid_pos_rel
        #                     # numerator = np.sum(weighted_diff, axis=1) # shape: [dim]
        #                     # denominator = np.sum(psi_values)  # scalar
        #                     v_exp_i = numerator / denominator  # shape: [dim] expected exploration velocity
        #                     # print(denominator, np.linalg.norm(v_exp_i))
        #                     if np.linalg.norm(v_exp_i) < 0.06:
        #                         is_uniforms[agent_i] = True

        #             ################################# reward form 4 #################################
        #             # # get the neighbors
        #             # list_nei = self.neighbor_index[agent_i]
        #             # list_nei = list_nei[list_nei != -1]

        #             # list_nei = np.append(list_nei, agent_i)
        #             # num_grid_in_voronoi = np.zeros_like(list_nei)
        #             # index_grid_in_voronoi = [[] for _ in range(len(list_nei))]
        #             # # get the sensed grids
        #             # list_grid = self.sensed_index[agent_i]
        #             # list_grid = list_grid[list_grid != -1]
        #             # # # get the occupied grids
        #             # # list_occupied_grid = self.occupied_index[agent_i]
        #             # # list_occupied_grid = list_occupied_grid[list_occupied_grid != -1]
        #             # # list_grid = np.concatenate((list_grid, list_occupied_grid))

        #             # for cell_index in list_grid:
        #             #     rel_pos_cell_nei = self.p[:,list_nei] - self.grid_center[:,[cell_index]]
        #             #     rel_pos_cell_nei_norm = np.linalg.norm(rel_pos_cell_nei, axis=0)
        #             #     min_index = np.argmin(rel_pos_cell_nei_norm)
        #             #     num_grid_in_voronoi[min_index] += 1
        #             #     index_grid_in_voronoi[min_index].append(cell_index)

        #             # # num_voronoi_nei_mean = np.mean(num_grid_in_voronoi[:-1])
        #             # # delta_num_percent = np.abs(num_grid_in_voronoi[-1] - num_voronoi_nei_mean) / len(list_grid)
        #             # # if delta_num_percent < 1:
        #             # #     # print(delta_num_percent)
        #             # #     is_uniforms[agent_i] = True

        #             # # if num_grid_in_voronoi[-1] > self.num_in_voronoi_last[agent_i]:
        #             # #     is_uniforms[agent_i] = True
        #             # # self.num_in_voronoi_last[agent_i] = num_grid_in_voronoi[-1]

        #             # if num_grid_in_voronoi[-1] > 0:
        #             #     sensed_grid_agent_i = self.grid_center[:, index_grid_in_voronoi[-1]] - self.p[:,[agent_i]]
        #             #     numerator = np.sum(sensed_grid_agent_i, axis=1) # shape: [dim]
        #             #     denominator = num_grid_in_voronoi[-1]  # scalar
        #             #     v_exp_i = numerator / denominator  # shape: [dim] expected exploration velocity
        #             #     # print(denominator, np.linalg.norm(v_exp_i))
        #             #     if np.linalg.norm(v_exp_i) < 0.06:
        #             #         is_uniforms[agent_i] = True

        #             # # num_voronoi_nei_mean = len(list_grid) / len(list_nei)
        #             # # delta_num_percent = np.abs(num_grid_in_voronoi[-1] - num_voronoi_nei_mean)
        #             # # # delta_num_percent = np.abs(num_grid_in_voronoi[-1] - self.num_voronoi_mean_g)
        #             # # print(delta_num_percent)
        #             # # if delta_num_percent < 2:
        #             # #     # print(delta_num_percent)
        #             # #     is_uniforms[agent_i] = True

        #         if self.in_flags[agent_i] == 1 and is_has_collisions[agent_i] == False and is_uniforms[agent_i] == True:
        #             reward_a[0, agent_i] += 1  

        # # control effort
        # if self.penalize_control_effort:
        #     if self.dynamics_mode == 'Cartesian':
        #         reward_a -= coefficients[6] * np.linalg.norm(a, axis=0, keepdims=True)
        #     elif self.dynamics_mode == 'Polar':
        #         reward_a -= coefficients[7]*np.abs(a[[0],:]) + coefficients[8]*np.abs(a[[1],:])

        # if self.penalize_collide_obstacles:
        #     reward_a -= coefficients[9] * self.is_collide_b2b[self.n_a:self.n_a, :self.n_a].sum(axis=0, keepdims=True)          
        
        # if self.penalize_collide_walls:
        #     reward_a -= coefficients[10] * self.is_collide_b2w[:, :self.n_a].sum(axis=0, keepdims=True)         

        # if self.reward_sharing_mode == 'sharing_mean':
        #     reward_a[:] = np.mean(reward_a) 
        # elif self.reward_sharing_mode == 'sharing_max':
        #     reward_a[:] = np.max(reward_a) 
        # elif self.reward_sharing_mode == 'individual':
        #     pass
        # else:
        #     print('reward mode error !!')

        # ss = np.sum(bb - reward_a)
        # if ss != 0:
        #     print(ss)

        # # print(reward_a)
        # if np.any(np.isnan(reward_a)):
        #     print(111111111111111111111111)
        return reward_a

    def _get_dist_b2b(self):
        all_pos = np.tile(self.p, (self.n_a, 1))   
        my_pos = self.p.T.reshape(2 * self.n_a, 1) 
        my_pos = np.tile(my_pos, (1, self.n_a))   
        relative_p_2n_n =  all_pos - my_pos
        if self.is_periodic == True:
            relative_p_2n_n = self._make_periodic(relative_p_2n_n, is_rel=True)
        d_b2b_center = np.sqrt(relative_p_2n_n[::2,:]**2 + relative_p_2n_n[1::2,:]**2)  
        d_b2b_edge = d_b2b_center - self.sizes
        isCollision = (d_b2b_edge < 0)
        d_b2b_edge = np.abs(d_b2b_edge)
        
        self.d_b2b_center = d_b2b_center
        self.d_b2b_edge = d_b2b_edge
        self.is_collide_b2b = isCollision
        return self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b
    
    def _get_dist_b2w(self):
        _LIB._get_dist_b2w(as_double_c_array(self.p), 
                           as_double_c_array(self.size), 
                           as_double_c_array(self.d_b2w), 
                           as_bool_c_array(self.is_collide_b2w),
                           ctypes.c_int(self.dim), 
                           ctypes.c_int(self.n_a), 
                           as_double_c_array(self.boundary_pos))

        # p = self.p
        # r = self.size
        # d_b2w = np.zeros((4, self.n_a))
        # # isCollision = np.zeros((4,self.n_a))
        # for i in range(self.n_a):
        #     d_b2w[:,i] = np.array([ p[0,i] - r[i] - self.boundary_pos[0], 
        #                             self.boundary_pos[1] - (p[1,i] + r[i]),
        #                             self.boundary_pos[2] - (p[0,i] + r[i]),
        #                             p[1,i] - r[i] - self.boundary_pos[3]])  
        # self.is_collide_b2w = d_b2w < 0
        # self.d_b2w = np.abs(d_b2w) 

    def _get_done(self):
        all_done = np.zeros( (1, self.n_a) ).astype(bool)
        return all_done

    def _get_info(self):
        return np.array( [None, None, None] ).reshape(3,1)

    def step(self, a): 
        self.simulation_time += self.dt 
        for _ in range(self.n_frames): 
            if self.dynamics_mode == 'Polar':  
                a[0, :self.n_a] *= self.angle_max
                a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2 

            ########################################################################################################
            self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b = self._get_dist_b2b() 
            # sf_b2b_all = np.zeros((2*self.n_a, self.n_a)) 
            sf_b2b = np.zeros((2, self.n_a))

            _LIB._sf_b2b_all(as_double_c_array(self.p), 
                             as_double_c_array(sf_b2b), 
                             as_double_c_array(self.d_b2b_edge), 
                             as_bool_c_array(self.is_collide_b2b),
                             as_double_c_array(self.boundary_pos),
                             as_double_c_array(self.d_b2b_center),
                             ctypes.c_int(self.n_a), 
                             ctypes.c_int(self.dim), 
                             ctypes.c_double(self.k_ball),
                             ctypes.c_bool(self.is_periodic))
            # for i in range(self.n_a):
            #     for j in range(i):
            #         delta = self.p[:,j]-self.p[:,i]
            #         if self.is_periodic:
            #             delta = self._make_periodic(delta, is_rel=True)
            #         dir = delta / self.d_b2b_center[i,j]
            #         sf_b2b_all[2*i:2*(i+1),j] = self.is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self.k_ball * (-dir)
            #         sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  
                   
            # sf_b2b_1 = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self.n_a,2).T 
            if self.is_boundary:
                self._get_dist_b2w()
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall 
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self.dp, self.dp), axis=0))  *  self.c_wall

            if self.agent_strategy == 'input':
                pass               
            elif self.agent_strategy == 'random':
                a = np.random.uniform(-1, 1, (self.act_dim_agent, self.n_a)) 
                if self.dynamics_mode == 'Polar': 
                    a[0, :self.n_a] *= self.angle_max
                    a[1, :self.n_a] = (self.Acc_max - self.Acc_min)/2 * a[1,:self.n_a] + (self.Acc_max + self.Acc_min)/2
            elif self.agent_strategy == 'rule':
                a = np.zeros((self.dim, self.n_a))
                k_1, k_2, k_3 = 1, 8, 18
                for i in range(self.n_a):
                    # entering velocity
                    in_flag, target_grid_pos, target_grid_vel, sensed_indices = self._get_trgt_grid_state(i)
                    target_grid_pos_rel = target_grid_pos - self.p[:,i]
                    target_grid_vel_rel = target_grid_vel - self.dp[:,i]
                    if in_flag == 1:
                        v_ent_i = np.zeros(self.dim)
                    else:
                        v_ent_i = k_1 * (target_grid_pos_rel / (np.linalg.norm(target_grid_pos_rel) + 1e-8)) + target_grid_vel_rel

                    # exploration velocity
                    num_sensed_grid_origin = len(sensed_indices)
                    if num_sensed_grid_origin > 0:
                        sensed_grid = self.grid_center[:,sensed_indices]
                        if in_flag == 1: # need to remove the occupied grids
                            # get the nearby agents
                            agent_pos_rel = self.p - self.p[:,[i]]
                            agent_pos_rel_norm = np.linalg.norm(agent_pos_rel, axis=0)
                            nearby_agents = np.where(agent_pos_rel_norm < (self.d_sen + self.r_avoid/2))[0]
                            # nearby_agents = nearby_agents[nearby_agents != i]
                            for nearby_i in nearby_agents: 
                                grid_neigh_pos_relative = sensed_grid - self.p[:,[nearby_i]]
                                grid_neigh_pos_relative_norm = np.linalg.norm(grid_neigh_pos_relative, axis=0)
                                mask = np.where(grid_neigh_pos_relative_norm > self.r_avoid / 2)[0]
                                sensed_grid = sensed_grid[:, mask]
                                sensed_indices = sensed_indices[mask]

                    # get the final sensed grid
                    num_sensed_grid = len(sensed_indices)
                    if num_sensed_grid > self.num_obs_grid_max:
                        step = (num_sensed_grid - 1) / (self.num_obs_grid_max - 1)
                        final_indices = np.round(np.arange(0, self.num_obs_grid_max) * step).astype(int)
                        final_indices = np.array(sensed_indices)[final_indices]
                        sensed_grid_pos = self.grid_center[:, final_indices]
                        sensed_index = final_indices
                    elif num_sensed_grid > 0 and num_sensed_grid <= self.num_obs_grid_max:
                        sensed_grid_pos = self.grid_center[:, sensed_indices]
                        sensed_index = sensed_indices
                    else:
                        sensed_grid_pos = None

                    v_exp_i = np.zeros(self.dim)
                    if sensed_grid_pos is not None:
                        sensed_grid_pos_rel = sensed_grid_pos - self.p[:,[i]]
                        sensed_grid_pos_rel_norm = np.linalg.norm(sensed_grid_pos_rel, axis=0)
                        psi_values = self._rho_cos_dec(sensed_grid_pos_rel_norm, 0, self.d_sen)
                        weighted_diff = psi_values * sensed_grid_pos_rel
                        numerator = np.sum(weighted_diff, axis=1) # shape: [dim]
                        denominator = np.sum(psi_values)  # scalar
                        if denominator == 0:
                            denominator = 1e-8  # To avoid division by zero
                        v_exp_i += k_2 * numerator / denominator  # shape: [dim] expected exploration velocity

                    # interaction velocity
                    agent_pos_rel = self.p - self.p[:,[i]]
                    agent_vel_rel = self.dp - self.dp[:,[i]]
                    agent_pos_rel_norm = np.linalg.norm(agent_pos_rel, axis=0)
                    nearby_agents = np.where(agent_pos_rel_norm < self.d_sen)[0]
                    nearby_agents = nearby_agents[nearby_agents != i]
                    v_int_i = np.zeros(self.dim)
                    if len(nearby_agents) > 0:
                        for nearby_i in nearby_agents:
                            if agent_pos_rel_norm[nearby_i] < self.r_avoid:
                                v_int_i += -k_3 * (self.r_avoid / agent_pos_rel_norm[nearby_i] - 1) * agent_pos_rel[:,nearby_i]

                            v_int_i += 5*agent_vel_rel[:,nearby_i] / len(nearby_agents)

                    a[:,i] = v_ent_i + v_exp_i + v_int_i
                    a[:,i] = np.clip(a[:,i], -1, 1)
            else:
                print('Wrong in Step function')

            if self.dynamics_mode == 'Cartesian':
                u = a   
            elif self.dynamics_mode == 'Polar':      
                self.theta += a[[0],:]
                self.theta = self._normalize_angle(self.theta)
                self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0) 
                u = a[[1], :] * self.heading 
            else:
                print('Wrong in updating dynamics')
            
            #########################################################################################
            if self.is_boundary:
                F = self.sensitivity * u + sf_b2b + sf_b2w + df_b2w
            else: 
                F = self.sensitivity * u + sf_b2b

            # acceleration
            self.ddp = F/self.m

            # velocity
            self.dp += self.ddp * self.dt
            self.dp = np.clip(self.dp, -self.Vel_max, self.Vel_max)

            # if self.obstacles_cannot_move:
            #     self.dp[:, self.n_a:self.n_a] = 0
        
            # position
            self.p += self.dp * self.dt
            # if self.obstacles_is_constant:
            #     self.p[:, self.n_a:self.n_a] = self.p_o
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            if self.render_traj == True:
                self.p_traj = np.concatenate((self.p_traj[1:,:,:], self.p.reshape(1, 2, self.n_a)), axis=0)

            # output
            obs = self._get_obs()
            rew = self._get_reward(a)
            done = self._get_done()
            info = self._get_info()
            # cent_obs = self._get_cent_obs()

        return obs, rew, done, info
        # return obs, cent_obs, rew, done, info

    def render(self, mode="human"): 

        size_agents = 500

        if self.plot_initialized == 0:

            plt.ion()

            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = self.figure_handle.add_axes([left, bottom, width, height],projection = None)

            # Plot agents position
            ax.scatter(self.p[0,:], self.p[1,:], s = size_agents, c = self.color, marker = ".", alpha = 1)
                
            # Observation range
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()
            
            ax.set_xlim(axes_lim[0],axes_lim[1])
            ax.set_ylim(axes_lim[2],axes_lim[3])
            ax.set_xlabel('X position [m]')
            ax.set_ylabel('Y position [m]')
            ax.set_title('Simulation time: %.2f seconds' % self.simulation_time)
            ax.grid(True)

            plt.ioff()
            plt.pause(0.01)

            self.plot_initialized = 1
        else:
            self.figure_handle.axes[0].cla()
            ax = self.figure_handle.axes[0]

            plt.ion()

            # Plot agents position
            ax.scatter(self.p[0,:], self.p[1,:], s = size_agents, c = self.color, marker = ".", alpha = 1)

            # circle = plt.Circle(self.p[:,0], self.d_sen, color='blue', fill=False)
            # ax.add_patch(circle)

            # for iii in range(self.n_a):
            #     circle = plt.Circle(self.p[:,iii], self.r_avoid/2, color='green', fill=False, alpha=0.3)
            #     ax.add_patch(circle)
                
            # for agent_index in range(self.n_a):
            #     ax.text(self.p[0,agent_index], self.p[1,agent_index], self.s_text[agent_index], fontsize=self.fontsize/2)

            ax.text(0.95, 0.95, f'r_avoid: {self.r_avoid:.4f} m', fontsize=self.fontsize, ha='right', va='top', transform=ax.transAxes)

            if self.simulation_time / self.dt > self.traj_len:
                for agent_index in range(self.n_a):
                    distance_index = self._calculate_distances(agent_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,agent_index], self.p_traj[distance_index:,1,agent_index], linestyle='-', color=self.color[agent_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,agent_index], self.p_traj[:,1,agent_index], linestyle='-', color=self.color[agent_index], alpha=0.4)

            if self.dynamics_mode == 'Polar':
                ax.quiver(self.p[0,:], self.p[1,:], self.heading[0,:], self.heading[1,:], scale=30, color=self.color, width = 0.002)
            
            # Plot target shape
            ax.scatter(self.grid_center[0,:], self.grid_center[1,:], s = 10, c = 'blue', marker = ".", alpha = 0.3)
            plt.imshow(self.target_shape, cmap='gray', origin='lower', aspect='equal', alpha=0.1, 
                       extent=[self.shape_bound_points[0], self.shape_bound_points[1], self.shape_bound_points[2], self.shape_bound_points[3]])
            
            ax.plot(np.array([self.boundary_pos[0], self.boundary_pos[0], self.boundary_pos[2], self.boundary_pos[2], self.boundary_pos[0]]), 
                    np.array([self.boundary_pos[3], self.boundary_pos[1], self.boundary_pos[1], self.boundary_pos[3], self.boundary_pos[3]]))
            
            # PLot voronoi partion
            # list_nei, index_grid_in_voroni = self._obtain_grid_in_voronoi(15) # agent index
            # color_voronoi = [(0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0), (0.85, 0.64, 0.12), (0.0, 0, 0), (0.48, 0.4, 0.93), (0, 1.0, 1.0)]
            # for a_i in range(len(list_nei) - 1, -1, -1):
            #     list_grid = index_grid_in_voroni[a_i]
            #     ax.scatter(self.grid_center[0,list_grid], self.grid_center[1,list_grid], s = 70, c = color_voronoi[len(list_nei) - 1 - a_i], marker = ".", alpha = 0.7)

            # for index_i in range(self.n_a):
            #     for index_j in range(index_i):
            #         dist_ij = np.linalg.norm(self.p[:, index_i] - self.p[:, index_j])
            #         if (dist_ij > self.r_avoid - 0.05) and (dist_ij < self.r_avoid + 0.05):
            #             ax.plot(np.array([self.p[0][index_i], self.p[0][index_j]]), np.array([self.p[1][index_i], self.p[1][index_j]]), linestyle='-', color=(0.1, 0.2, 0.3))
            
            # Observation range
            if self.center_view_on_swarm == False:
                axes_lim = self.axis_lim_view_static()
            else:
                axes_lim = self.axis_lim_view_dynamic()

            ax.set_xlim(axes_lim[0],axes_lim[1])
            ax.set_ylim(axes_lim[2],axes_lim[3])
            ax.set_xlabel('X position [m]', fontsize=self.fontsize)
            ax.set_ylabel('Y position [m]', fontsize=self.fontsize)
            ax.set_title('Simulation time: %.2f seconds' % self.simulation_time, fontsize=self.fontsize)
            ax.tick_params(axis='both', labelsize=self.fontsize)
            ax.grid(True)

            plt.ioff()
            plt.pause(0.01)
        
            if self.video:
                self.video.update()

    def axis_lim_view_static(self):
        indent = 0.1
        x_min = self.boundary_pos[0] - indent
        x_max = self.boundary_pos[2] + indent
        y_min = self.boundary_pos[3] - indent
        y_max = self.boundary_pos[1] + indent
        return [x_min, x_max, y_min, y_max]
    
    def axis_lim_view_dynamic(self):
        indent = 0.5
        x_min = np.min(self.p[0]) - indent
        x_max = np.max(self.p[0]) + indent
        y_min = np.min(self.p[1]) - indent
        y_max = np.max(self.p[1]) + indent

        return [x_min, x_max, y_min, y_max]

    def _make_periodic(self, x, is_rel):
        if is_rel:
            x[x >  self.L] -= 2*self.L 
            x[x < -self.L] += 2*self.L
        else:
            x[0, x[0,:] < self.boundary_pos[0]] += 2*self.L
            x[0, x[0,:] > self.boundary_pos[2]] -= 2*self.L
            x[1, x[1,:] < self.boundary_pos[3]] += 2*self.L
            x[1, x[1,:] > self.boundary_pos[1]] -= 2*self.L
        return x
    
    def _normalize_angle(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_size(self):
        size = np.concatenate((np.array([self.size_a for _ in range(self.n_a)]),
                               np.array([self.size_o for _ in range(self.n_o)])))  
        sizes = np.tile(size.reshape(self.n_a,1), (1,self.n_a))
        sizes = sizes + sizes.T
        sizes[np.arange(self.n_a), np.arange(self.n_a)] = 0
        return size, sizes
    
    def _get_mass(self):
        m = np.concatenate((np.array([self.m_a for _ in range(self.n_a)]), 
                            np.array([self.m_o for _ in range(self.n_o)]))) 
        return m

    def _get_observation_space(self):
        if self.is_con_self_state:
            self_flag = 1
        else:
            self_flag = 0

        # self.obs_dim_agent = 2*self.dim*(self.topo_nei_max + 1 + self_flag)
        self.obs_dim_agent = 2*self.dim*(self.topo_nei_max + 1 + self_flag) + self.dim*self.num_obs_grid_max
        # self.obs_dim_agent = 2*self.dim*(self.topo_nei_max + 1 + self_flag) + self.dim*self.num_obs_grid_max + self.topo_nei_max + 1
        # self.obs_dim_agent = (2*self.dim + 1)*(self.topo_nei_max + self_flag) + 2*self.dim + self.dim*self.num_obs_grid_max

        if self.dynamics_mode == 'Polar':
            self.obs_dim_agent += self.dim

        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_agent, self.n_a), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.act_dim_agent, self.n_a), dtype=np.float32)
        return action_space

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
        norms = np.linalg.norm(Pos, axis=0)
        sorted_seq = np.argsort(norms)    
        Pos = Pos[:, sorted_seq]   
        norms = norms[sorted_seq] 
        Pos = Pos[:, norms < norm_threshold] 
        sorted_seq = sorted_seq[norms < norm_threshold]   
        if remove_self == True:
            Pos = Pos[:,1:]  
            sorted_seq = sorted_seq[1:]                    
        Vel = Vel[:, sorted_seq]
        target_Pos = np.zeros( (2, width) )
        target_Vel = np.zeros( (2, width) )
        until_idx = np.min( [Pos.shape[1], width] )
        target_Pos[:, :until_idx] = Pos[:, :until_idx] 
        target_Vel[:, :until_idx] = Vel[:, :until_idx]
        target_Nei = sorted_seq[:until_idx]
        return target_Pos, target_Vel, target_Nei
    
    def _get_trgt_grid_state(self, self_id):
        rel_pos = self.grid_center - self.p[:,[self_id]]
        rel_pos_norm = np.linalg.norm(rel_pos, axis=0)
        min_index = np.argmin(rel_pos_norm)
        min_dist = rel_pos_norm[min_index]
        if min_dist < np.sqrt(2) * self.l_cell / 2:
            in_flag = 1
            target_pos = self.p[:, self_id]
            target_vel = self.dp[:, self_id]
        else:
            in_flag = 0
            target_pos = self.grid_center[:,min_index]
            target_vel = np.array([0, 0])

        in_sense_indices = np.where(rel_pos_norm < self.d_sen)[0]

        return in_flag, target_pos, target_vel, in_sense_indices

    def _regularize_min_velocity(self):
        norms = np.linalg.norm(self.dp, axis=0)
        mask = norms < self.Vel_min
        self.dp[:, mask] *= self.Vel_min / (norms[mask] + 0.00001)

    def _calculate_distances(self, id_self):
        x_coords = self.p_traj[:, 0, id_self]
        y_coords = self.p_traj[:, 1, id_self]
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        points_greater = np.where(distances > self.boundary_L_half)[0]
        
        if len(points_greater) > 0:
            return points_greater[-1] + 1
        else:
            return False
        
    # def _rho_cos_dec(z, delta, r):
    #     if z < delta * r:
    #         return 1.0
    #     elif z < r:
    #         return 0.5 * (1.0 + np.cos(np.pi * (z / r - delta) / (1.0 - delta)))
    #     else:
    #         return 0.0
        
    def _rho_cos_dec(self, z, delta, r):
        z = np.asarray(z)  # Convert input z to NumPy array for vector operations
        result = np.where(z < delta * r, 1.0, np.where(z < r, 0.5 * (1.0 + np.cos(np.pi * (z / r - delta) / (1.0 - delta))), 0.0))
        return result
    
    def _obtain_grid_in_voronoi(self, agent_i):
        # get the neighbors
        list_nei = self.neighbor_index[agent_i]
        list_nei = list_nei[list_nei != -1]

        list_nei = np.append(list_nei, agent_i)
        num_grid_in_voronoi = np.zeros_like(list_nei)
        index_grid_in_voronoi = [[] for _ in range(len(list_nei))]

        # get the sensed grids
        list_grid = self.sensed_index[agent_i]
        list_grid = list_grid[list_grid != -1]

        # get the occupied grids
        list_occupied_grid = self.occupied_index[agent_i]
        list_occupied_grid = list_occupied_grid[list_occupied_grid != -1]

        list_grid = np.concatenate((list_grid, list_occupied_grid))

        for cell_index in list_grid:
            rel_pos_cell_nei = self.p[:,list_nei] - self.grid_center[:,[cell_index]]
            rel_pos_cell_nei_norm = np.linalg.norm(rel_pos_cell_nei, axis=0)
            min_index = np.argmin(rel_pos_cell_nei_norm)
            num_grid_in_voronoi[min_index] += 1
            index_grid_in_voronoi[min_index].append(cell_index)

        return list_nei, index_grid_in_voronoi
            


