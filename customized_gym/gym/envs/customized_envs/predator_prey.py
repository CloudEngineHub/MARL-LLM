__credits__ = ["zhugb@buaa.edu.cn"]

import gym
from gym import error, spaces, utils
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .VideoWriter import VideoWriter
from .envs_cplus.c_lib import as_double_c_array, as_bool_c_array, as_int32_c_array, _load_lib
import ctypes


_LIB = _load_lib(env_name='PredatorPrey')

class PredatorPreySwarmEnv(gym.Env):
    ''' A kind of MPE env.
    ball1 ball2 ball3 ...     (in order of pursuers, escapers and obstacles)
    sf = spring force,  df = damping force
    the forces include: u, ball(spring forces), aerodynamic forces
    x, dx: (2,n), position and vel of agent i;  
    '''

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self, n_p=3, n_e=10):
        
        self.reward_sharing_mode = 'individual'   # select one from ['sharing_mean', 'sharing_max', 'individual'] 
        
        self.penalize_control_effort = False
        self.penalize_collide_walls = True         
        self.penalize_distance = False
        self.penalize_collide_agents = False      
        self.penalize_collide_obstacles = False  

        # dimension
        self.dim = 2 

        # Numbers of agents
        # self.n_p = n_p   # number of pursuers
        # self.n_e = n_e   # number of escapers
        self.n_o = 0     # number of obstacles

        # Observation 
        self.topo_nei_p2p = 2     # pursuer to pursuer 
        self.topo_nei_p2e = 5     # pursuer to escaper 
        self.topo_nei_e2p = 2     # escaper to pursuer 
        self.topo_nei_e2e = 5     # escaper to escaper 
        
        # Action
        self.act_dim_pursuer = 2
        self.act_dim_escaper = 2
        
        # Mass
        self.m_p = 1
        self.m_e = 1     
        self.m_o = 10
        
        # Size
        self.size_p = 0.06    
        self.size_e = 0.035 
        self.size_o = 0.2
        
        # radius of FoV
        self.d_sen_p = 0.3   
        self.d_sen_e = 0.3

        self.linVel_p_max = 0.5  
        self.linVel_e_max = 0.5
       
        self.linAcc_p_max = 1
        self.linAcc_e_max = 1

        ## Properties of obstacles
        self.obstacles_cannot_move = True 
        self.obstacles_is_constant = False
        if self.obstacles_is_constant:   # then specify their locations:
            self.p_o = np.array([[-0.5,0.5], [0,0]])
        ## ======================================== end ========================================

        # Half boundary length
        self.boundary_L_half = 0.5
        self.bound_center = np.zeros(2)

        ## Venue
        self.L = self.boundary_L_half
        self.k_ball = 12       # sphere-sphere contact stiffness  N/m 
        # self.c_ball = 5      # sphere-sphere contact damping N/m/s
        self.k_wall = 100      # sphere-wall contact stiffness  N/m
        self.c_wall = 5        # sphere-wall contact damping N/m/s
        self.c_aero = 2        # sphere aerodynamic drag coefficient N/m/s

        ## Simulation Steps
        self.simulation_time = 0
        self.dt = 0.1
        self.n_frames = 1  
        self.sensitivity = 1 

        ## Rendering
        self.traj_len = 12
        self.plot_initialized = 0
        self.center_view_on_swarm = False
        self.fontsize = 24
        width = 16
        height = 16
        self.figure_handle = plt.figure(figsize = (width,height))
        
    def __reinit__(self, args):
        self.n_p = args.n_p
        self.n_e = args.n_e
        self.n_pe = self.n_p + self.n_e
        self.n_peo = self.n_p + self.n_e + self.n_o
        self.s_text_p = np.char.mod('%d',np.arange(self.n_p))
        self.s_text_e = np.char.mod('%d',np.arange(self.n_e))
        self.render_traj = args.render_traj
        self.traj_len = args.traj_len
        self.video = args.video

        self.is_boundary = args.is_boundary
        if self.is_boundary:
            self.is_periodic = False
        else:
            self.is_periodic = True

        self.dynamics_mode = args.dynamics_mode
        self.billiards_mode = args.billiards_mode
        self.pursuer_strategy = args.pursuer_strategy
        self.escaper_strategy = args.escaper_strategy
        self.is_con_self_state = args.is_con_self_state

        self.is_collide_b2b = np.random.choice([True, False], size=(self.n_pe, self.n_pe))
        self.is_collide_b2w = np.zeros((4, self.n_pe), dtype=bool)
        self.d_b2w = np.ones((4, self.n_peo))

        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self.m = self._get_mass()  
        self.size, self.sizes = self._get_size()  

        if self.billiards_mode:
            self.c_wall = 0.2
            self.c_aero = 0.01

        if self.dynamics_mode == 'Cartesian':
            self.is_Cartesian = True
            self.linAcc_p_min = -1    
            self.linAcc_e_min = -1
            assert (self.linAcc_p_min, self.linAcc_e_min, self.linAcc_p_max, self.linAcc_e_max) == (-1, -1, 1, 1)
        elif self.dynamics_mode == 'Polar':
            self.is_Cartesian = False
            self.linAcc_p_min = 0    
            self.linAcc_e_min = 0
            self.angle_p_max = 0.5   
            self.angle_e_max = 0.5 
            assert self.linAcc_p_min >= 0
            assert self.linAcc_e_min >= 0
        else:
            print('Wrong in linAcc_p_min')
            
        # Energy
        if self.dynamics_mode == 'Cartesian':
            self.max_energy_p = 1000. 
            self.max_energy_e = 1000.  
        elif self.dynamics_mode == 'Polar':
            self.max_energy_p = 1000. 
            self.max_energy_e = 1000. 

        self.color_p = np.tile(np.array([0, 0, 1]), (self.n_p, 1))
        self.color_e = np.tile(np.array([1, 0.5, 0]), (self.n_e, 1))

        if self.video:
            self.video = VideoWriter(output_rate=self.dt, fps=20)
            self.video.video.setup(self.figure_handle, args.video_path) 

    def reset(self):
        self.simulation_time = 0
        self.heading = np.zeros((self.dim, self.n_pe))

        self.bound_center = np.array([0, 0])
        # x_min, y_max, x_max, y_min
        self.boundary_pos = np.array([self.bound_center[0] - self.boundary_L_half,
                                      self.bound_center[1] + self.boundary_L_half,
                                      self.bound_center[0] + self.boundary_L_half,
                                      self.bound_center[1] - self.boundary_L_half], dtype=np.float64) 

        # position
        max_size = np.max(self.size)
        max_respawn_times = 100
        # random_int = np.random.randint(1, 7)
        random_int = 0.5
        # for respawn_time in range(max_respawn_times):
        self.p = np.random.uniform(-random_int + max_size, random_int - max_size, (2, self.n_peo))   # Initialize self.p
            # if self.obstacles_is_constant:
            #     self.p[:, self.n_pe:self.n_peo] = self.p_o
            # _, _, is_collide_b2b = self._get_dist_b2b()
            # if is_collide_b2b.sum() == 0:
            #     break
            # if respawn_time == max_respawn_times - 1:
            #     print('Some particles are overlapped at the initial time !')

        if self.render_traj == True:
            self.p_traj = np.zeros((self.traj_len, 2, self.n_peo))
            self.p_traj[0,:,:] = self.p
        
        # velocity
        self.dp = np.zeros((2, self.n_peo))  
        if self.billiards_mode:
            self.dp = np.random.uniform(-1,1,(2,self.n_peo))  # ice mode                        
        if self.obstacles_cannot_move:
            self.dp[:, self.n_pe:self.n_peo] = 0

        # acceleration
        self.ddp = np.zeros((2, self.n_peo))  

        # energy                                   
        self.energy = np.array([self.max_energy_p for _ in range(self.n_p)] + [self.max_energy_e for _ in range(self.n_e)]).reshape(1, self.n_pe)

        if self.dynamics_mode == 'Polar': 
            self.theta = np.pi * np.random.uniform(-1,1, (1, self.n_peo))
            # self.theta = np.pi * np.zeros((1, self.n_peo))
            self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0)  

        return self._get_obs()

    def _get_obs(self):

        self.obs = np.zeros(self.observation_space.shape) 
        # self.neighbor_index = -1 * np.ones((self.n_a, self.topo_nei_max), dtype=np.int32)
        conditions = np.array([self.is_periodic, self.is_Cartesian, self.is_con_self_state])

        _LIB._get_observation(as_double_c_array(self.p), 
                              as_double_c_array(self.dp), 
                              as_double_c_array(self.heading),
                              as_double_c_array(self.obs),
                              as_double_c_array(self.boundary_pos),
                              ctypes.c_double(self.d_sen_p), 
                              ctypes.c_double(self.d_sen_e), 
                              ctypes.c_int(self.topo_nei_p2p), 
                              ctypes.c_int(self.topo_nei_p2e), 
                              ctypes.c_int(self.topo_nei_e2p), 
                              ctypes.c_int(self.topo_nei_e2e), 
                              ctypes.c_int(self.n_p),
                              ctypes.c_int(self.n_e), 
                              ctypes.c_int(self.obs_dim_max), 
                              ctypes.c_int(self.dim), 
                              as_bool_c_array(conditions))

        # self.obs = np.zeros(self.observation_space.shape)   

        # for i in range(self.n_p):
        #     '''
        #     捕食者： 自己的位置速度，相对其他捕食者的位置速度，相对其他逃跑者的位置速度，自己的能量; 如果是polar_mode 则加上heading
        #     有一个最大关注个数的 topo_n, 且有FoV
        #     '''
        #     relPos_p2p = self.p[:,:self.n_p] - self.p[:,[i]]
        #     if self.is_periodic:
        #         relPos_p2p = self._make_periodic(relPos_p2p, is_rel=True)
        #     relVel_p2p = self.dp[:,:self.n_p] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,:self.n_p] - self.heading[:, [i]]
        #     relPos_p2p, relVel_p2p, _ = self._get_focused(relPos_p2p, relVel_p2p, self.d_sen_p, self.topo_nei_p2p, True)  
           
        #     relPos_p2e = self.p[:,self.n_p:self.n_pe] - self.p[:,[i]]
        #     if self.is_periodic: 
        #         relPos_p2e = self._make_periodic(relPos_p2e, is_rel=True)
        #     relVel_p2e = self.dp[:,self.n_p:self.n_pe] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,self.n_p:self.n_pe] - self.heading[:,[i]]
        #     relPos_p2e, relVel_p2e, _ = self._get_focused(relPos_p2e, relVel_p2e, self.d_sen_p, self.topo_nei_p2e, False) 
          
        #     if self.is_con_self_state:
        #         vel_or_heading = self.dp[:, [i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,[i]]
        #         obs_pursuer_pos = np.concatenate((self.p[:, [i]], relPos_p2p, relPos_p2e), axis=1)
        #         obs_pursuer_vel = np.concatenate((vel_or_heading, relVel_p2p, relVel_p2e), axis=1)
        #         obs_pursuer = np.concatenate((obs_pursuer_pos, obs_pursuer_vel), axis=0) 
        #     else:
        #         obs_pursuer_pos = np.concatenate((relPos_p2p, relPos_p2e), axis=1)
        #         obs_pursuer_vel = np.concatenate((relVel_p2p, relVel_p2e), axis=1)
        #         obs_pursuer = np.concatenate((obs_pursuer_pos, obs_pursuer_vel), axis=0)

        #     # if self.dynamics_mode == 'Cartesian':
        #     #     self.obs[:, i] = obs_pursuer.T.reshape(-1)       
        #     # elif self.dynamics_mode == 'Polar':
        #     #     self.obs[:self.obs_dim_pursuer - self.dim, i] = obs_pursuer.T.reshape(-1)       
        #     #     self.obs[self.obs_dim_pursuer - self.dim:self.obs_dim_pursuer, i] = self.heading[:,i]
        #     self.obs[:, i] = obs_pursuer.T.reshape(-1) 

        # for i in range(self.n_p, self.n_pe):
        #     '''
        #     逃跑者： 自己的位置速度，相对其他捕食者的位置速度，相对其他逃跑者的位置速度，自己的能量; 如果是polar_mode 则加上heading
        #     有一个最大关注个数的 topo_n, 且有FoV
        #     '''
        #     relPos_e2p = self.p[:,:self.n_p] - self.p[:,[i]]  
        #     if self.is_periodic: 
        #         relPos_e2p = self._make_periodic(relPos_e2p, is_rel=True)
        #     relVel_e2p = self.dp[:,:self.n_p] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,:self.n_p] - self.heading[:,[i]]
        #     relPos_e2p, relVel_e2p, _ = self._get_focused(relPos_e2p, relVel_e2p, self.d_sen_e, self.topo_nei_e2p, False) 
            
        #     relPos_e2e = self.p[:,self.n_p:self.n_pe] - self.p[:,[i]]
        #     if self.is_periodic: 
        #         relPos_e2e = self._make_periodic(relPos_e2e, is_rel=True)
        #     relVel_e2e = self.dp[:,self.n_p:self.n_pe] - self.dp[:,[i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,self.n_p:self.n_pe] - self.heading[:,[i]]
        #     relPos_e2e, relVel_e2e, _ = self._get_focused(relPos_e2e, relVel_e2e, self.d_sen_e, self.topo_nei_e2e, True)  

        #     if self.is_con_self_state:
        #         vel_or_heading = self.dp[:, [i]] if self.dynamics_mode == 'Cartesian' else self.heading[:,[i]]
        #         obs_escaper_pos = np.concatenate((self.p[:, [i]], relPos_e2p, relPos_e2e), axis=1)
        #         obs_escaper_vel = np.concatenate((vel_or_heading, relVel_e2p, relVel_e2e), axis=1)
        #         obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel), axis=0)
        #     else:
        #         obs_escaper_pos = np.concatenate((relPos_e2p, relPos_e2e), axis=1)
        #         obs_escaper_vel = np.concatenate((relVel_e2p, relVel_e2e), axis=1)
        #         obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel), axis=0)

        #     # if self.dynamics_mode == 'Cartesian':
        #     #     self.obs[:, i] = obs_escaper.T.reshape(-1)         
        #     # elif self.dynamics_mode == 'Polar':
        #     #     self.obs[:self.obs_dim_escaper - self.dim, i] = obs_escaper.T.reshape(-1)        
        #     #     self.obs[self.obs_dim_escaper - self.dim:self.obs_dim_escaper, i] = self.heading[:,i] 
        #     self.obs[:, i] = obs_escaper.T.reshape(-1)

        return self.obs
      
    def _get_reward(self, a):

        reward_p = np.zeros((1, self.n_p))
        reward_e = np.zeros((1, self.n_e))
        reward_p =   1.0 * self.is_collide_b2b[self.n_p:self.n_pe, :self.n_p].sum(axis=0, keepdims=True).astype(float)                      
        reward_e = - 1.0 * self.is_collide_b2b[self.n_p:self.n_pe, :self.n_p].sum(axis=1, keepdims=True).astype(float).reshape(1,self.n_e)  

        if self.penalize_distance:
            reward_p += - self.d_b2b_center[self.n_p:self.n_pe, :self.n_p].sum(axis=0, keepdims=True)
            reward_e +=   self.d_b2b_center[self.n_p:self.n_pe, :self.n_p].sum(axis=1, keepdims=True).reshape(1,self.n_e)

        if self.penalize_control_effort:
            if self.dynamics_mode == 'Cartesian':
                reward_p -= 1 * np.sqrt(a[[0],:self.n_p]**2 + a[[1],:self.n_p]**2)
                reward_e -= 1 * np.sqrt(a[[0],self.n_p:self.n_pe]**2 + a[[1],self.n_p:self.n_pe]**2)
            elif self.dynamics_mode == 'Polar':
                reward_p -= 1 * np.abs(a[[0],:self.n_p]) + 0 * np.abs(a[[1],:self.n_p])
                reward_e -= 1 * np.abs(a[[0],self.n_p:self.n_pe]) + 0 * np.abs(a[[1],self.n_p:self.n_pe])     
      
        if self.penalize_collide_agents:
            reward_p -= self.is_collide_b2b[:self.n_p, :self.n_p].sum(axis=0, keepdims=True)
            reward_e -= self.is_collide_b2b[self.n_p:self.n_pe, self.n_p:self.n_pe].sum(axis=0, keepdims=True)

        if self.penalize_collide_obstacles:
            reward_p -= 5 * self.is_collide_b2b[self.n_pe:self.n_peo, 0:self.n_p].sum(axis=0, keepdims=True)          
            reward_e -= 5 * self.is_collide_b2b[self.n_pe:self.n_peo, self.n_p:self.n_pe].sum(axis=0, keepdims=True) 
        
        if self.penalize_collide_walls and self.is_periodic == False:
            reward_p -= 0 * self.is_collide_b2w[:, :self.n_p].sum(axis=0, keepdims=True)            
            reward_e -= 15 * self.is_collide_b2w[:, self.n_p:self.n_pe].sum(axis=0, keepdims=True)  
            # print(np.sum(self.is_collide_b2w[:, self.n_p:self.n_pe].sum(axis=0, keepdims=True)))

        if self.reward_sharing_mode == 'sharing_mean':
            reward_p[:] = np.mean(reward_p) 
            reward_e[:] = np.mean(reward_e)
        elif self.reward_sharing_mode == 'sharing_max':
            reward_p[:] = np.max(reward_p) 
            reward_e[:] = np.max(reward_e)
        elif self.reward_sharing_mode == 'individual':
            pass
        else:
            print('reward mode error !!')

        reward = np.concatenate((reward_p, reward_e), axis=1) 
        # print(reward)
        return reward

    def _get_dist_b2b(self):
        all_pos = np.tile(self.p, (self.n_peo, 1))   
        my_pos = self.p.T.reshape(2*self.n_peo, 1) 
        my_pos = np.tile(my_pos, (1, self.n_peo))   
        relative_p_2n_n =  all_pos - my_pos
        if self.is_periodic == True:
            relative_p_2n_n = self._make_periodic(relative_p_2n_n, is_rel=True)
        self.d_b2b_center = np.sqrt(relative_p_2n_n[::2,:]**2 + relative_p_2n_n[1::2,:]**2)  
        d_b2b_edge = self.d_b2b_center - self.sizes
        isCollision = (d_b2b_edge < 0)
        d_b2b_edge = np.abs(d_b2b_edge)
        # print(self.d_b2b_center)
        return self.d_b2b_center, d_b2b_edge, isCollision

    def _get_dist_b2w(self):
        _LIB._get_dist_b2w(as_double_c_array(self.p), 
                           as_double_c_array(self.size), 
                           as_double_c_array(self.d_b2w), 
                           as_bool_c_array(self.is_collide_b2w),
                           ctypes.c_int(self.dim), 
                           ctypes.c_int(self.n_pe), 
                           as_double_c_array(self.boundary_pos))
        # p = self.p
        # r = self.size
        # d_b2w = np.zeros((4, self.n_pe))
        # # isCollision = np.zeros((4,self.n_ao))
        # for i in range(self.n_pe):
        #     d_b2w[:,i] = np.array([ p[0,i] - r[i] - self.boundary_pos[0], 
        #                             self.boundary_pos[1] - (p[1,i] + r[i]),
        #                             self.boundary_pos[2] - (p[0,i] + r[i]),
        #                             p[1,i] - r[i] - self.boundary_pos[3]])  
        # self.is_collide_b2w = d_b2w < 0
        # self.d_b2w = np.abs(d_b2w) 

    def _get_done(self):
        all_done = np.zeros((1, self.n_pe)).astype(bool)
        return all_done

    def _get_info(self):
        return np.array( [None, None, None] ).reshape(3,1)

    def step(self, a):  
        self.simulation_time += self.dt 

        for _ in range(self.n_frames): 
            if self.dynamics_mode == 'Polar':  
                a[0, :self.n_p] *= self.angle_p_max
                a[0, self.n_p:self.n_pe] *= self.angle_e_max
                a[1, :self.n_p] =          (self.linAcc_p_max-self.linAcc_p_min)/2 * a[1,:self.n_p] +          (self.linAcc_p_max+self.linAcc_p_min)/2 
                a[1, self.n_p:self.n_pe] = (self.linAcc_e_max-self.linAcc_e_min)/2 * a[1,self.n_p:self.n_pe] + (self.linAcc_e_max+self.linAcc_e_min)/2 

            self.d_b2b_center, self.d_b2b_edge, self.is_collide_b2b = self._get_dist_b2b()

            # inter-agent elastic force
            sf_b2b = np.zeros((2, self.n_peo))
            _LIB._sf_b2b_all(as_double_c_array(self.p), 
                             as_double_c_array(sf_b2b), 
                             as_double_c_array(self.d_b2b_edge), 
                             as_bool_c_array(self.is_collide_b2b),
                             as_double_c_array(self.boundary_pos),
                             as_double_c_array(self.d_b2b_center),
                             ctypes.c_int(self.n_peo), 
                             ctypes.c_int(self.dim), 
                             ctypes.c_double(self.k_ball),
                             ctypes.c_bool(self.is_periodic))
            # sf_b2b_all = np.zeros((2*self.n_peo, self.n_peo))   
            # for i in range(self.n_peo):
            #     for j in range(i):
            #         delta = self.p[:,j] - self.p[:,i]
            #         if self.is_periodic:
            #             delta = self._make_periodic(delta, is_rel=True)
            #         dir = delta / self.d_b2b_center[i,j]
            #         sf_b2b_all[2*i:2*(i+1),j] = self.is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self.k_ball * (-dir)
            #         sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  
                   
            # sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self.n_peo,2).T

            if self.is_periodic == False:
                self._get_dist_b2w()
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self.k_wall   
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self.dp, self.dp), axis=0))  *  self.c_wall   

            if self.pursuer_strategy == 'input':
                pass
            elif self.pursuer_strategy == 'static':
                a[:,:self.n_p] = np.zeros((self.act_dim_pursuer, self.n_p))                
            elif self.pursuer_strategy == 'random':
                a[:,:self.n_p] = np.random.uniform(-1, 1, (self.act_dim_pursuer, self.n_p)) 
                if self.dynamics_mode == 'Polar': 
                    a[0, :self.n_p] *= self.angle_p_max
                    a[1, :self.n_p] = (self.linAcc_p_max-self.linAcc_p_min)/2 * a[1,:self.n_p] + (self.linAcc_p_max+self.linAcc_p_min)/2 
            elif self.pursuer_strategy == 'nearest':
                ind_nearest = np.argmin(self.d_b2b_center[:self.n_p, self.n_p:self.n_pe], axis=1)
                goto_pos =  self.p[:, self.n_p + ind_nearest] - self.p[:,:self.n_p]   
                if self.is_periodic == True:
                    goto_pos = self._make_periodic(goto_pos, is_rel=True)
                ranges = np.sqrt(goto_pos[[0],:]**2 + goto_pos[[1],:]**2)
                goto_dir = goto_pos / ranges   
                if self.dynamics_mode == 'Cartesian':
                    a[:,:self.n_p] = 1 * goto_dir
                elif self.dynamics_mode == 'Polar':
                    goto_dir = np.concatenate((goto_dir, np.zeros((1,self.n_p))), axis=0).T 
                    heading = np.concatenate((self.heading[:,:self.n_p], np.zeros((1, self.n_p))), axis=0).T
                    desired_rotate_angle = np.cross(heading, goto_dir)[:,-1] 
                    desired_rotate_angle[desired_rotate_angle>self.angle_p_max] = self.angle_p_max
                    desired_rotate_angle[desired_rotate_angle<-self.angle_p_max] = -self.angle_p_max
                    a[0, :self.n_p] = desired_rotate_angle
                    a[1, :self.n_p] = self.linAcc_p_max
            else:
                print('Wrong in Step function')
                    
            if self.escaper_strategy == 'input':
                pass
            elif self.escaper_strategy == 'static':
                a[:,self.n_p:self.n_pe] = np.zeros((self.act_dim_pursuer, self.n_e)) 
            elif self.escaper_strategy == 'nearest':
                ind_nearest = np.argmin(self.d_b2b_center[self.n_p:self.n_pe, :self.n_p], axis=1)
                goto_pos =  - self.p[:, ind_nearest] + self.p[:, self.n_p:self.n_pe]  
                if self.is_periodic == True:
                    goto_pos = self._make_periodic(goto_pos, is_rel=True)
                ranges = np.sqrt(goto_pos[[0],:]**2 + goto_pos[[1],:]**2)
                goto_dir = goto_pos / ranges  
                if self.dynamics_mode == 'Cartesian':
                    a[:, self.n_p:self.n_pe] = 1 * goto_dir
                elif self.dynamics_mode == 'Polar':
                    goto_dir = np.concatenate((goto_dir, np.zeros((1,self.n_e))), axis=0).T 
                    heading = np.concatenate((self.heading[:,self.n_p:self.n_pe], np.zeros((1, self.n_e))), axis=0).T 
                    desired_rotate_angle = np.cross(heading, goto_dir)[:,-1]  
                    desired_rotate_angle[desired_rotate_angle>self.angle_e_max] = self.angle_e_max
                    desired_rotate_angle[desired_rotate_angle<-self.angle_e_max] = -self.angle_e_max
                    a[0, self.n_p:self.n_pe] = desired_rotate_angle
                    a[1, self.n_p:self.n_pe] = self.linAcc_e_max     

            if self.dynamics_mode == 'Cartesian':
                u = a   
            elif self.dynamics_mode == 'Polar':      
                self.theta += a[[0],:]
                self.theta = self._normalize_angle(self.theta)
                self.heading = np.concatenate((np.cos(self.theta), np.sin(self.theta)), axis=0) 
                u = a[[1], :] * self.heading 
            else:
                print('Wrong in updating dynamics')

            if self.is_periodic == True:
                F = self.sensitivity * u + sf_b2b - self.c_aero*self.dp
                # F = self.sensitivity * u  + sf_b2b + df_b2b - self.c_aero*dp
            elif self.is_periodic == False:
                F = self.sensitivity * u + sf_b2b - self.c_aero*self.dp + sf_b2w + df_b2w 
            else:
                print('Wrong in consider walls !!!')

            # acceleration
            self.ddp = F/self.m

            # velocity
            self.dp += self.ddp * self.dt
            if self.obstacles_cannot_move:
                self.dp[:, self.n_pe:self.n_peo] = 0
            self.dp[:,:self.n_p] = np.clip(self.dp[:,:self.n_p], -self.linVel_p_max, self.linVel_p_max)
            self.dp[:,self.n_p:self.n_pe] = np.clip(self.dp[:,self.n_p:self.n_pe], -self.linVel_e_max, self.linVel_e_max)

            # energy
            energy = np.tile(self.energy, (2, 1))
            self.dp[:,:self.n_pe][energy < 0.5] = 0
            speeds = np.sqrt(self.dp[[0],:self.n_pe]**2 + self.dp[[1],:self.n_pe]**2)
            self.energy -= speeds
            self.energy[speeds < 0.01] += 0.1   
            self.energy[0,:self.n_p][self.energy[0,:self.n_p]>self.max_energy_p] = self.max_energy_p
            self.energy[0,self.n_p:][self.energy[0,self.n_p:]>self.max_energy_e] = self.max_energy_e

            # position
            self.p += self.dp * self.dt
            if self.obstacles_is_constant:
                self.p[:, self.n_pe:self.n_peo] = self.p_o
            if self.is_periodic:
                self.p = self._make_periodic(self.p, is_rel=False)

            if self.render_traj == True:
                self.p_traj = np.concatenate((self.p_traj[1:,:,:], self.p.reshape(1, 2, self.n_peo)), axis=0)

        return self._get_obs(), self._get_reward(a), self._get_done(), self._get_info()

    def render(self, mode="human"): 

        # size_pursuer = 4500
        # size_escaper = 2000
        size_pursuer = 3200
        size_escaper = 1200
        # radius_m = 0.06
        # area_m2 = np.pi * (radius_m/2) ** 2
        # size_pursuer = area_m2 * (39.3701 * 72 / 4 * 3) ** 2

        if self.plot_initialized == 0:

            plt.ion()

            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = self.figure_handle.add_axes([left, bottom, width, height],projection = None)

            # Plot agents position
            ax.scatter(self.p[0,:self.n_p], self.p[1,:self.n_p], s = size_pursuer, c = self.color_p, marker = ".", alpha = 1)
            ax.scatter(self.p[0,self.n_p:self.n_pe], self.p[1,self.n_p:self.n_pe], s = size_escaper, c = self.color_e, marker = ".", alpha = 1)
                
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
            ax.grid

            plt.ioff()
            plt.pause(0.01)

            self.plot_initialized = 1
        else:
            self.figure_handle.axes[0].cla()
            ax = self.figure_handle.axes[0]

            plt.ion()

            # Plot agents position
            # center = (0.5, 0.5)
            # radius = 0.2
            # circle = patches.Circle(center, radius, edgecolor='blue', facecolor='cyan', alpha=0.5)
            # ax.add_patch(circle)

            ax.scatter(self.p[0,:self.n_p], self.p[1,:self.n_p], s = size_pursuer, c = self.color_p, marker = ".", alpha = 1)
            ax.scatter(self.p[0,self.n_p:self.n_pe], self.p[1,self.n_p:self.n_pe], s = size_escaper, c = self.color_e, marker = ".", alpha = 1)

            # for pursuer_index in range(self.n_p):
            #     ax.text(self.p[0,pursuer_index], self.p[1,pursuer_index], self.s_text_p[pursuer_index], fontsize=self.fontsize)
            # for escaper_index in range(self.n_p, self.n_pe):
            #     ax.text(self.p[0,escaper_index], self.p[1,escaper_index], self.s_text_e[escaper_index - self.n_p], fontsize=self.fontsize)

            if self.simulation_time / self.dt > self.traj_len:
                for pursuer_index in range(self.n_p):
                    distance_index = self._calculate_distances(pursuer_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,pursuer_index], self.p_traj[distance_index:,1,pursuer_index], linestyle='-', color=self.color_p[pursuer_index], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,pursuer_index], self.p_traj[:,1,pursuer_index], linestyle='-', color=self.color_p[pursuer_index], alpha=0.4)

                for escaper_index in range(self.n_p, self.n_pe):
                    distance_index = self._calculate_distances(escaper_index)
                    if distance_index:
                        ax.plot(self.p_traj[distance_index:,0,escaper_index], self.p_traj[distance_index:,1,escaper_index], linestyle='-', color=self.color_e[escaper_index - self.n_p], alpha=0.4)
                    else:
                        ax.plot(self.p_traj[:,0,escaper_index], self.p_traj[:,1,escaper_index], linestyle='-', color=self.color_e[escaper_index - self.n_p], alpha=0.4)

            if self.dynamics_mode == 'Polar':
                ax.quiver(self.p[0,:self.n_p], self.p[1,:self.n_p], self.heading[0,:self.n_p], self.heading[1,:self.n_p], scale=20, color=self.color_p, width = 0.002)
                ax.quiver(self.p[0,self.n_p:self.n_pe], self.p[1,self.n_p:self.n_pe], self.heading[0,self.n_p:self.n_pe], self.heading[1,self.n_p:self.n_pe], scale=20, color=self.color_e, width = 0.002)
            
            ax.plot(np.array([self.boundary_pos[0], self.boundary_pos[0], self.boundary_pos[2], self.boundary_pos[2], self.boundary_pos[0]]), 
                    np.array([self.boundary_pos[3], self.boundary_pos[1], self.boundary_pos[1], self.boundary_pos[3], self.boundary_pos[3]]))
            
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
        indent = 0.2
        x_min = self.boundary_pos[0] - indent
        x_max = self.boundary_pos[2] + indent
        y_min = self.boundary_pos[3] - indent
        y_max = self.boundary_pos[1] + indent
        return [x_min, x_max, y_min, y_max]
    
    def axis_lim_view_dynamic(self):
        indent = 0.2
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
        size = np.concatenate((np.array([self.size_p for _ in range(self.n_p)]), 
                               np.array([self.size_e for _ in range(self.n_e)]), 
                               np.array([self.size_o for _ in range(self.n_o)])))  
        sizes = np.tile(size.reshape(self.n_peo,1), (1,self.n_peo))
        sizes = sizes + sizes.T
        sizes[np.arange(self.n_peo), np.arange(self.n_peo)] = 0
        return size, sizes
    
    def _get_mass(self):
        m = np.concatenate((np.array([self.m_p for _ in range(self.n_p)]), 
                            np.array([self.m_e for _ in range(self.n_e)]), 
                            np.array([self.m_o for _ in range(self.n_o)]))) 
        return m

    def _get_observation_space(self):
        if self.is_con_self_state:
            self_flag = 1
        else:
            self_flag = 0

        self.topo_n_p = self.topo_nei_p2p + self.topo_nei_p2e
        self.topo_n_e = self.topo_nei_e2p + self.topo_nei_e2e 
        self.obs_dim_pursuer = (self_flag + self.topo_n_p) * self.dim * 2  
        self.obs_dim_escaper = (self_flag + self.topo_n_e) * self.dim * 2
        if self.dynamics_mode == 'Polar':
            self.obs_dim_pursuer += 2
            self.obs_dim_escaper += 2  
        self.obs_dim_max = np.max([self.obs_dim_pursuer, self.obs_dim_escaper])   
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_max, self.n_pe), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        act_dim_max = np.max([self.act_dim_pursuer, self.act_dim_escaper])
        action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(act_dim_max, self.n_pe), dtype=np.float32)
        return action_space

    def _get_focused(self, Pos, Vel, norm_threshold, width, remove_self):
        # assert A.shape[0] == 2
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

    def _calculate_distances(self, id_self):
        x_coords = self.p_traj[:, 0, id_self]
        y_coords = self.p_traj[:, 1, id_self]
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        points_greater = np.where(distances > self.boundary_L_half)[0]
        
        if len(points_greater) > 0:
            return points_greater[-1] + 1
        else:
            return False