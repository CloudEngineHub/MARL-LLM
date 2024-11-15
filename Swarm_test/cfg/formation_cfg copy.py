'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import numpy.linalg as lg
import sympy as sp
import argparse
import time
import os
import matplotlib.pyplot as plt

start_time = time.time()
parser = argparse.ArgumentParser("Gym-FormationSwarm Arguments")


col_num_dict = {
    "rectangle": 8
}

# 根据 shape 获取对应的列数
shape = "rectangle"  # 这里假设 shape 的值为 "I"，您可以根据实际情况更改
col_num = col_num_dict.get(shape, None)

current_file_path = os.path.dirname(os.path.abspath(__file__))
file_path = current_file_path + '/dot_mat/' + shape + '.txt'

with open(file_path, 'r') as file_id:
    dot_ = []
    for line in file_id:
        # 去除末尾空格并按逗号分割
        line_elements = line.strip().split(',')
        # 检查最后一个元素是否为空
        if line_elements[-1] == '':
            line_elements.pop()
        
        dot_.append(line_elements)

col_num = 8

# 获取十六进制数据并转换为十进制
dec_data = np.array([[int(hex_value, 16) for hex_value in line] for line in dot_])

# 获取二进制数据（字符格式）
bin_data_char = np.array([list(format(dec_value, '08b')) for dec_value in dec_data.flat], dtype=int)

# 获取二进制数据（数值格式）
# bin_data_num = bin_data_char.reshape(64 * col_num, 8)

# 重置计数器
count = 0

# 创建二进制数据数组
bin_data = np.zeros((64, 8 * col_num), dtype=int)

# 将二进制数据填充到数组中
for j in range(col_num):
    for i in range(64):
        bin_data[i] = bin_data_char[8 * i: 8 * i + 8].flatten()
        count += 1

# 获取所有网格的中心位置
x_pos = []
y_pos = []
for i in range(64):
    for j in range(8 * col_num):
        if bin_data[i, j] == 1:
            x_pos.append(j - 0.5)
            y_pos.append(64 - (i - 0.5))

# 计算平均位置和相对位置
center_pos = np.array([np.mean(x_pos), np.mean(y_pos), 0])
grid_pos = np.array([x_pos, y_pos, np.zeros(len(x_pos))])
grid_pos_rel = grid_pos - center_pos[:, np.newaxis]

# 缩放目标形状
target_hight = 1
real_hight = np.max(y_pos) - np.min(y_pos)
h_scale = target_hight / real_hight
grid_pos_rel_nor = h_scale * grid_pos_rel

# 扩展网格
k_grid = 1
grid_x_subdivide = []
grid_y_subdivide = []
for grid_index in range(len(grid_pos_rel_nor[0])):
    x_amplify = np.linspace(grid_pos_rel_nor[0, grid_index] - 0.25 * h_scale,
                            grid_pos_rel_nor[0, grid_index] + h_scale * (1 / 2 + 1 / 4 - 1 / k_grid), k_grid)
    y_amplify = np.linspace(grid_pos_rel_nor[1, grid_index] - 0.25 * h_scale,
                            grid_pos_rel_nor[1, grid_index] + h_scale * (1 / 2 + 1 / 4 - 1 / k_grid), k_grid)
    for x_amp_index in range(len(x_amplify)):
        for y_amp_index in range(len(y_amplify)):
            grid_x_subdivide.append(x_amplify[x_amp_index])
            grid_y_subdivide.append(y_amplify[y_amp_index])
grid_center = np.array([grid_x_subdivide, grid_y_subdivide, np.zeros(len(grid_x_subdivide))])

# max_x_index = np.argmax(grid_center[0])
# max_x_point = grid_center[:, max_x_index]
# min_x_index = np.argmin(grid_center[0])
# min_x_point = grid_center[:, min_x_index]
# max_y_index = np.argmax(grid_center[1])
# max_y_point = grid_center[:, max_y_index]
# min_y_index = np.argmin(grid_center[1])
# min_y_point = grid_center[:, min_y_index]
# boundary_points = np.column_stack((max_x_point, min_y_point, max_y_point))

from sklearn.neighbors import NearestNeighbors
grid_centers = grid_center[:2].T  # 转置为 (1000, 2)，即每行是一个点
# 最近邻参数设置
nbrs = NearestNeighbors(n_neighbors=14, algorithm='ball_tree').fit(grid_centers)
distances, indices = nbrs.kneighbors(grid_centers)
# 设置一个距离阈值，用于判断边界
distance_threshold = 0.05
# 找到距离较大的点，即邻居较少的点
boundary_points_mask = np.max(distances, axis=1) > distance_threshold
# 提取边缘格子的中心坐标
boundary_points = grid_centers[boundary_points_mask]
# 转置回 2xn 的矩阵
boundary_points = boundary_points.T

# 绘制结果
plt.scatter(grid_centers[:, 0], grid_centers[:, 1], color='blue', label='Grid centers')
plt.scatter(boundary_points[0, :], boundary_points[1, :], color='red', label='Boundary points')
plt.legend()
plt.show()

# # 创建一个三维图形对象
# fig = plt.figure(figsize=(12,12))

# # 获取坐标
# x_coords = grid_center[0, :]
# y_coords = grid_center[1, :]
# z_coords = grid_center[2, :]

# # 绘制散点图
# plt.scatter(x_coords, y_coords, c='b', marker='o')
# plt.axis('equal')

# # 显示图形
# plt.show()


## ==================== User settings ===================='''
parser.add_argument("--n_a", type=int, default=20, help='number of agents') 
parser.add_argument("--is_boundary", type=bool, default=True, help='Set whether has wall or periodic boundaries') 
parser.add_argument("--is_con_self_state", type=bool, default=True, help="Whether contain myself state in the observation") 
parser.add_argument("--dynamics_mode", type=str, default='Cartesian', help=" select one from ['Cartesian', 'Polar']") 
parser.add_argument("--render-traj", type=bool, default=True, help=" whether render trajectories of agents") 
parser.add_argument("--traj_len", type=int, default=55, help="length of the trajectory")
parser.add_argument("--agent_strategy", type=str, default='input', help="the agent's strategy, please select one from ['input','random','rule']") 
parser.add_argument("--grid_center_pos", type=type(grid_center), default=grid_center, help="an array containing all position of the grid centers")
parser.add_argument("--boundary_points", type=type(boundary_points), default=boundary_points, help="boundary_points")
parser.add_argument("--h_scale", type=float, default=h_scale, help="h_scale") 
parser.add_argument("--k_grid", type=float, default=k_grid, help="k_grid")
parser.add_argument("--video", type=bool, default=False, help="Record video")
## ==================== End of User settings ====================

## ==================== Training Parameters ====================
parser.add_argument("--env_name", default="formation", type=str)
parser.add_argument("--seed", default=226, type=int, help="Random seed")
parser.add_argument("--n_rollout_threads", default=1, type=int)
parser.add_argument("--n_training_threads", default=5, type=int)
parser.add_argument("--buffer_length", default=int(1e4), type=int) # 5e5
parser.add_argument("--n_episodes", default=1200, type=int)
parser.add_argument("--episode_length", default=200, type=int)
parser.add_argument("--batch_size", default=512, type=int, help="Batch size for model training") # 256    
parser.add_argument("--hidden_dim", default=64, type=int)
parser.add_argument("--lr_actor", default=1e-4, type=float)
parser.add_argument("--lr_critic", default=1e-3, type=float)
parser.add_argument("--lambda_s", default=30, type=float, help="the coefficient of smoothness-inducing regularization")
parser.add_argument("--epsilon_p", default=0.03, type=float, help="the amptitude of state perturbation")
parser.add_argument("--epsilon", default=0.1, type=float)
parser.add_argument("--noise_scale", default=0.7, type=float)
parser.add_argument("--tau", default=0.01, type=float)
parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'])
parser.add_argument("--save_interval", default=50, type=int, help="save data for every 'save_interval' episodes")
## ==================== Training Parameters ====================

gpsargs = parser.parse_args()
end_time = time.time()
print('load config parameters takes ', end_time - start_time)