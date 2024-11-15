'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import numpy.linalg as lg
import pickle
import argparse
import time
import os
import matplotlib.pyplot as plt
import cv2
import glob

start_time = time.time()
parser = argparse.ArgumentParser("Gym-FormationSwarm Arguments")

results = {
    "l_cell": [],
    "grid_coords": [],
    "binary_image": [],
    "shape_bound_points": []
}

# 处理每张图片的函数
def process_image(image_path):
    # 读取图像并将其二值化
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 裁减边上的空余部份，并翻转图像
    black_pixels = np.argwhere(binary_image == 0)
    min_y, min_x = black_pixels.min(axis=0)
    max_y, max_x = black_pixels.max(axis=0)
    binary_image = binary_image[min_y:max_y + 1, min_x:max_x + 1]

    height, width = binary_image.shape
    binary_image = np.dot(np.fliplr(np.eye(height)), binary_image)

    # 设置网格的大小
    grid_size = 36  # 根据需求调整网格大小 36--star, 36--shi

    # 提取黑色区域中的网格中心
    black_grid_coords = []
    for i in range(grid_size, height - grid_size, grid_size):
        for j in range(grid_size, width - grid_size, grid_size):
            # 网格区域
            grid_section = binary_image[i:i + grid_size, j:j + grid_size]
            
            # 计算网格中心的坐标
            center_x = j + grid_size / 2
            center_y = i + grid_size / 2

            # 计算黑色像素的数量和比例
            black_pixel_count = np.sum(grid_section == 0)
            total_pixel_count = grid_size * grid_size
            black_pixel_ratio = black_pixel_count / total_pixel_count

            # 如果黑色像素比例达到阈值，保存该网格中心
            if black_pixel_ratio >= 1:
                black_grid_coords.append([center_x, center_y])

    # 将提取的网格坐标转化为numpy数组
    black_grid_coords = np.array(black_grid_coords, dtype=np.float64)

    print("The number of grid: ", len(black_grid_coords))

    # 网格平移到原点，并计算网格端点
    x_mean_grid = np.mean(black_grid_coords[:,0])
    y_mean_grid = np.mean(black_grid_coords[:,1])
    black_grid_coords[:,0] -= x_mean_grid
    black_grid_coords[:,1] -= y_mean_grid

    x_min = np.min(black_grid_coords[:,0])
    x_max = np.max(black_grid_coords[:,0])
    y_min = np.min(black_grid_coords[:,1])
    y_max = np.max(black_grid_coords[:,1])

    # 缩放目标形状
    target_hight = 2.2
    real_hight = y_max - y_min
    # real_hight = height
    h_scale = target_hight / real_hight
    grid_coords = h_scale * black_grid_coords
    print(grid_size * h_scale)

    # # 显示结果
    fig, ax = plt.subplots(figsize=(8, 8))

    # # 画出原始图像，获取坐标范围
    img = plt.imshow(binary_image, cmap='gray', origin='lower', aspect='equal')
    origin_extent = img.get_extent()
    img.remove()

    # # 画出最后图像
    new_extent = [origin_extent[0] - x_mean_grid, origin_extent[1] - x_mean_grid, origin_extent[2] - y_mean_grid, origin_extent[3] - y_mean_grid]
    # plt.imshow(binary_image, cmap='gray', extent=new_extent, origin='lower', aspect='equal', alpha=0.2)
    # plt.scatter(black_grid_coords[:, 0], black_grid_coords[:, 1], color='green', marker='o', alpha=0.8, label='Black Area Grids')
    plt.imshow(binary_image, cmap='gray', extent=[new_extent[0]*h_scale, new_extent[1]*h_scale, new_extent[2]*h_scale, new_extent[3]*h_scale], origin='lower', aspect='equal')
    plt.scatter(grid_coords[:, 0], grid_coords[:, 1], color='green', marker='o', alpha=0.8, label='Black Area Grids')

    plt.legend()
    plt.title('Grid Centers in Black Areas and Edge Grids')
    ax.relim()           # 重新计算当前轴的数据限制
    ax.autoscale_view()  # 自动调整坐标轴范围
    # plt.show()
    plt.close()

    shape_bound_points = np.array([new_extent[0]*h_scale, new_extent[1]*h_scale, new_extent[2]*h_scale, new_extent[3]*h_scale])

    # 将计算结果添加到结果字典中
    results["l_cell"].append(grid_size * h_scale)
    results["grid_coords"].append(grid_coords)
    results["binary_image"].append(binary_image)
    results["shape_bound_points"].append(shape_bound_points)

image_folder = '/home/zhugb/Pictures/fig/' 
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))

# 对每张图片运行处理函数
for image_path in image_paths:
    # image_path = '/home/zhugb/Pictures/fig/3.png'
    process_image(image_path)

# 将 results 保存到文件
results_file = os.path.join(image_folder, 'results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)


## ==================== User settings ===================='''
parser.add_argument("--n_a", type=int, default=50, help='number of agents') 
parser.add_argument("--is_boundary", type=bool, default=True, help='Set whether has wall or periodic boundaries') 
parser.add_argument("--is_con_self_state", type=bool, default=True, help="Whether contain myself state in the observation") 
parser.add_argument("--dynamics_mode", type=str, default='Cartesian', help=" select one from ['Cartesian', 'Polar']") 
parser.add_argument("--render-traj", type=bool, default=True, help=" whether render trajectories of agents") 
parser.add_argument("--traj_len", type=int, default=15, help="length of the trajectory")
parser.add_argument("--agent_strategy", type=str, default='input', help="the agent's strategy, please select one from ['input','random','rule']") 
# parser.add_argument("--grid_center_pos", type=type(grid_coords), default=grid_coords, help="an array containing all position of the grid centers")
# parser.add_argument("--shape_image", type=type(binary_image), default=binary_image, help="the image of the target shape")
# parser.add_argument("--shape_bound_points", type=type(shape_bound_points), default=shape_bound_points, help="the boundary points of the target shape")
parser.add_argument("--results_file", type=type(results_file), default=results_file, help="results_file")
# parser.add_argument("--l_cell", type=float, default=grid_size*h_scale, help="l_cell")
parser.add_argument("--video", type=bool, default=False, help="Record video")
## ==================== End of User settings ====================

## ==================== Training Parameters ====================
parser.add_argument("--env_name", default="formation", type=str)
parser.add_argument("--seed", default=226, type=int, help="Random seed")
parser.add_argument("--n_rollout_threads", default=1, type=int)
parser.add_argument("--n_training_threads", default=5, type=int)
parser.add_argument("--buffer_length", default=int(1.5e4), type=int) # 5e5
parser.add_argument("--n_episodes", default=4500, type=int) # 220
parser.add_argument("--episode_length", default=200, type=int)
parser.add_argument("--batch_size", default=512, type=int, help="Batch size for model training") # 256    
parser.add_argument("--hidden_dim", default=180, type=int)
parser.add_argument("--lr_actor", default=1e-4, type=float)
parser.add_argument("--lr_critic", default=1e-3, type=float)
parser.add_argument("--lr_discriminator", default=2e-3, type=float)
parser.add_argument("--disc_use_linear_lr_decay", default=True, type=bool)
parser.add_argument("--use_shaping_reward", default=True, type=bool)
parser.add_argument("--lambda_s", default=30, type=float, help="the coefficient of smoothness-inducing regularization")
parser.add_argument("--epsilon_p", default=0.03, type=float, help="the amptitude of state perturbation")
parser.add_argument("--epsilon", default=0.1, type=float) # 0.5
parser.add_argument("--noise_scale", default=0.9, type=float)
parser.add_argument("--tau", default=0.01, type=float)
parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'])
parser.add_argument("--save_interval", default=50, type=int, help="save data for every 'save_interval' episodes")
## ==================== Training Parameters ====================

gpsargs = parser.parse_args()
end_time = time.time()
print('load config parameters takes ', end_time - start_time)