import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy.spatial import distance_matrix

# 定义A字形区域的边界
def create_A_shape():
    vertices = np.array([[-1, 0], [0, 2], [1, 0], [0, -2], [-1, 0]])  # A字形区域顶点
    return Polygon(vertices, closed=True, edgecolor='black', facecolor='lightgray')

# 检查点是否在A字形区域内
def in_A_shape(point, polygon):
    return polygon.contains_point(point)

# 生成初始分布
def generate_initial_distribution(num_points, area_bounds):
    x = np.random.uniform(area_bounds[0], area_bounds[1], num_points)
    y = np.random.uniform(area_bounds[2], area_bounds[3], num_points)
    return np.stack((x, y), axis=1)

# 计算Wasserstein梯度流的每步更新
def update_points(points, target_points, learning_rate=0.1):
    cost_matrix = distance_matrix(points, target_points)  # 计算点与目标点之间的距离
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 使用匈牙利算法寻找最优匹配
    optimal_match = target_points[col_ind] - points[row_ind]  # 计算匹配的点与当前点的偏差
    points[row_ind] += learning_rate * optimal_match  # 沿着梯度方向更新位置
    return points

# 初始化参数
num_points = 100  # 粒子数目
area_bounds = [-1.5, 1.5, -2.5, 2.5]  # 区域边界
learning_rate = 0.05
num_iterations = 100  # 动画帧数

# 创建“A”字形的多边形区域
A_shape_polygon = create_A_shape()

# 生成初始分布
initial_points = generate_initial_distribution(num_points, area_bounds)

# 保证所有初始点都在“A”字形区域内
valid_points = []
for point in initial_points:
    if in_A_shape(point, A_shape_polygon):
        valid_points.append(point)
initial_points = np.array(valid_points)

# 创建目标分布，这里我们将目标分布等间隔放置在“A”字形区域内
theta = np.linspace(0, 2 * np.pi, num_points)
radius = np.random.uniform(0, 1, num_points)  # 使用不同的半径值，以便生成均匀分布
target_points = np.array([radius * np.cos(theta), radius * np.sin(theta)]).T

# 创建动画
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(A_shape_polygon)
scatter = ax.scatter(initial_points[:, 0], initial_points[:, 1], c='blue', s=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 2.5)

# 动画更新函数
def animate(i):
    global initial_points
    # 每帧更新粒子分布
    initial_points = update_points(initial_points, target_points, learning_rate)
    
    # 确保点保持在“A”字形区域内
    valid_points = [point for point in initial_points if in_A_shape(point, A_shape_polygon)]
    initial_points = np.array(valid_points)

    scatter.set_offsets(initial_points)  # 更新粒子位置
    return scatter,

# 使用FuncAnimation创建动图
ani = FuncAnimation(fig, animate, frames=num_iterations, interval=100, blit=True)

# 显示动图
plt.show()
