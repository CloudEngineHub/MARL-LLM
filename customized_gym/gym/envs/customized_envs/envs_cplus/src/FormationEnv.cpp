#include "FormationEnv.h"
#include <iostream>
#include <tuple>
#include <algorithm>
#include <vector>
#include <cmath> 
#include <numeric>
#include <typeinfo>
#include <random>

///////////////////////////////////////////////////////////////// observation form 1 /////////////////////////////////////////////////////////////////
// Function to calculate observation for each env
void _get_observation(double *p_input, 
                      double *dp_input, 
                      double *heading_input,
                      double *obs_input,  
                      double *boundary_pos_input, 
                      double *grid_center_input,
                      int *neighbor_index_input,
                      int *in_flags_input,
                      int *sensed_index_input,
                      int *occupied_index_input,
                      double d_sen,
                      double r_avoid,
                      double l_cell, 
                      int topo_nei_max, 
                      int num_obs_grid_max, 
                      int num_occupied_grid_max, 
                      int n_a, 
                      int n_g,
                      int obs_dim_agent, 
                      int dim, 
                      bool *condition) 
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_a));
    Matrix matrix_grid_center(dim, std::vector<double>(n_g));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<double> boundary_pos(4, 0.0);
    std::vector<int> in_flags(n_a, 0);
    std::vector<std::vector<int>> sensed_index(n_a, std::vector<int>(num_obs_grid_max, -1));
    std::vector<std::vector<int>> occupied_index(n_a, std::vector<int>(num_occupied_grid_max, -1));

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
            // std::cout << matrix_p[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_g; ++j) {
            matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
            // std::cout << n_g << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    for (int agent_i = 0; agent_i < n_a; ++agent_i) {
        // Calculate relative positions and velocities
        Matrix relPos_a2a(dim, std::vector<double>(n_a, 0.0));
        Matrix relVel_a2a(dim, std::vector<double>(n_a, 0.0));

        for (int j = 0; j < n_a; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_a2a[k][j] = matrix_p[k][j] - matrix_p[k][agent_i];
                if (condition[1]) {
                    relVel_a2a[k][j] = matrix_dp[k][j] - matrix_dp[k][agent_i];
                } else {
                    relVel_a2a[k][j] = matrix_heading[k][j] - matrix_heading[k][agent_i];
                }
            }
        }

        if (condition[0]) {
            _make_periodic(relPos_a2a, L, boundary_pos, true);
            // std::cout << L << std::endl;
        }

        // Obtain focused observations
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs = _get_focused(relPos_a2a, relVel_a2a, d_sen, topo_nei_max, true);
        Matrix relPos_a2a_focused = std::get<0>(focused_obs);
        Matrix relVel_a2a_focused = std::get<1>(focused_obs);
        std::vector<int> nei_index = std::get<2>(focused_obs);
        
        for (int i = 0; i < nei_index.size(); ++i) {
            neighbor_index[agent_i][i] = nei_index[i];
        }

        Matrix obs_agent;
        if (condition[2]) { // whether contain myself state in the observation
            Matrix obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_focused, 1);
            Matrix obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_focused, 1);
            obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);
        } else {
            obs_agent = _concatenate(relPos_a2a_focused, relVel_a2a_focused, 0);
        }

        // 将 obs_agent 转置，并展平为一维数组，然后赋值给 obs 的前部分
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        //////////////////////////////////////////////////// 获取目标状态 ////////////////////////////////////////////////////
        bool in_flag;
        std::vector<double> target_grid_pos(2), target_grid_vel(2);
        std::vector<int> sensed_indices;
        std::tie(in_flag, target_grid_pos, target_grid_vel, sensed_indices) = _get_target_grid_state(agent_i, matrix_p, matrix_dp, matrix_grid_center, l_cell, d_sen);

        in_flags[agent_i] = in_flag;
        // 相对位置和速度
        std::vector<double> target_grid_pos_rel = {target_grid_pos[0] - matrix_p[0][agent_i], target_grid_pos[1] - matrix_p[1][agent_i]};
        std::vector<double> target_grid_vel_rel = {target_grid_vel[0] - matrix_dp[0][agent_i], target_grid_vel[1] - matrix_dp[1][agent_i]};


        //////////////////////////////////////////////////// 删除被占据的网格 ////////////////////////////////////////////////////
        size_t num_sensed_grid_origin = sensed_indices.size();
        Matrix sensed_grid(2, std::vector<double>(num_sensed_grid_origin));

        std::vector<int> occupied_indices = sensed_indices;
        if (num_sensed_grid_origin > 0) {
            for (size_t i = 0; i < num_sensed_grid_origin; ++i) {
                sensed_grid[0][i] = matrix_grid_center[0][sensed_indices[i]];
                sensed_grid[1][i] = matrix_grid_center[1][sensed_indices[i]];
            }

            if (in_flags[agent_i] == 1) {
                // get the nearby agents
                std::vector<int> nearby_agents;
                std::vector<double> agent_pos_rel_norm(n_a, 0.0);
                for (size_t j = 0; j < n_a; ++j) {
                    // if (j != agent_i) {
                        double dx = matrix_p[0][j] - matrix_p[0][agent_i];
                        double dy = matrix_p[1][j] - matrix_p[1][agent_i];
                        agent_pos_rel_norm[j] = std::sqrt(dx * dx + dy * dy); // 计算范数
                    // }
                }

                for (size_t j = 0; j < agent_pos_rel_norm.size(); ++j) {
                    // if (j != agent_i && agent_pos_rel_norm[j] < (d_sen + r_avoid/2.0)) {
                    if (agent_pos_rel_norm[j] < (d_sen + r_avoid/2.0)) {
                        nearby_agents.push_back(j);  // 将索引 j 添加到 nearby_agents 中
                    }
                }
                
                for (int nearby_i : nearby_agents) {
                    // 计算邻居相对位置
                    Matrix grid_neigh_pos_relative(2, std::vector<double>(sensed_grid[0].size()));
                    for (size_t j = 0; j < sensed_grid[0].size(); ++j) {
                        grid_neigh_pos_relative[0][j] = sensed_grid[0][j] - matrix_p[0][nearby_i];
                        grid_neigh_pos_relative[1][j] = sensed_grid[1][j] - matrix_p[1][nearby_i];
                    }

                    // 计算邻居位置的范数
                    std::vector<double> grid_neigh_pos_relative_norm(grid_neigh_pos_relative[0].size());
                    std::vector<double> neigh_rel_j(2, 0.0);
                    for (size_t j = 0; j < grid_neigh_pos_relative[0].size(); ++j) {
                        neigh_rel_j = _extract_column_one(grid_neigh_pos_relative, j);
                        grid_neigh_pos_relative_norm[j] = _norm(neigh_rel_j);
                    }

                    // 根据范数过滤
                    std::vector<bool> mask(grid_neigh_pos_relative_norm.size(), false);
                    for (size_t j = 0; j < grid_neigh_pos_relative_norm.size(); ++j) {
                        mask[j] = grid_neigh_pos_relative_norm[j] > r_avoid/2.0;
                    }

                    // 筛选 sensed_grid
                    Matrix filtered_sensed_grid(2);
                    for (size_t j = 0; j < mask.size(); ++j) {
                        if (mask[j]) {
                            filtered_sensed_grid[0].push_back(sensed_grid[0][j]);
                            filtered_sensed_grid[1].push_back(sensed_grid[1][j]);
                        }
                    }
                    // 筛选 sensed_indices
                    std::vector<int> filtered_sensed_indices;
                    for (size_t j = 0; j < mask.size(); ++j) {
                        if (mask[j]) {
                            filtered_sensed_indices.push_back(sensed_indices[j]);
                        }
                    }
                    sensed_grid = filtered_sensed_grid; // 更新 sensed_grid
                    sensed_indices = filtered_sensed_indices; // 更新 sensed_indices
                }
            }
        }

        //////////////////////////////////////////////////// 从 occupied_indices 中移除 sensed_indices 中存在的元素 ////////////////////////////////////////////////////
        std::vector<int> temp_occupied_indices;
        for (int i = 0; i < occupied_indices.size(); ++i) {
            if (std::find(sensed_indices.begin(), sensed_indices.end(), occupied_indices[i]) == sensed_indices.end()) {
                temp_occupied_indices.push_back(occupied_indices[i]);
            }
        }
        occupied_indices = temp_occupied_indices;
        int num_occupied_grid = occupied_indices.size();
        if (num_occupied_grid > num_occupied_grid_max) {
            double step = static_cast<double>(num_occupied_grid - 1) / (num_occupied_grid_max - 1); // -1 是为了确保第一个和最后一个元素都选到
            // 均匀选取索引
            std::vector<int> final_indices;
            for (int i = 0; i < num_occupied_grid_max; ++i) {
                int index = static_cast<int>(std::round(i * step)); // 根据步长选取索引
                final_indices.push_back(occupied_indices[index]);
            }
            for (int i = 0; i < num_occupied_grid_max; ++i) {
                occupied_index[agent_i][i] = final_indices[i];
            }
        } else if (num_occupied_grid > 0 && num_occupied_grid <= num_occupied_grid_max) {
            for (int j = 0; j < num_occupied_grid; ++j) {
                occupied_index[agent_i][j] = occupied_indices[j];
            }
        }

        //////////////////////////////////////////////////// 获取未被占据的网格位置 //////////////////////////////////////////////////// 
        Matrix sensed_grid_pos;
        int num_sensed_grid = sensed_indices.size();
        if (num_sensed_grid > num_obs_grid_max) {
            Matrix sensed_grid_pos_1(2, std::vector<double>(num_obs_grid_max));
            // std::cout << 11111 << std::endl;
            // 计算步长
            double step = static_cast<double>(num_sensed_grid - 1) / (num_obs_grid_max - 1); // -1 是为了确保第一个和最后一个元素都选到
            // 均匀选取索引
            std::vector<int> final_indices;
            for (int i = 0; i < num_obs_grid_max; ++i) {
                int index = static_cast<int>(std::round(i * step)); // 根据步长选取索引
                final_indices.push_back(sensed_indices[index]);
            }
            for (size_t j = 0; j < num_obs_grid_max; ++j) {
                sensed_grid_pos_1[0][j] = matrix_grid_center[0][final_indices[j]];
                sensed_grid_pos_1[1][j] = matrix_grid_center[1][final_indices[j]];
            }
            sensed_grid_pos = sensed_grid_pos_1;

            for (int i = 0; i < num_obs_grid_max; ++i) {
                sensed_index[agent_i][i] = final_indices[i];
            }
        } else if (num_sensed_grid > 0 && num_sensed_grid <= num_obs_grid_max) {
            Matrix sensed_grid_pos_2(2, std::vector<double>(num_sensed_grid));
            for (size_t j = 0; j < num_sensed_grid; ++j) {
                sensed_grid_pos_2[0][j] = matrix_grid_center[0][sensed_indices[j]];
                sensed_grid_pos_2[1][j] = matrix_grid_center[1][sensed_indices[j]];
            }
            sensed_grid_pos = sensed_grid_pos_2;

            for (int i = 0; i < num_sensed_grid; ++i) {
                sensed_index[agent_i][i] = sensed_indices[i];
            }
        } else {
            Matrix sensed_grid_pos_3 = {}; // 空的二维向量
            sensed_grid_pos = sensed_grid_pos_3;
        }

        //////////////////////////////////////////////////// 初始化 sensed_grid_pos_rel，大小为 dim x num_obs_grid_max ////////////////////////////////////////////////////
        Matrix sensed_grid_pos_rel(dim, std::vector<double>(num_obs_grid_max, 0.0));
        // 如果 sensed_grid_pos 不是空的
        if (!sensed_grid_pos.empty()) {
            int num_obs_grid = sensed_grid_pos[0].size();  // 获取观测到的网格数量
            for (int j = 0; j < num_obs_grid; ++j) {
                sensed_grid_pos_rel[0][j] = sensed_grid_pos[0][j] - matrix_p[0][agent_i];
                sensed_grid_pos_rel[1][j] = sensed_grid_pos[1][j] - matrix_p[1][agent_i];
            }
        }
        // 将 sensed_grid_pos_rel 转置并展平为一维数组，然后赋值给 obs 的对应部分
        std::vector<double> sensed_grid_pos_rel_flat;
        sensed_grid_pos_rel_flat.reserve(sensed_grid_pos_rel.size() * sensed_grid_pos_rel[0].size());
        for (size_t j = 0; j < sensed_grid_pos_rel[0].size(); ++j) {
            for (size_t i = 0; i < sensed_grid_pos_rel.size(); ++i) {
                sensed_grid_pos_rel_flat.push_back(sensed_grid_pos_rel[i][j]);
            }
        }

        //////////////////////////////////////////////////// 根据 dynamics_mode 设置观测矩阵 ////////////////////////////////////////////////////
        if (condition[1]) {
            for (int j = 0; j < obs_dim_agent - (2 + num_obs_grid_max) * dim; ++j) {
                matrix_obs[j][agent_i] = obs_agent_flat[j];
            }
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - (2 + num_obs_grid_max) * dim + j][agent_i] = target_grid_pos_rel[j];
            }
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - (1 + num_obs_grid_max) * dim + j][agent_i] = target_grid_vel_rel[j];
            }
            for (int j = 0; j < num_obs_grid_max * dim; ++j) {
                matrix_obs[obs_dim_agent - num_obs_grid_max * dim + j][agent_i] = sensed_grid_pos_rel_flat[j];
            }
        } else {
            for (int j = 0; j < obs_dim_agent - 3 * dim; ++j) {
                matrix_obs[j][agent_i] = obs_agent_flat[j];
            }
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - 3 * dim + j][agent_i] = target_grid_pos_rel[j];
            }
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = target_grid_vel_rel[j];
            }
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - dim + j][agent_i] = matrix_heading[j][agent_i];
            }
        }

    }

    for (int i = 0; i < obs_dim_agent; ++i) {
        for (int j = 0; j < n_a; ++j) {
            obs_input[i * n_a + j] = matrix_obs[i][j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index_input[i * topo_nei_max + j] = neighbor_index[i][j];
            // std::cout << neighbor_index[i][j] << "\t";
        }
    }

    for (int i = 0; i < n_a; ++i) {
        in_flags_input[i] = in_flags[i];
        // std::cout << neighbor_index[i][j] << "\t";
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_obs_grid_max; ++j) {
            sensed_index_input[i * num_obs_grid_max + j] = sensed_index[i][j];
            // std::cout << neighbor_index[i][j] << "\t";
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_occupied_grid_max; ++j) {
            occupied_index_input[i * num_occupied_grid_max + j] = occupied_index[i][j];
            // std::cout << neighbor_index[i][j] << "\t";
        }
    }
}

///////////////////////////////////////////////////////////////// observation form 2 /////////////////////////////////////////////////////////////////
// // Function to calculate observation for each env
// void _get_observation(double *p_input, 
//                       double *dp_input, 
//                       double *heading_input,
//                       double *obs_input,  
//                       double *boundary_pos_input, 
//                       double *grid_center_input,
//                       int *neighbor_index_input,
//                       int *in_flags_input,
//                       int *sensed_index_input,
//                       int *occupied_index_input,
//                       double d_sen,
//                       double r_avoid,
//                       double l_cell, 
//                       int topo_nei_max, 
//                       int num_obs_grid_max, 
//                       int num_occupied_grid_max, 
//                       int n_a, 
//                       int n_g,
//                       int obs_dim_agent, 
//                       int dim, 
//                       bool *condition) 
// {
//     Matrix matrix_p(dim, std::vector<double>(n_a));
//     Matrix matrix_dp(dim, std::vector<double>(n_a));
//     Matrix matrix_heading(dim, std::vector<double>(n_a));
//     Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_a));
//     Matrix matrix_grid_center(dim, std::vector<double>(n_g));
//     std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
//     std::vector<double> boundary_pos(4, 0.0);
//     std::vector<int> in_flags(n_a, 0);
//     std::vector<std::vector<int>> sensed_index(n_a, std::vector<int>(num_obs_grid_max, -1));
//     std::vector<std::vector<int>> occupied_index(n_a, std::vector<int>(num_occupied_grid_max, -1));

//     for (int i = 0; i < dim; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             matrix_p[i][j] = p_input[i * n_a + j];
//             matrix_dp[i][j] = dp_input[i * n_a + j];
//             matrix_heading[i][j] = heading_input[i * n_a + j];
//             // std::cout << matrix_p[i][j] << " ";
//         }
//         // std::cout << std::endl;
//     }

//     for (int i = 0; i < dim; ++i) {
//         for (int j = 0; j < n_g; ++j) {
//             matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
//             // std::cout << n_g << " ";
//         }
//         // std::cout << std::endl;
//     }

//     for (int i = 0; i < 4; ++i) {
//         boundary_pos[i] = boundary_pos_input[i];
//     }

//     double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;
//     for (int agent_i = 0; agent_i < n_a; ++agent_i) {
//         // Calculate relative positions and velocities
//         Matrix relPos_a2a(dim, std::vector<double>(n_a, 0.0));
//         Matrix relVel_a2a(dim, std::vector<double>(n_a, 0.0));

//         for (int j = 0; j < n_a; ++j) {
//             for (int k = 0; k < dim; ++k) {
//                 relPos_a2a[k][j] = matrix_p[k][j] - matrix_p[k][agent_i];
//                 if (condition[1]) {
//                     relVel_a2a[k][j] = matrix_dp[k][j] - matrix_dp[k][agent_i];
//                 } else {
//                     relVel_a2a[k][j] = matrix_heading[k][j] - matrix_heading[k][agent_i];
//                 }
//             }
//         }

//         if (condition[0]) {
//             _make_periodic(relPos_a2a, L, boundary_pos, true);
//             // std::cout << L << std::endl;
//         }

//         // Obtain focused observations
//         std::tuple<Matrix, Matrix, std::vector<int>> focused_obs = _get_focused(relPos_a2a, relVel_a2a, d_sen, topo_nei_max, true);
//         Matrix relPos_a2a_focused = std::get<0>(focused_obs);
//         Matrix relVel_a2a_focused = std::get<1>(focused_obs);
//         std::vector<int> nei_index = std::get<2>(focused_obs);
        
//         for (int i = 0; i < nei_index.size(); ++i) {
//             neighbor_index[agent_i][i] = nei_index[i];
//         }

//         Matrix obs_agent;
//         if (condition[2]) { // whether contain myself state in the observation
//             Matrix obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_focused, 1);
//             Matrix obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_focused, 1);
//             obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);
//         } else {
//             obs_agent = _concatenate(relPos_a2a_focused, relVel_a2a_focused, 0);
//         }

//         // 将 obs_agent 转置，并展平为一维数组，然后赋值给 obs 的前部分
//         std::vector<double> obs_agent_flat;
//         obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
//         for (size_t j = 0; j < obs_agent[0].size(); ++j) {
//             for (size_t i = 0; i < obs_agent.size(); ++i) {
//                 obs_agent_flat.push_back(obs_agent[i][j]);
//             }
//         }

//         //////////////////////////////////////////////////// 获取目标状态 ////////////////////////////////////////////////////
//         bool in_flag;
//         std::vector<double> target_grid_pos(2), target_grid_vel(2);
//         std::vector<int> sensed_indices;
//         std::tie(in_flag, target_grid_pos, target_grid_vel, sensed_indices) = _get_target_grid_state(agent_i, matrix_p, matrix_dp, matrix_grid_center, l_cell, d_sen);

//         in_flags[agent_i] = in_flag;
//         // 相对位置和速度
//         std::vector<double> target_grid_pos_rel = {target_grid_pos[0] - matrix_p[0][agent_i], target_grid_pos[1] - matrix_p[1][agent_i]};
//         std::vector<double> target_grid_vel_rel = {target_grid_vel[0] - matrix_dp[0][agent_i], target_grid_vel[1] - matrix_dp[1][agent_i]};

//         //////////////////////////////////////////////////// 删除被占据的网格 ////////////////////////////////////////////////////
//         size_t num_sensed_grid_origin = sensed_indices.size();
//         std::vector<int> occupied_indices = sensed_indices;
//         // std::vector<double> num_grid_in_voronoi(nei_index.size() + 1, 0.0); // !!!!!!!!!!!!!!!!!!!!
//         if (num_sensed_grid_origin > 0) {
//             if (in_flags[agent_i] == 1) {
//                 // get the nearby agents
//                 nei_index.insert(nei_index.begin(), agent_i);
//                 std::vector<int> index_grid_in_voronoi;
//                 // std::vector<std::vector<int>> index_grid_in_voronoi(nei_index.size()); // !!!!!!!!!!!!!!!!!!!!

//                 // 遍历每个 sensed_indices 中的 cell_index
//                 for (int cell_index : sensed_indices) {
//                     // 计算当前网格相对于所有邻居的相对位置
//                     Matrix rel_pos_cell_nei(2, std::vector<double>(nei_index.size()));
//                     for (size_t j = 0; j < nei_index.size(); ++j) {
//                         rel_pos_cell_nei[0][j] = matrix_p[0][nei_index[j]] - matrix_grid_center[0][cell_index];
//                         rel_pos_cell_nei[1][j] = matrix_p[1][nei_index[j]] - matrix_grid_center[1][cell_index];
//                     }

//                     // 计算每个相对位置的范数
//                     std::vector<double> rel_pos_cell_nei_norm(nei_index.size(), 0.0);
//                     for (size_t j = 0; j < nei_index.size(); ++j) {
//                         std::vector<double> neigh_rel_j = _extract_column_one(rel_pos_cell_nei, j);
//                         rel_pos_cell_nei_norm[j] = _norm(neigh_rel_j);
//                     }

//                     // 找到范数最小的索引
//                     int min_index = min_element(rel_pos_cell_nei_norm.begin(), rel_pos_cell_nei_norm.end()) - rel_pos_cell_nei_norm.begin();

//                     // num_grid_in_voronoi[min_index] += 1.0; // !!!!!!!!!!!!!!!!!!!!

//                     // 如果是最小的索引，则将其添加到 index_grid_in_voronoi
//                     if (min_index == 0) {
//                         index_grid_in_voronoi.push_back(cell_index);
//                         // index_grid_in_voronoi[min_index].push_back(cell_index); // !!!!!!!!!!!!!!!!!!!!
//                     }
//                 }

//                 // 更新 sensed_indices
//                 sensed_indices = index_grid_in_voronoi;
//                 // sensed_indices = index_grid_in_voronoi[0]; // !!!!!!!!!!!!!!!!!!!!

//                 // 归一化 num_grid_in_voronoi // !!!!!!!!!!!!!!!!!!!!
//                 // for (double& value : num_grid_in_voronoi) {
//                 //     value /= static_cast<double>(num_sensed_grid_origin);
//                 // }

//             }
//         }

//         // num_grid_in_voronoi.resize(topo_nei_max + 1, 0.0);   // !!!!!!!!!!!!!!!!!!!!

//         //////////////////////////////////////////////////// 从 occupied_indices 中移除 sensed_indices 中存在的元素 ////////////////////////////////////////////////////
//         std::vector<int> temp_occupied_indices;
//         for (int i = 0; i < occupied_indices.size(); ++i) {
//             if (std::find(sensed_indices.begin(), sensed_indices.end(), occupied_indices[i]) == sensed_indices.end()) {
//                 temp_occupied_indices.push_back(occupied_indices[i]);
//             }
//         }
//         occupied_indices = temp_occupied_indices;
//         int num_occupied_grid = occupied_indices.size();
//         if (num_occupied_grid > num_occupied_grid_max) {
//             double step = static_cast<double>(num_occupied_grid - 1) / (num_occupied_grid_max - 1); // -1 是为了确保第一个和最后一个元素都选到
//             // 均匀选取索引
//             std::vector<int> final_indices;
//             for (int i = 0; i < num_occupied_grid_max; ++i) {
//                 int index = static_cast<int>(std::round(i * step)); // 根据步长选取索引
//                 final_indices.push_back(occupied_indices[index]);
//             }
//             for (int i = 0; i < num_occupied_grid_max; ++i) {
//                 occupied_index[agent_i][i] = final_indices[i];
//             }
//         } else if (num_occupied_grid > 0 && num_occupied_grid <= num_occupied_grid_max) {
//             for (int j = 0; j < num_occupied_grid; ++j) {
//                 occupied_index[agent_i][j] = occupied_indices[j];
//             }
//         }

//         //////////////////////////////////////////////////// 获取未被占据的网格位置 //////////////////////////////////////////////////// 
//         Matrix sensed_grid_pos;
//         int num_sensed_grid = sensed_indices.size();
//         if (num_sensed_grid > num_obs_grid_max) {
//             Matrix sensed_grid_pos_1(2, std::vector<double>(num_obs_grid_max));
//             // std::cout << 11111 << std::endl;
//             // 计算步长
//             double step = static_cast<double>(num_sensed_grid - 1) / (num_obs_grid_max - 1); // -1 是为了确保第一个和最后一个元素都选到
//             // 均匀选取索引
//             std::vector<int> final_indices;
//             for (int i = 0; i < num_obs_grid_max; ++i) {
//                 int index = static_cast<int>(std::round(i * step)); // 根据步长选取索引
//                 final_indices.push_back(sensed_indices[index]);
//             }
//             for (size_t j = 0; j < num_obs_grid_max; ++j) {
//                 sensed_grid_pos_1[0][j] = matrix_grid_center[0][final_indices[j]];
//                 sensed_grid_pos_1[1][j] = matrix_grid_center[1][final_indices[j]];
//             }
//             sensed_grid_pos = sensed_grid_pos_1;

//             for (int i = 0; i < num_obs_grid_max; ++i) {
//                 sensed_index[agent_i][i] = final_indices[i];
//             }
//         } else if (num_sensed_grid > 0 && num_sensed_grid <= num_obs_grid_max) {
//             Matrix sensed_grid_pos_2(2, std::vector<double>(num_sensed_grid));
//             for (size_t j = 0; j < num_sensed_grid; ++j) {
//                 sensed_grid_pos_2[0][j] = matrix_grid_center[0][sensed_indices[j]];
//                 sensed_grid_pos_2[1][j] = matrix_grid_center[1][sensed_indices[j]];
//             }
//             sensed_grid_pos = sensed_grid_pos_2;

//             for (int i = 0; i < num_sensed_grid; ++i) {
//                 sensed_index[agent_i][i] = sensed_indices[i];
//             }
//         } else {
//             Matrix sensed_grid_pos_3 = {}; // 空的二维向量
//             sensed_grid_pos = sensed_grid_pos_3;
//         }

//         //////////////////////////////////////////////////// 初始化 sensed_grid_pos_rel，大小为 dim x num_obs_grid_max ////////////////////////////////////////////////////
//         Matrix sensed_grid_pos_rel(dim, std::vector<double>(num_obs_grid_max, 0.0));
//         // 如果 sensed_grid_pos 不是空的
//         if (!sensed_grid_pos.empty()) {
//             int num_obs_grid = sensed_grid_pos[0].size();  // 获取观测到的网格数量
//             for (int j = 0; j < num_obs_grid; ++j) {
//                 sensed_grid_pos_rel[0][j] = sensed_grid_pos[0][j] - matrix_p[0][agent_i];
//                 sensed_grid_pos_rel[1][j] = sensed_grid_pos[1][j] - matrix_p[1][agent_i];
//             }
//         }
//         // 将 sensed_grid_pos_rel 转置并展平为一维数组，然后赋值给 obs 的对应部分
//         std::vector<double> sensed_grid_pos_rel_flat;
//         sensed_grid_pos_rel_flat.reserve(sensed_grid_pos_rel.size() * sensed_grid_pos_rel[0].size());
//         for (size_t j = 0; j < sensed_grid_pos_rel[0].size(); ++j) {
//             for (size_t i = 0; i < sensed_grid_pos_rel.size(); ++i) {
//                 sensed_grid_pos_rel_flat.push_back(sensed_grid_pos_rel[i][j]);
//             }
//         }

//         //////////////////////////////////////////////////// 根据 dynamics_mode 设置观测矩阵 ////////////////////////////////////////////////////
//         int index_1 = (2 + num_obs_grid_max) * dim;
//         int index_2 = num_obs_grid_max * dim;
//         // int index_1 = (2 + num_obs_grid_max) * dim + topo_nei_max + 1; // !!!!!!!!!!!!!!!!!!!!
//         // int index_2 = num_obs_grid_max * dim + topo_nei_max + 1;
//         // int index_3 = topo_nei_max + 1;
//         if (condition[1]) {
//             for (int j = 0; j < obs_dim_agent - index_1; ++j) {
//                 matrix_obs[j][agent_i] = obs_agent_flat[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - index_1 + j][agent_i] = target_grid_pos_rel[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - index_1 + dim + j][agent_i] = target_grid_vel_rel[j];
//             }
//             for (int j = 0; j < num_obs_grid_max * dim; ++j) {
//                 matrix_obs[obs_dim_agent - index_2 + j][agent_i] = sensed_grid_pos_rel_flat[j];
//             }
//             // for (int j = 0; j < num_grid_in_voronoi.size(); ++j) {
//             //     matrix_obs[obs_dim_agent - index_3 + j][agent_i] = num_grid_in_voronoi[j]; // !!!!!!!!!!!!!!!!!!!!
//             // }
//         }

//     }

//     for (int i = 0; i < obs_dim_agent; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             obs_input[i * n_a + j] = matrix_obs[i][j];
//         }
//     }

//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < topo_nei_max; ++j) {
//             neighbor_index_input[i * topo_nei_max + j] = neighbor_index[i][j];
//             // std::cout << neighbor_index[i][j] << "\t";
//         }
//     }

//     for (int i = 0; i < n_a; ++i) {
//         in_flags_input[i] = in_flags[i];
//         // std::cout << neighbor_index[i][j] << "\t";
//     }

//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < num_obs_grid_max; ++j) {
//             sensed_index_input[i * num_obs_grid_max + j] = sensed_index[i][j];
//             // std::cout << neighbor_index[i][j] << "\t";
//         }
//     }

//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < num_occupied_grid_max; ++j) {
//             occupied_index_input[i * num_occupied_grid_max + j] = occupied_index[i][j];
//             // std::cout << neighbor_index[i][j] << "\t";
//         }
//     }
// }

///////////////////////////////////////////////////////////////// observation form 3 /////////////////////////////////////////////////////////////////
// // Function to calculate observation for each env
// void _get_observation(double *p_input, 
//                       double *dp_input, 
//                       double *heading_input,
//                       double *obs_input,  
//                       double *boundary_pos_input, 
//                       double *grid_center_input,
//                       int *neighbor_index_input,
//                       int *in_flags_input,
//                       double d_sen,
//                       double r_avoid,
//                       double l_cell, 
//                       int topo_nei_max, 
//                       int n_a, 
//                       int n_g,
//                       int obs_dim_agent, 
//                       int dim, 
//                       bool *condition) 
// {
//     Matrix matrix_p(dim, std::vector<double>(n_a));
//     Matrix matrix_dp(dim, std::vector<double>(n_a));
//     Matrix matrix_heading(dim, std::vector<double>(n_a));
//     Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_a));
//     Matrix matrix_grid_center(dim, std::vector<double>(n_g));
//     std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
//     std::vector<double> boundary_pos(4, 0.0);
//     std::vector<int> in_flags(n_a, 0);

//     for (int i = 0; i < dim; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             matrix_p[i][j] = p_input[i * n_a + j];
//             matrix_dp[i][j] = dp_input[i * n_a + j];
//             matrix_heading[i][j] = heading_input[i * n_a + j];
//             // std::cout << matrix_p[i][j] << " ";
//         }
//         // std::cout << std::endl;
//     }

//     for (int i = 0; i < dim; ++i) {
//         for (int j = 0; j < n_g; ++j) {
//             matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
//             // std::cout << n_g << " ";
//         }
//         // std::cout << std::endl;
//     }

//     for (int i = 0; i < 4; ++i) {
//         boundary_pos[i] = boundary_pos_input[i];
//     }

//     double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;
//     for (int agent_i = 0; agent_i < n_a; ++agent_i) {
//         // Calculate relative positions and velocities
//         Matrix relPos_a2a(dim, std::vector<double>(n_a, 0.0));
//         Matrix relVel_a2a(dim, std::vector<double>(n_a, 0.0));

//         for (int j = 0; j < n_a; ++j) {
//             for (int k = 0; k < dim; ++k) {
//                 relPos_a2a[k][j] = matrix_p[k][j] - matrix_p[k][agent_i];
//                 if (condition[1]) {
//                     relVel_a2a[k][j] = matrix_dp[k][j] - matrix_dp[k][agent_i];
//                 } else {
//                     relVel_a2a[k][j] = matrix_heading[k][j] - matrix_heading[k][agent_i];
//                 }
//             }
//         }

//         if (condition[0]) {
//             _make_periodic(relPos_a2a, L, boundary_pos, true);
//             // std::cout << L << std::endl;
//         }

//         // Obtain focused observations
//         std::tuple<Matrix, Matrix, std::vector<int>> focused_obs = _get_focused(relPos_a2a, relVel_a2a, d_sen, topo_nei_max, true);
//         Matrix relPos_a2a_focused = std::get<0>(focused_obs);
//         Matrix relVel_a2a_focused = std::get<1>(focused_obs);
//         std::vector<int> nei_index = std::get<2>(focused_obs);
        
//         for (int i = 0; i < nei_index.size(); ++i) {
//             neighbor_index[agent_i][i] = nei_index[i];
//         }

//         Matrix obs_agent;
//         if (condition[2]) { // whether contain myself state in the observation
//             Matrix obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_focused, 1);
//             Matrix obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_focused, 1);
//             obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);
//         } else {
//             obs_agent = _concatenate(relPos_a2a_focused, relVel_a2a_focused, 0);
//         }

//         // 将 obs_agent 转置，并展平为一维数组，然后赋值给 obs 的前部分
//         std::vector<double> obs_agent_flat;
//         obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
//         for (size_t j = 0; j < obs_agent[0].size(); ++j) {
//             for (size_t i = 0; i < obs_agent.size(); ++i) {
//                 obs_agent_flat.push_back(obs_agent[i][j]);
//             }
//         }

//         //////////////////////////////////////////////////// 获取目标状态 ////////////////////////////////////////////////////
//         bool in_flag;
//         std::vector<double> target_grid_pos(2), target_grid_vel(2);
//         std::vector<int> sensed_indices;
//         std::tie(in_flag, target_grid_pos, target_grid_vel, sensed_indices) = _get_target_grid_state(agent_i, matrix_p, matrix_dp, matrix_grid_center, l_cell, d_sen);

//         in_flags[agent_i] = in_flag;
//         // 相对位置和速度
//         std::vector<double> target_grid_pos_rel = {target_grid_pos[0] - matrix_p[0][agent_i], target_grid_pos[1] - matrix_p[1][agent_i]};
//         std::vector<double> target_grid_vel_rel = {target_grid_vel[0] - matrix_dp[0][agent_i], target_grid_vel[1] - matrix_dp[1][agent_i]};

//         //////////////////////////////////////////////////// 根据 dynamics_mode 设置观测矩阵 ////////////////////////////////////////////////////
//         if (condition[1]) {
//             for (int j = 0; j < obs_dim_agent - 2 * dim; ++j) {
//                 matrix_obs[j][agent_i] = obs_agent_flat[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = target_grid_pos_rel[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - dim + j][agent_i] = target_grid_vel_rel[j];
//             }
//         } else {
//             for (int j = 0; j < obs_dim_agent - 3 * dim; ++j) {
//                 matrix_obs[j][agent_i] = obs_agent_flat[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - 3 * dim + j][agent_i] = target_grid_pos_rel[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = target_grid_vel_rel[j];
//             }
//             for (int j = 0; j < dim; ++j) {
//                 matrix_obs[obs_dim_agent - dim + j][agent_i] = matrix_heading[j][agent_i];
//             }
//         }

//     }

//     for (int i = 0; i < obs_dim_agent; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             obs_input[i * n_a + j] = matrix_obs[i][j];
//         }
//     }

//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < topo_nei_max; ++j) {
//             neighbor_index_input[i * topo_nei_max + j] = neighbor_index[i][j];
//             // std::cout << neighbor_index[i][j] << "\t";
//         }
//     }

//     for (int i = 0; i < n_a; ++i) {
//         in_flags_input[i] = in_flags[i];
//         // std::cout << neighbor_index[i][j] << "\t";
//     }
// }

///////////////////////////////////////////////////////////////// reward form 1 /////////////////////////////////////////////////////////////////
void _get_reward(double *p_input, 
                 double *dp_input,
                 double *heading_input, 
                 double *act_input, 
                 double *reward_input, 
                 double *boundary_pos_input, 
                 double *grid_center_input,
                 double *num_in_voronoi_last_input,
                 int *neighbor_index_input,
                 int *in_flags_input,
                 int *sensed_index_input,
                 int *occupied_index_input,
                 double d_sen, 
                 double r_avoid, 
                 int topo_nei_max, 
                 int num_obs_grid_max,
                 int num_occupied_grid_max,
                 int n_a, 
                 int n_g,
                 int dim, 
                 bool *condition, 
                 bool *is_collide_b2b_input, 
                 bool *is_collide_b2w_input, 
                 double *coefficients) 
{
    // std::cout <<  d_sen << typeid(d_sen).name() << ":" << r_avoid << typeid(r_avoid).name() << std::endl;
    
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_act(dim, std::vector<double>(n_a));
    Matrix matrix_grid_center(dim, std::vector<double>(n_g));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<std::vector<int>> sensed_index(n_a, std::vector<int>(num_obs_grid_max, -1));
    std::vector<std::vector<int>> occupied_index(n_a, std::vector<int>(num_occupied_grid_max, -1));
    std::vector<int> in_flags(n_a, 0);
    std::vector<double> boundary_pos(4, 0.0);
    std::vector<double> num_in_voronoi_last(n_a, 0.0);
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<std::vector<bool>> is_collide_b2w(4, std::vector<bool>(n_a, false));

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
            // std::cout << matrix_p[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int j = 0; j < n_a; ++j) {
        for (int i = 0; i < dim; ++i) {
            matrix_act[i][j] = act_input[j * dim + i];
            // std::cout << act_input[j * dim + i] << " ";
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_g; ++j) {
            matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
            // std::cout << n_g << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index[i][j] = neighbor_index_input[i * topo_nei_max + j];
            // std::cout << neighbor_index[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < n_a; ++i) {
        in_flags[i] = in_flags_input[i];
        num_in_voronoi_last[i] = num_in_voronoi_last_input[i];
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_obs_grid_max; ++j) {
            sensed_index[i][j] = sensed_index_input[i * num_obs_grid_max + j];
            // std::cout << sensed_index[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_occupied_grid_max; ++j) {
            occupied_index[i][j] = occupied_index_input[i * num_occupied_grid_max + j];
            // std::cout << neighbor_index[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2w[i][j] = is_collide_b2w_input[i * n_a + j];
            // std::cout << is_collide_b2w[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // Initialize reward_a matrix
    std::vector<double> reward_a(n_a, 0.0);

    // bool is_all_in_shape = false;
    // bool is_has_collision = false;
    // bool contains_zero = false;
    std::vector<bool> is_has_collisions(n_a, false);
    std::vector<bool> is_uniforms(n_a, false);

    // shape-entering reward
    // if (condition[2]) {
    //     std::vector<double> flags(n_a);

    //     for (size_t i = 0; i < n_a; ++i) {
    //         flags[i] = coefficients[0] * (static_cast<double>(in_flags[i]) - 1.0);
    //     }

    //     for (size_t i = 0; i < n_a; ++i) {
    //         reward_a[i] += flags[i];
    //     }
    //     // for (size_t i = 0; i < n_a; ++i) {
    //     //     if (in_flags[i] == 0) {
    //     //         contains_zero = true;
    //     //         break; // 一旦找到一个 0，就可以退出循环
    //     //     }
    //     // }

    //     // bool is_all_in_shape = !contains_zero;
        
    // }

    // interaction reward
    if (condition[3]) {
        for (int agent = 0; agent < n_a; ++agent) {

            std::vector<int> list_nei;
            for (int i = 0; i < neighbor_index[agent].size(); ++i) {
                if (neighbor_index[agent][i] != -1) {
                    list_nei.push_back(neighbor_index[agent][i]);
                    // std::cout << neighbor_index[agent][i] << std::endl;
                }
            }

            std::vector<double> pos_rel(2, 0.0);
            std::vector<double> avg_neigh_vel(2, 0.0);

            // bool is_has_collision = false;

            if (!list_nei.empty()) {
                for (int agent2 : list_nei) {
                    if (condition[0]) {
                        Matrix pos_rel_mat = {{matrix_p[0][agent2] - matrix_p[0][agent]}, {matrix_p[1][agent2] - matrix_p[1][agent]}};
                        _make_periodic(pos_rel_mat, L, boundary_pos, true);
                        pos_rel = _matrix_to_vector(pos_rel_mat);
                    } else {
                        pos_rel = {matrix_p[0][agent2] - matrix_p[0][agent], matrix_p[1][agent2] - matrix_p[1][agent]};
                    }

                    if (r_avoid > _norm(pos_rel)) {
                        // reward_a[agent] -= coefficients[1];
                        is_has_collisions[agent] = true;
                        break;
                    }
                    // reward_a[agent] -= std::max(coefficients[1] * (r_avoid - _norm(pos_rel)), 0.0);
                    // std::cout << _norm(pos_rel) << ": " << agent << std::endl;
                    std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent2);
                    avg_neigh_vel[0] += matrix_dp[0][agent2];
                    avg_neigh_vel[1] += matrix_dp[1][agent2];
                    
                }
                
                avg_neigh_vel[0] /= list_nei.size();
                avg_neigh_vel[1] /= list_nei.size();
                // std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent);
                // double norm_dp_agent = _norm(dp_agent) + 1E-8;
                double vel_diff_norm = std::sqrt(std::pow(avg_neigh_vel[0] - matrix_dp[0][agent], 2) + std::pow(avg_neigh_vel[1] - matrix_dp[1][agent], 2));
                // reward_a[agent] -= coefficients[2] * vel_diff_norm;
                if (vel_diff_norm > 0.2) {
                    is_has_collisions[agent] = true;
                }

            }

            // if (in_flags[agent] == 1 && is_has_collisions[agent] == false) {
            //     reward_a[agent] += 1.0;
            // }
        }
    }

    // if (is_all_in_shape == true && is_has_collision == false) {
    //     for (int agent_i = 0; agent_i < n_a; ++agent_i) {
    //         reward_a[agent_i] += 1.0;
    //     }
    // }

    // exploration reward
    if (condition[4]) {
        // std::cout << 12 << std::endl;
        for (int agent_i = 0; agent_i < n_a; ++agent_i) {
            if (in_flags[agent_i] == 1) {
                //////////////////////////////////////////////// reward form 1 ////////////////////////////////////////////////
                // // 获取邻居
                // std::vector<int> list_nei;
                // for (int i = 0; i < neighbor_index[agent_i].size(); ++i) {
                //     if (neighbor_index[agent_i][i] != -1) {
                //         list_nei.push_back(neighbor_index[agent_i][i]);
                //         // std::cout << neighbor_index[agent_i][i] << std::endl;
                //     }
                // }

                // // 获取感知到的网格
                // std::vector<int> list_grid;
                // for (int i = 0; i < sensed_index[agent_i].size(); ++i) {
                //     if (sensed_index[agent_i][i] != -1) {
                //         list_grid.push_back(sensed_index[agent_i][i]);
                //     }
                // }

                // if (!list_nei.empty()) {
                //     double num_list_grid_norm = static_cast<double>(list_grid.size()) / num_obs_grid_max;

                //     double num_list_grid_neigh_avg = 0.0;
                //     for (const auto& neigh_i : list_nei) {
                //         // 获取感知到的网格
                //         std::vector<int> list_grid_neigh = sensed_index[neigh_i];
                        
                //         // 移除 -1 元素
                //         list_grid_neigh.erase(
                //             std::remove(list_grid_neigh.begin(), list_grid_neigh.end(), -1), 
                //             list_grid_neigh.end()
                //         );

                //         // 计算平均邻居网格数量
                //         num_list_grid_neigh_avg += static_cast<double>(list_grid_neigh.size()) / num_obs_grid_max;
                //     }

                //     num_list_grid_neigh_avg /= static_cast<double>(list_nei.size());
                //     reward_a[agent_i] -= coefficients[3] * std::abs(num_list_grid_norm - num_list_grid_neigh_avg);
                // }

                //////////////////////////////////////////////// reward form 2 ////////////////////////////////////////////////
                // // 获取感知到的网格
                // std::vector<int> list_grid;
                // for (int i = 0; i < sensed_index[agent_i].size(); ++i) {
                //     if (sensed_index[agent_i][i] != -1) {
                //         list_grid.push_back(sensed_index[agent_i][i]);
                //     }
                // }

                // // 获取占据的网格
                // std::vector<int> list_occupied_grid;
                // for (int i = 0; i < occupied_index[agent_i].size(); ++i) {
                //     if (occupied_index[agent_i][i] != -1) {
                //         list_occupied_grid.push_back(occupied_index[agent_i][i]);
                //     }
                // }

                // list_grid.insert(list_grid.end(), list_occupied_grid.begin(), list_occupied_grid.end());

                // Matrix sensed_grid_pos;
                // Matrix occupied_grid_pos;

                // // 获取感知网格和占据网格的位置
                // for (int grid_idx : list_grid) {
                //     sensed_grid_pos.push_back({matrix_grid_center[0][grid_idx], matrix_grid_center[1][grid_idx]});
                // }
                // for (int occupied_idx : list_occupied_grid) {
                //     occupied_grid_pos.push_back({matrix_grid_center[0][occupied_idx], matrix_grid_center[1][occupied_idx]});
                // }

                // int num_list_grid = list_grid.size();
                // int num_list_occupied_grid = list_occupied_grid.size();
                // if (num_list_grid > 0 && num_list_occupied_grid > 0) {
                //     // 计算感知到的网格和占据网格的位置均值
                //     std::vector<double> mean_sensed_grid_pos = {0.0, 0.0};
                //     std::vector<double> mean_occupied_grid_pos = {0.0, 0.0};

                //     for (const auto& pos : sensed_grid_pos) {
                //         mean_sensed_grid_pos[0] += pos[0];
                //         mean_sensed_grid_pos[1] += pos[1];
                //     }
                //     for (const auto& pos : occupied_grid_pos) {
                //         mean_occupied_grid_pos[0] += pos[0];
                //         mean_occupied_grid_pos[1] += pos[1];
                //     }

                //     mean_sensed_grid_pos[0] /= static_cast<double>(num_list_grid);
                //     mean_sensed_grid_pos[1] /= static_cast<double>(num_list_grid);
                    
                //     mean_occupied_grid_pos[0] /= static_cast<double>(num_list_occupied_grid);
                //     mean_occupied_grid_pos[1] /= static_cast<double>(num_list_occupied_grid);

                //     // 计算 reward
                //     std::vector<double> mean_pos_rel = {mean_sensed_grid_pos[0] - mean_occupied_grid_pos[0], mean_sensed_grid_pos[1] - mean_occupied_grid_pos[1]};
                //     double mean_pos_rel_norm = _norm(mean_pos_rel);
                //     // reward_a[agent_i] -= coefficients[3] * mean_pos_rel_norm;
                //     // std::cout << mean_pos_rel_norm << std::endl;
                //     if (mean_pos_rel_norm < 0.05) {
                //         // reward_a[agent_i] -= coefficients[3];
                //         is_uniforms[agent_i] = true;
                //     }
                // }

                //////////////////////////////////////////////// reward form 3 ////////////////////////////////////////////////
                // 获取感知到的网格
                std::vector<int> list_grid;
                for (int i = 0; i < sensed_index[agent_i].size(); ++i) {
                    if (sensed_index[agent_i][i] != -1) {
                        list_grid.push_back(sensed_index[agent_i][i]);
                    }
                }

                if (!list_grid.empty()) {
                    // 计算相对位置
                    Matrix sensed_grid_pos_rel(2, std::vector<double>(list_grid.size()));
                    for (size_t j = 0; j < list_grid.size(); ++j) {
                        sensed_grid_pos_rel[0][j] = matrix_grid_center[0][list_grid[j]] - matrix_p[0][agent_i];
                        sensed_grid_pos_rel[1][j] = matrix_grid_center[1][list_grid[j]] - matrix_p[1][agent_i];
                    }

                    // 计算 weighted_diff 和 v_exp_i
                    std::vector<double> numerator(2, 0.0);
                    double denominator = 0.0;
                    for (size_t j = 0; j < list_grid.size(); ++j) {
                        numerator[0] += sensed_grid_pos_rel[0][j];
                        numerator[1] += sensed_grid_pos_rel[1][j];
                        denominator += 1.0;
                    }

                    std::vector<double> v_exp_i(2, 0.0);
                    v_exp_i[0] = 1.0 * numerator[0] / denominator;
                    v_exp_i[1] = 1.0 * numerator[1] / denominator;

                    double v_exp_i_norm = _norm(v_exp_i);
                    // std::cout << denominator << v_exp_i_norm << std::endl;
                    if (v_exp_i_norm < 0.07) {
                    // if (v_exp_i_norm < 0.02 && delta_num_voronoi_agent_i) {
                        is_uniforms[agent_i] = true;
                    }

                    // Matrix sensed_grid(2, std::vector<double>(list_grid.size()));
                    // for (size_t i = 0; i < list_grid.size(); ++i) {
                    //     sensed_grid[0][i] = matrix_grid_center[0][list_grid[i]];
                    //     sensed_grid[1][i] = matrix_grid_center[1][list_grid[i]];
                    // }

                    // if (!sensed_grid[0].empty()) {
                    //     // 计算相对位置
                    //     Matrix sensed_grid_pos_rel(2, std::vector<double>(sensed_grid[0].size()));
                    //     for (size_t j = 0; j < sensed_grid[0].size(); ++j) {
                    //         sensed_grid_pos_rel[0][j] = sensed_grid[0][j] - matrix_p[0][agent_i];
                    //         sensed_grid_pos_rel[1][j] = sensed_grid[1][j] - matrix_p[1][agent_i];
                    //     }

                    //     // 计算范数
                    //     std::vector<double> sensed_grid_pos_rel_norm(sensed_grid_pos_rel[0].size());
                    //     std::vector<double> agent_rel_j(2, 0.0);
                    //     for (size_t j = 0; j < sensed_grid_pos_rel[0].size(); ++j) {
                    //         agent_rel_j = _extract_column_one(sensed_grid_pos_rel, j);
                    //         sensed_grid_pos_rel_norm[j] = _norm(agent_rel_j);
                    //     }

                    //     // 计算 psi_values
                    //     std::vector<double> psi_values(sensed_grid_pos_rel_norm.size());
                    //     for (size_t j = 0; j < sensed_grid_pos_rel_norm.size(); ++j) {
                    //         psi_values[j] = _rho_cos_dec(sensed_grid_pos_rel_norm[j], 0.0, d_sen);
                    //     }

                    //     // 计算 weighted_diff 和 v_exp_i
                    //     std::vector<double> numerator(2, 0.0);
                    //     double denominator = 0.0;
                    //     for (size_t j = 0; j < psi_values.size(); ++j) {
                    //         numerator[0] += psi_values[j] * sensed_grid_pos_rel[0][j];
                    //         numerator[1] += psi_values[j] * sensed_grid_pos_rel[1][j];
                    //         denominator += psi_values[j];
                    //     }

                    //     if (denominator == 0) {
                    //         denominator = 1E-8; // 避免分母为零
                    //     }

                    //     std::vector<double> v_exp_i(2);
                    //     v_exp_i[0] = 1.0 * numerator[0] / denominator;
                    //     v_exp_i[1] = 1.0 * numerator[1] / denominator;
                        
                    //     double v_exp_i_norm = _norm(v_exp_i);
                    //     // std::cout << v_exp_i_norm << std::endl;
                    //     if (v_exp_i_norm < 0.01) {
                    //         is_uniforms[agent_i] = true;
                    //     }
                    // }
                }


                //////////////////////////////////////////////// reward form 4 ////////////////////////////////////////////////
                // std::vector<int> list_nei;
                // for (int i = 0; i < neighbor_index[agent_i].size(); ++i) {
                //     if (neighbor_index[agent_i][i] != -1) {
                //         list_nei.push_back(neighbor_index[agent_i][i]);
                //         // std::cout << neighbor_index[agent_i][i] << std::endl;
                //     }
                // }

                // // 获取感知到的网格
                // std::vector<int> list_grid;
                // for (int i = 0; i < sensed_index[agent_i].size(); ++i) {
                //     if (sensed_index[agent_i][i] != -1) {
                //         list_grid.push_back(sensed_index[agent_i][i]);
                //     }
                // }

                // // // 获取占据的网格
                // // std::vector<int> list_occupied_grid;
                // // for (int i = 0; i < occupied_index[agent_i].size(); ++i) {
                // //     if (occupied_index[agent_i][i] != -1) {
                // //         list_occupied_grid.push_back(occupied_index[agent_i][i]);
                // //     }
                // // }

                // // list_grid.insert(list_grid.end(), list_occupied_grid.begin(), list_occupied_grid.end());

                // list_nei.push_back(agent_i);
                // std::vector<int> num_grid_in_voronoi(list_nei.size(), 0);
                // std::vector<std::vector<int>> index_grid_in_voronoi(list_nei.size());
                // num_grid_in_voronoi.assign(list_nei.size(), 0); 
                // // 遍历每个 cell_index
                // for (int cell_index : list_grid) {
                //     std::vector<double> rel_pos_cell_nei_norm;
                    
                //     // 计算 rel_pos_cell_nei_norm
                //     for (int nei_index : list_nei) {
                //         std::vector<double> rel_pos_cell_nei = {
                //             matrix_p[0][nei_index] - matrix_grid_center[0][cell_index],
                //             matrix_p[1][nei_index] - matrix_grid_center[1][cell_index]
                //         };
                //         rel_pos_cell_nei_norm.push_back(_norm(rel_pos_cell_nei));
                //     }

                //     // 找到最小元素的索引
                //     int min_index = std::min_element(rel_pos_cell_nei_norm.begin(), rel_pos_cell_nei_norm.end()) - rel_pos_cell_nei_norm.begin();
                //     num_grid_in_voronoi[min_index]++;
                //     index_grid_in_voronoi[min_index].push_back(cell_index);
                // }

                // int last_index = list_nei.size() - 1;
                // if (num_grid_in_voronoi[last_index] > 0) {
                //     // 计算代理 agent_i 的探索期望速度
                //     std::vector<double> numerator = {0.0, 0.0};
                //     // 对 index_grid_in_voronoi 中的每个网格累加位置偏移量
                //     for (int cell_index : index_grid_in_voronoi[last_index]) {
                //         numerator[0] += matrix_grid_center[0][cell_index] - matrix_p[0][agent_i];
                //         numerator[1] += matrix_grid_center[1][cell_index] - matrix_p[1][agent_i];
                //     }
                //     double denominator = static_cast<double>(num_grid_in_voronoi[last_index]);
                //     // 计算期望探索速度 v_exp_i
                //     std::vector<double> v_exp_i = { numerator[0] / denominator, numerator[1] / denominator };
                //     // 检查 v_exp_i 的模长是否小于阈值
                //     double norm_v_exp_i = _norm(v_exp_i);
                //     if (norm_v_exp_i < 0.06) {
                //         is_uniforms[agent_i] = true;
                //     }
                // }

                // // // 计算 num_voronoi_nei_mean
                // // double num_voronoi_nei_sum = std::accumulate(num_grid_in_voronoi.begin(), num_grid_in_voronoi.end() - 1, 0.0);
                // // double num_voronoi_nei_mean = num_voronoi_nei_sum / static_cast<double>(list_nei.size() - 1);
                // // // 计算 delta_num_percent
                // // double delta_num_percent = std::abs(num_grid_in_voronoi.back() - num_voronoi_nei_mean) / static_cast<double>(list_grid.size());
                // // if (delta_num_percent < 0.05) {
                // //     is_uniforms[agent_i] = true;
                // // }

                // // // 检查最后一个元素是否大于之前记录的值
                // // if (num_grid_in_voronoi.back() > num_in_voronoi_last[agent_i]) {
                // //     is_uniforms[agent_i] = true;
                // // }
                // // num_in_voronoi_last[agent_i] = num_grid_in_voronoi.back();

                // // double num_voronoi_nei_mean = static_cast<double>(list_grid.size()) / static_cast<double>(list_nei.size());
                // // double delta_num_percent = std::abs(static_cast<double>(num_grid_in_voronoi.back()) - num_voronoi_nei_mean);
                // // // double delta_num_percent = std::abs(static_cast<double>(num_grid_in_voronoi.back()) - num_voronoi_mean_g);
                // // if (delta_num_percent < 5) {
                // //     // std::cout << delta_num_percent << std::endl;
                // //     is_uniforms[agent_i] = true;
                // // }

            }

            if (in_flags[agent_i] == 1 && is_has_collisions[agent_i] == false && is_uniforms[agent_i] == true) {
                reward_a[agent_i] += 1.0;
            }

        }
    }

    // penalize_control_effort
    if (condition[5]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double norm_a = 0.0;
            if (condition[1]) { // dynamics_mode == Cartesian
                norm_a = coefficients[6] * std::sqrt(std::pow(matrix_act[0][agent], 2) + std::pow(matrix_act[1][agent], 2));
            } else { // dynamics_mode == Polar
                norm_a = coefficients[7] * std::abs(matrix_act[0][agent]) + coefficients[8] * std::abs(matrix_act[1][agent]);
            }
            reward_a[agent] -= norm_a;
            // std::cout << coefficients[6] * norm_a << " ";
        }
        // std::cout << std::endl;
    }

    // penalize_collide_obstacles
    if (condition[6]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = n_a; i < n_a; ++i) {
                sum += is_collide_b2b[i][agent];
            }
            reward_a[agent] -= coefficients[9] * sum;
        }
    }

    // penalize_collide_walls
    if (condition[7]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = 0; i < 4; ++i) {
                sum += static_cast<double>(is_collide_b2w[i][agent]);
            }
            reward_a[agent] -= coefficients[10] * sum;
        }
    }

    for (int i = 0; i < n_a; ++i) {
        reward_input[i] = reward_a[i];
        num_in_voronoi_last_input[i] = num_in_voronoi_last[i];
    }
}

std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self) 
{
    std::vector<double> norms(Pos[0].size());
    for (int i = 0; i < Pos[0].size(); ++i) {
        norms[i] = std::sqrt(Pos[0][i] * Pos[0][i] + Pos[1][i] * Pos[1][i]);
    }

    std::vector<int> sorted_seq(norms.size());
    std::iota(sorted_seq.begin(), sorted_seq.end(), 0);
    std::sort(sorted_seq.begin(), sorted_seq.end(), [&](int a, int b) { return norms[a] < norms[b]; });

    Matrix sorted_Pos(2, std::vector<double>(Pos[0].size()));
    for (int i = 0; i < Pos[0].size(); ++i) {
        sorted_Pos[0][i] = Pos[0][sorted_seq[i]];
        sorted_Pos[1][i] = Pos[1][sorted_seq[i]];
    }

    std::vector<double> sorted_norms(norms.size());
    for (int i = 0; i < norms.size(); ++i) {
        sorted_norms[i] = norms[sorted_seq[i]];
    }

    Matrix new_Pos;
    for (int i = 0; i < 2; ++i) {
        std::vector<double> col;
        for (int j = 0; j < sorted_Pos[0].size(); ++j) {
            if (sorted_norms[j] < norm_threshold) {
                col.push_back(sorted_Pos[i][j]);
            }
        }
        new_Pos.push_back(col);
    }

    std::vector<int> new_sorted_seq;
    for (int i = 0; i < sorted_Pos[0].size(); ++i) {
        if (sorted_norms[i] < norm_threshold) {
            new_sorted_seq.push_back(sorted_seq[i]);
        }
    }

    if (remove_self) {
        new_Pos[0].erase(new_Pos[0].begin());
        new_Pos[1].erase(new_Pos[1].begin());
        new_sorted_seq.erase(new_sorted_seq.begin());
    }

    Matrix new_Vel(2, std::vector<double>(new_sorted_seq.size()));
    for (int i = 0; i < new_sorted_seq.size(); ++i) {
        new_Vel[0][i] = Vel[0][new_sorted_seq[i]];
        new_Vel[1][i] = Vel[1][new_sorted_seq[i]];
    }

    Matrix target_Pos(2, std::vector<double>(width));
    Matrix target_Vel(2, std::vector<double>(width));

    size_t until_idx = std::min(new_Pos[0].size(), static_cast<size_t>(width));
    std::vector<int> target_Nei(until_idx, -1);
    for (int i = 0; i < until_idx; ++i) {
        target_Pos[0][i] = new_Pos[0][i];
        target_Pos[1][i] = new_Pos[1][i];
        target_Vel[0][i] = new_Vel[0][i];
        target_Vel[1][i] = new_Vel[1][i];
        target_Nei[i] = new_sorted_seq[i];
    }

    return std::make_tuple(target_Pos, target_Vel, target_Nei);
}

void _make_periodic(Matrix& x, double L, std::vector<double> bound_pos, bool is_rel) {
    
    if (is_rel) {
        for (int i = 0; i < x.size(); ++i) {
            for (int j = 0; j < x[i].size(); ++j) {
                // 如果元素大于 L，就减去 2*L
                if (x[i][j] > L)
                    x[i][j] -= 2 * L;
                // 如果元素小于 -L，就加上 2*L
                else if (x[i][j] < -L)
                    x[i][j] += 2 * L;
            }
        }
    } else {
        for (int j = 0; j < x[0].size(); ++j) {
            if (x[0][j] < bound_pos[0]) {
                x[0][j] += 2 * L;
            } else if (x[0][j] > bound_pos[2]) {
                x[0][j] -= 2 * L;
            }
            if (x[1][j] < bound_pos[3]) {
                x[1][j] += 2 * L;
            } else if (x[1][j] > bound_pos[1]) {
                x[1][j] -= 2 * L;
            }
        }
    }
    
}

void _sf_b2b_all(double *p_input,
                 double *sf_b2b_input, 
                 double *d_b2b_edge_input,
                 bool *is_collide_b2b_input,
                 double *boundary_pos_input,
                 double *d_b2b_center_input,
                 int n_a,
                 int dim,
                 double k_ball,
                 bool is_periodic)
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_d_b2b_edge(n_a, std::vector<double>(n_a));
    Matrix matrix_d_b2b_center(n_a, std::vector<double>(n_a));
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<double> boundary_pos(4, 0.0);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            // std::cout << matrix_p[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_d_b2b_edge[i][j] = d_b2b_edge_input[i * n_a + j];
            matrix_d_b2b_center[i][j] = d_b2b_center_input[i * n_a + j];
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    Matrix sf_b2b_all(2 * n_a, std::vector<double>(n_a, 0.0));
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // 循环计算 sf_b2b_all
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < i; ++j) {
            Matrix delta = {
                {matrix_p[0][j] - matrix_p[0][i]},
                {matrix_p[1][j] - matrix_p[1][i]}
            };
            if (is_periodic) {
                _make_periodic(delta, L, boundary_pos, true);
            }

            double delta_x = delta[0][0] / matrix_d_b2b_center[i][j];
            double delta_y = delta[1][0] / matrix_d_b2b_center[i][j];
            sf_b2b_all[2 * i][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_x);
            sf_b2b_all[2 * i + 1][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_y);

            sf_b2b_all[2 * j][i] = -sf_b2b_all[2 * i][j];
            sf_b2b_all[2 * j + 1][i] = -sf_b2b_all[2 * i + 1][j];
            
            
        }
    }

    // 计算 sf_b2b
    Matrix sf_b2b(2, std::vector<double>(n_a));
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n_a; ++k) {
                sum += sf_b2b_all[2 * i + j][k];
            }
            sf_b2b[j][i] = sum;
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
           sf_b2b_input[i * n_a + j] = sf_b2b[i][j];
        }
    }

}

void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_a, 
                   double *boundary_pos) 
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
        }
    }

    Matrix d_b2w(4, std::vector<double>(n_a, 0.0));
    std::vector<std::vector<bool>> isCollision(4, std::vector<bool>(n_a, false));
    
    for (int i = 0; i < n_a; ++i) {
        d_b2w[0][i] = matrix_p[0][i] - r_input[i] - boundary_pos[0];
        d_b2w[1][i] = boundary_pos[1] - (matrix_p[1][i] + r_input[i]);
        d_b2w[2][i] = boundary_pos[2] - (matrix_p[0][i] + r_input[i]);
        d_b2w[3][i] = matrix_p[1][i] - r_input[i] - boundary_pos[3];
    }
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            isCollision[i][j] = (d_b2w[i][j] < 0);
            d_b2w[i][j] = std::abs(d_b2w[i][j]);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            d_b2w_input[i * n_a + j] = d_b2w[i][j];
            isCollision_input[i * n_a + j] = isCollision[i][j];
        }
    }
}

// 获取目标状态
std::tuple<bool, std::vector<double>, std::vector<double>, std::vector<int>> _get_target_grid_state(int self_id, 
                                                                                                    const Matrix& p, 
                                                                                                    const Matrix& dp,
                                                                                                    const Matrix& grid_center, 
                                                                                                    const double l_cell, 
                                                                                                    const double d_sen) 
{

    std::vector<double> target_pos(2), target_vel(2);

    // 计算 self_id 的相对位置
    Matrix rel_pos(2, std::vector<double>(grid_center[0].size()));
    for (size_t j = 0; j < grid_center[0].size(); ++j) {
        rel_pos[0][j] = grid_center[0][j] - p[0][self_id];
        rel_pos[1][j] = grid_center[1][j] - p[1][self_id];
    }

    // 计算距离
    std::vector<double> rel_pos_norm(grid_center[0].size());
    std::vector<double> rel_pos_j(2, 0.0);
    for (size_t j = 0; j < grid_center[0].size(); ++j) {
        rel_pos_j = _extract_column_one(rel_pos, j);
        rel_pos_norm[j] = _norm(rel_pos_j);
    }

    // 找到最近的网格中心
    auto min_it = std::min_element(rel_pos_norm.begin(), rel_pos_norm.end());
    int min_index = std::distance(rel_pos_norm.begin(), min_it);
    double min_dist = *min_it;

    bool in_flag;
    if (min_dist < std::sqrt(2) * l_cell / 2) {
        in_flag = true;
        target_pos = {p[0][self_id], p[1][self_id]};
        target_vel = {dp[0][self_id], dp[1][self_id]};
    } else {
        in_flag = false;
        target_pos = {grid_center[0][min_index], grid_center[1][min_index]};
        target_vel = {0.0, 0.0};
    }

    // 获取在感知范围内的网格索引
    std::vector<int> in_sense_indices;
    for (size_t j = 0; j < rel_pos_norm.size(); ++j) {
        if (rel_pos_norm[j] < d_sen) {
            in_sense_indices.push_back(j);
        }
    }

    return std::make_tuple(in_flag, target_pos, target_vel, in_sense_indices);
}

// 按行或按列拼接两个二维数组
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis) {
    if (axis == 0) { // 按行拼接
        // 创建一个新的二维数组，行数为两个数组的行数之和，列数为第一个数组的列数
        Matrix result(arr1.size() + arr2.size(), std::vector<double>(arr1[0].size()));

        // 将arr1复制到结果数组中
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // 将arr2复制到结果数组中
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[arr1.size() + i].begin());
        }

        return result;
    } else if (axis == 1) { // 按列拼接
        // 创建一个新的二维数组，行数为第一个数组的行数，列数为两个数组的列数之和
        Matrix result(arr1.size(), std::vector<double>(arr1[0].size() + arr2[0].size()));

        // 将arr1复制到结果数组中
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // 将arr2复制到结果数组中
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[i].begin() + arr1[0].size());
        }

        return result;
    } else {
        // 如果axis参数不是0或1，则返回空数组
        return Matrix();
    }
}

// 提取二维数组的指定列，并返回一个二维数组
Matrix _extract_column(const Matrix& arr, size_t col_index) {
    Matrix result;

    // 检查索引是否有效
    if (col_index < arr[0].size()) {
        // 遍历二维数组的每一行，并提取指定列的数据作为一个新的行
        for (const auto& row : arr) {
            result.push_back({row[col_index]});
        }
    }
    return result;
}

// 提取二维数组的指定列，并返回一个一维数组
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index) {
    std::vector<double> result;

    // 检查索引是否有效
    if (col_index < arr[0].size()) {
        // 遍历二维数组的每一行，并提取指定列的数据作为一个新的行
        for (const auto& row : arr) {
            result.push_back(row[col_index]);
        }
    }
    return result;
}

// Define a function to calculate _norm of a vector
double _norm(std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        // sum += x * x;
        sum += std::pow(x, 2);
    }
    return std::sqrt(sum);
}

bool _all_elements_greater_than_(std::vector<int>& arr, int n_l) {
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] <= (n_l - 1)) {
            return false;
        }
    }
    return true;
}

// Define a function to calculate cosine decay
static double _rho_cos_dec(double z, double delta, double r) {
    if (z < delta * r) {
        return 1.0;
    } else if (z < r) {
        // return 0.5 + 0.25 * (1.0 + std::cos(M_PI * (z / r - delta) / (1.0 - delta)));
        return (1.0 / 2.0) * (1.0 + std::cos(M_PI * (z / r - delta) / (1.0 - delta)));
        // return 1.0;
    } else {
        return 0.0;
    }
}


Matrix _vector_to_matrix(const std::vector<double>& vec) {
    Matrix matrix(vec.size(), std::vector<double>(1));

    for (size_t i = 0; i < vec.size(); ++i) {
        matrix[i][0] = vec[i];
    }

    return matrix;
}

std::vector<double> _matrix_to_vector(const Matrix& matrix) {
    std::vector<double> vec;
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            vec.push_back(element);
        }
    }
    return vec;
}

Matrix _transpose(const Matrix& matrix) {
    // 获取原始矩阵的行数和列数
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // 创建一个转置后的矩阵，行数和列数调换
    Matrix transposed(cols, std::vector<double>(rows));

    // 进行转置操作
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}
