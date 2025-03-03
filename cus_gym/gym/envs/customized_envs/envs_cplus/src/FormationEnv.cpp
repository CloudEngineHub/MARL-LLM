#include "FormationEnv.h"
#include <iostream>
#include <tuple>
#include <algorithm>
#include <vector>
#include <cmath> 
#include <numeric>
#include <typeinfo>
#include <random>

template <typename T>
T clamp(T value, T min_value, T max_value) {
    return std::max(min_value, std::min(value, max_value));
}

///////////////////////////////////////////////////////////////// observation form /////////////////////////////////////////////////////////////////
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
                      double Vel_max, 
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
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_g; ++j) {
            matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
        }
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    double boundary_width = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    double boundary_height = (boundary_pos[1] - boundary_pos[3]) / 2.0;
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
            _make_periodic(relPos_a2a, boundary_width, boundary_height, boundary_pos, true);
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
        if (condition[2]) { // Whether to include the agent's own state in the observation
            // Compute obs_agent_pos
            Matrix obs_agent_pos;
            obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_focused, 1);

            // Compute obs_agent_vel
            Matrix obs_agent_vel;
            obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_focused, 1);

            // Combine pos and vel into obs_agent
            obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);

        } else { // When not including the agent's own state
            obs_agent = _concatenate(relPos_a2a_focused, relVel_a2a_focused, 0);
        }

        // Transpose obs_agent and flatten it into a 1D array, then assign it to the front part of obs
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        //////////////////////////////////////////////////// Get target state ////////////////////////////////////////////////////
        bool in_flag;
        std::vector<double> target_grid_pos(2), target_grid_vel(2);
        std::vector<int> sensed_indices;
        std::tie(in_flag, target_grid_pos, target_grid_vel, sensed_indices) = _get_target_grid_state(agent_i, matrix_p, matrix_dp, matrix_grid_center, l_cell, d_sen);

        in_flags[agent_i] = in_flag;
        // Relative position and velocity
        std::vector<double> target_grid_pos_rel = {target_grid_pos[0] - matrix_p[0][agent_i], target_grid_pos[1] - matrix_p[1][agent_i]};
        std::vector<double> target_grid_vel_rel = {target_grid_vel[0] - matrix_dp[0][agent_i], target_grid_vel[1] - matrix_dp[1][agent_i]};

        //////////////////////////////////////////////////// Remove occupied grids ////////////////////////////////////////////////////
        size_t num_sensed_grid_origin = sensed_indices.size();
        Matrix sensed_grid(2, std::vector<double>(num_sensed_grid_origin));

        std::vector<int> occupied_indices = sensed_indices;
        if (num_sensed_grid_origin > 0) {
            for (size_t i = 0; i < num_sensed_grid_origin; ++i) {
                sensed_grid[0][i] = matrix_grid_center[0][sensed_indices[i]];
                sensed_grid[1][i] = matrix_grid_center[1][sensed_indices[i]];
            }

            if (in_flags[agent_i] == 1) {
                // Get the nearby agents
                std::vector<int> nearby_agents;
                std::vector<double> agent_pos_rel_norm(n_a, 0.0);
                for (size_t j = 0; j < n_a; ++j) {
                    double dx = matrix_p[0][j] - matrix_p[0][agent_i];
                    double dy = matrix_p[1][j] - matrix_p[1][agent_i];
                    agent_pos_rel_norm[j] = std::sqrt(dx * dx + dy * dy); // Calculate norm
                }

                for (size_t j = 0; j < agent_pos_rel_norm.size(); ++j) {
                    if (agent_pos_rel_norm[j] < (d_sen + r_avoid/2.0)) {
                        nearby_agents.push_back(j);  // Add index j to nearby_agents
                    }
                }
                
                for (int nearby_i : nearby_agents) {
                    // Calculate relative position of neighbors
                    Matrix grid_neigh_pos_relative(2, std::vector<double>(sensed_grid[0].size()));
                    for (size_t j = 0; j < sensed_grid[0].size(); ++j) {
                        grid_neigh_pos_relative[0][j] = sensed_grid[0][j] - matrix_p[0][nearby_i];
                        grid_neigh_pos_relative[1][j] = sensed_grid[1][j] - matrix_p[1][nearby_i];
                    }

                    // Calculate norm of neighbor positions
                    std::vector<double> grid_neigh_pos_relative_norm(grid_neigh_pos_relative[0].size());
                    std::vector<double> neigh_rel_j(2, 0.0);
                    for (size_t j = 0; j < grid_neigh_pos_relative[0].size(); ++j) {
                        neigh_rel_j = _extract_column_one(grid_neigh_pos_relative, j);
                        grid_neigh_pos_relative_norm[j] = _norm(neigh_rel_j);
                    }

                    // Filter based on norm
                    std::vector<bool> mask(grid_neigh_pos_relative_norm.size(), false);
                    for (size_t j = 0; j < grid_neigh_pos_relative_norm.size(); ++j) {
                        mask[j] = grid_neigh_pos_relative_norm[j] > r_avoid/2.0;
                    }

                    // Filter sensed_grid
                    Matrix filtered_sensed_grid(2);
                    for (size_t j = 0; j < mask.size(); ++j) {
                        if (mask[j]) {
                            filtered_sensed_grid[0].push_back(sensed_grid[0][j]);
                            filtered_sensed_grid[1].push_back(sensed_grid[1][j]);
                        }
                    }
                    // Filter sensed_indices
                    std::vector<int> filtered_sensed_indices;
                    for (size_t j = 0; j < mask.size(); ++j) {
                        if (mask[j]) {
                            filtered_sensed_indices.push_back(sensed_indices[j]);
                        }
                    }
                    sensed_grid = filtered_sensed_grid; // Update sensed_grid
                    sensed_indices = filtered_sensed_indices; // Update sensed_indices
                }
            }
        }

        //////////////////////////////////////////////////// Remove elements from occupied_indices that exist in sensed_indices ////////////////////////////////////////////////////
        std::vector<int> temp_occupied_indices;
        for (int i = 0; i < occupied_indices.size(); ++i) {
            if (std::find(sensed_indices.begin(), sensed_indices.end(), occupied_indices[i]) == sensed_indices.end()) {
                temp_occupied_indices.push_back(occupied_indices[i]);
            }
        }
        occupied_indices = temp_occupied_indices;
        int num_occupied_grid = occupied_indices.size();
        if (num_occupied_grid > num_occupied_grid_max) {
            double step = static_cast<double>(num_occupied_grid - 1) / (num_occupied_grid_max - 1); // -1 to ensure both first and last elements are selected
            // Uniformly select indices
            std::vector<int> final_indices;
            for (int i = 0; i < num_occupied_grid_max; ++i) {
                int index = static_cast<int>(std::round(i * step)); // Select index based on step
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

        //////////////////////////////////////////////////// Get positions of unoccupied grids //////////////////////////////////////////////////// 
        Matrix sensed_grid_pos;
        int num_sensed_grid = sensed_indices.size();
        if (num_sensed_grid > num_obs_grid_max) {
            Matrix sensed_grid_pos_1(2, std::vector<double>(num_obs_grid_max));
            // Calculate step
            double step = static_cast<double>(num_sensed_grid - 1) / (num_obs_grid_max - 1); // -1 to ensure both first and last elements are selected
            // Uniformly select indices
            std::vector<int> final_indices;
            for (int i = 0; i < num_obs_grid_max; ++i) {
                int index = static_cast<int>(std::round(i * step)); // Select index based on step
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
            Matrix sensed_grid_pos_3 = {}; // Empty 2D vector
            sensed_grid_pos = sensed_grid_pos_3;
        }

        //////////////////////////////////////////////////// Initialize sensed_grid_pos_rel with size dim x num_obs_grid_max ////////////////////////////////////////////////////
        Matrix sensed_grid_pos_rel(dim, std::vector<double>(num_obs_grid_max, 0.0));
        // If sensed_grid_pos is not empty
        if (!sensed_grid_pos.empty()) {
            int num_obs_grid = sensed_grid_pos[0].size();  // Get the number of observed grids
            for (int j = 0; j < num_obs_grid; ++j) {
                // Calculate relative position
                sensed_grid_pos_rel[0][j] = sensed_grid_pos[0][j] - matrix_p[0][agent_i];
                sensed_grid_pos_rel[1][j] = sensed_grid_pos[1][j] - matrix_p[1][agent_i];
            }
        }
        // Transpose sensed_grid_pos_rel and flatten it into a 1D array, then assign it to the corresponding part of obs
        std::vector<double> sensed_grid_pos_rel_flat;
        sensed_grid_pos_rel_flat.reserve(sensed_grid_pos_rel.size() * sensed_grid_pos_rel[0].size());
        for (size_t j = 0; j < sensed_grid_pos_rel[0].size(); ++j) {
            for (size_t i = 0; i < sensed_grid_pos_rel.size(); ++i) {
                sensed_grid_pos_rel_flat.push_back(sensed_grid_pos_rel[i][j]);
            }
        }

        //////////////////////////////////////////////////// Set observation matrix based on dynamics_mode ////////////////////////////////////////////////////
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
        }
    }

    for (int i = 0; i < n_a; ++i) {
        in_flags_input[i] = in_flags[i];
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_obs_grid_max; ++j) {
            sensed_index_input[i * num_obs_grid_max + j] = sensed_index[i][j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < num_occupied_grid_max; ++j) {
            occupied_index_input[i * num_occupied_grid_max + j] = occupied_index[i][j];
        }
    }
}

///////////////////////////////////////////////////////////////// reward form 1 /////////////////////////////////////////////////////////////////
void _get_reward(double *p_input, 
                 double *dp_input,
                 double *heading_input, 
                 double *act_input, 
                 double *reward_input, 
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

    double boundary_width = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    double boundary_height = (boundary_pos[1] - boundary_pos[3]) / 2.0;

    ///////////////////////////////////////////////////////////////////// manual designed reward /////////////////////////////////////////////////////////////////////
    // initialize reward_a matrix
    std::vector<double> reward_a(n_a, 0.0);
    std::vector<bool> is_collisions(n_a, false);
    std::vector<bool> is_uniforms(n_a, false);

    // interaction reward
    if (condition[3]) {
        for (int agent = 0; agent < n_a; ++agent) {

            std::vector<int> list_nei;
            for (int i = 0; i < neighbor_index[agent].size(); ++i) {
                if (neighbor_index[agent][i] != -1) {
                    list_nei.push_back(neighbor_index[agent][i]);
                }
            }

            std::vector<double> pos_rel(2, 0.0);
            std::vector<double> avg_neigh_vel(2, 0.0);

            if (!list_nei.empty()) {
                for (int agent2 : list_nei) {
                    if (condition[0]) {
                        Matrix pos_rel_mat = {{matrix_p[0][agent2] - matrix_p[0][agent]}, {matrix_p[1][agent2] - matrix_p[1][agent]}};
                        _make_periodic(pos_rel_mat, boundary_width, boundary_height, boundary_pos, true);
                        pos_rel = _matrix_to_vector(pos_rel_mat);
                    } else {
                        pos_rel = {matrix_p[0][agent2] - matrix_p[0][agent], matrix_p[1][agent2] - matrix_p[1][agent]};
                    }

                    if (r_avoid > _norm(pos_rel)) {
                        // reward_a[agent] -= coefficients[1];
                        is_collisions[agent] = true;
                        break;
                    }

                    // std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent2);
                    // avg_neigh_vel[0] += matrix_dp[0][agent2];
                    // avg_neigh_vel[1] += matrix_dp[1][agent2];
                    
                }

                // avg_neigh_vel[0] /= list_nei.size();
                // avg_neigh_vel[1] /= list_nei.size();
                // // std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent);
                // // double norm_dp_agent = _norm(dp_agent) + 1E-8;
                // double vel_diff_norm = std::sqrt(std::pow(avg_neigh_vel[0] - matrix_dp[0][agent], 2) + std::pow(avg_neigh_vel[1] - matrix_dp[1][agent], 2));
                // // std::cout << vel_diff_norm << std::endl;
                // // reward_a[agent] -= coefficients[2] * vel_diff_norm;
                // if (vel_diff_norm > 0.2) {
                //     is_collisions[agent] = true;
                // }
            }
        }
    }

    // exploration reward
    if (condition[4]) {
        for (int agent_i = 0; agent_i < n_a; ++agent_i) {
            if (in_flags[agent_i] == 1) {
                //////////////////////////////////////////////// reward form 1 ////////////////////////////////////////////////
                // get the list of sensed grids
                std::vector<int> list_grid;
                for (int i = 0; i < sensed_index[agent_i].size(); ++i) {
                    if (sensed_index[agent_i][i] != -1) {
                        list_grid.push_back(sensed_index[agent_i][i]);
                    }
                }

                if (!list_grid.empty()) {
                    // // compute relative positions
                    // Matrix sensed_grid_pos_rel(2, std::vector<double>(list_grid.size()));
                    // for (size_t j = 0; j < list_grid.size(); ++j) {
                    //     sensed_grid_pos_rel[0][j] = matrix_grid_center[0][list_grid[j]] - matrix_p[0][agent_i];
                    //     sensed_grid_pos_rel[1][j] = matrix_grid_center[1][list_grid[j]] - matrix_p[1][agent_i];
                    // }

                    // // compute weighted_diff and v_exp_i
                    // std::vector<double> numerator(2, 0.0);
                    // double denominator = 0.0;
                    // for (size_t j = 0; j < list_grid.size(); ++j) {
                    //     numerator[0] += sensed_grid_pos_rel[0][j];
                    //     numerator[1] += sensed_grid_pos_rel[1][j];
                    //     denominator += 1.0;
                    // }

                    // std::vector<double> v_exp_i(2, 0.0);
                    // v_exp_i[0] = 1.0 * numerator[0] / denominator;
                    // v_exp_i[1] = 1.0 * numerator[1] / denominator;

                    // double v_exp_i_norm = _norm(v_exp_i);
                    // // std::cout << denominator << v_exp_i_norm << std::endl;
                    // if (v_exp_i_norm < 0.06) {
                    // // if (v_exp_i_norm < 0.02 && delta_num_voronoi_agent_i) {
                    //     is_uniforms[agent_i] = true;
                    // }

                    // compute relative positions
                    Matrix sensed_grid_pos_rel(2, std::vector<double>(list_grid.size()));
                    for (size_t j = 0; j < list_grid.size(); ++j) {
                        sensed_grid_pos_rel[0][j] = matrix_grid_center[0][list_grid[j]] - matrix_p[0][agent_i];
                        sensed_grid_pos_rel[1][j] = matrix_grid_center[1][list_grid[j]] - matrix_p[1][agent_i];
                    }

                    // compute norm of sensed_grid_pos_rel
                    std::vector<double> sensed_grid_pos_rel_norm(sensed_grid_pos_rel[0].size());
                    std::vector<double> agent_rel_j(2, 0.0);
                    for (size_t j = 0; j < sensed_grid_pos_rel[0].size(); ++j) {
                        agent_rel_j = _extract_column_one(sensed_grid_pos_rel, j);
                        sensed_grid_pos_rel_norm[j] = _norm(agent_rel_j);
                    }

                    // compute psi_values
                    std::vector<double> psi_values(sensed_grid_pos_rel_norm.size());
                    for (size_t j = 0; j < sensed_grid_pos_rel_norm.size(); ++j) {
                        psi_values[j] = _rho_cos_dec(sensed_grid_pos_rel_norm[j], 0.0, d_sen);
                    }

                    // compute weighted_diff and v_exp_i
                    std::vector<double> numerator(2, 0.0);
                    double denominator = 0.0;
                    for (size_t j = 0; j < psi_values.size(); ++j) {
                        numerator[0] += psi_values[j] * sensed_grid_pos_rel[0][j];
                        numerator[1] += psi_values[j] * sensed_grid_pos_rel[1][j];
                        denominator += psi_values[j];
                    }

                    if (denominator == 0) {
                        denominator = 1E-8; // Avoid division by zero
                    }

                    std::vector<double> v_exp_i(2);
                    v_exp_i[0] = 1.0 * numerator[0] / denominator;
                    v_exp_i[1] = 1.0 * numerator[1] / denominator;
                    
                    double v_exp_i_norm = _norm(v_exp_i);
                    // std::cout << v_exp_i_norm << std::endl;
                    if (v_exp_i_norm < 0.05) { // 0.05
                        is_uniforms[agent_i] = true;
                    }
                }

            }

            if (in_flags[agent_i] == 1 && is_collisions[agent_i] == false && is_uniforms[agent_i] == true) {
                reward_a[agent_i] += 1.0;
            }

        }
    }

    ///////////////////////////////////////////////////////////////////// irl reward /////////////////////////////////////////////////////////////////////
    // Initialize reward_a matrix
    // std::vector<double> reward_a(n_a, 0.0);

    ///////////////////////////////////////////////////////////////////// llm reward only /////////////////////////////////////////////////////////////////////
    // // Initialize reward_a matrix
    // std::vector<double> reward_a(n_a, 0.0);

    // for (int robot_id = 0; robot_id < n_a; ++robot_id) {
    //     // check if the robot is within the target region
    //     bool in_target = isWithinTargetRegion(robot_id, in_flags);
    //     if (!in_target) {
    //         continue; // 奖励保持为 0
    //     }

    //     // check for collisions with neighbors
    //     std::vector<int> neighbors = getNeighborIds(robot_id, neighbor_index);
    //     std::vector<double> robot_pos(2), robot_vel(2);
    //     std::tie(robot_pos, robot_vel) = getPositionAndVelocity(robot_id, matrix_p, matrix_dp);
    //     bool collision = false;
    //     std::vector<double> neighbor_pos(2), neighbor_vel(2), pos_to_neighbor(2);
    //     for (int neighbor_id : neighbors) {
    //         std::tie(neighbor_pos, neighbor_vel) = getPositionAndVelocity(neighbor_id, matrix_p, matrix_dp);
    //         pos_to_neighbor = {neighbor_pos[0] - robot_pos[0], neighbor_pos[1] - robot_pos[1]};
    //         double distance = _norm(pos_to_neighbor);
    //         if (distance < r_avoid) {
    //             collision = true;
    //             // std::cout << robot_id << ", " << neighbor_id << ", " << distance << std::endl;
    //             break;
    //         }
    //     }
    //     if (collision) {
    //         continue; // remain reward as 0
    //     }

    //     // check if the robot has explored unoccupied areas
    //     Matrix unoccupied_cells = getUnoccupiedCellsPosition(robot_id, sensed_index, matrix_grid_center);
    //     if (unoccupied_cells[0].empty()) {
    //         continue; // there is no unoccupied cell
    //     }

    //     // compute centroid of unoccupied cells
    //     std::vector<double> centroid(2, 0.0);
    //     for (size_t j = 0; j < unoccupied_cells[0].size(); ++j) {
    //         centroid[0] += unoccupied_cells[0][j];
    //         centroid[1] += unoccupied_cells[1][j];
    //     }
    //     centroid[0] /= static_cast<double>(unoccupied_cells[0].size());
    //     centroid[1] /= static_cast<double>(unoccupied_cells[0].size());

    //     // compute distance to centroid
    //     std::vector<double> pos_to_centroid = {centroid[0] - robot_pos[0], centroid[1] - robot_pos[1]};
    //     double distance_to_centroid = _norm(pos_to_centroid);
    //     // std::cout << distance_to_centroid << std::endl;

    //     // check if the robot is close to the centroid
    //     if (distance_to_centroid < 0.05) {
    //         // std::cout << coefficients[0] << std::endl;
    //         reward_a[robot_id] = 1.0; // the task is completed
    //     }
    // }

    for (int i = 0; i < n_a; ++i) {
        reward_input[i] = reward_a[i];
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

void _make_periodic(Matrix& x, double boundary_width_half, double boundary_height_half, std::vector<double> boundary_pos, bool is_rel) {
    if (is_rel) {
        for (int j = 0; j < x[0].size(); ++j) {
            // Handle width direction
            if (x[0][j] < -boundary_width_half) {
                x[0][j] += 2 * boundary_width_half;
            } else if (x[0][j] > boundary_width_half) {
                x[0][j] -= 2 * boundary_width_half;
            }
            // Handle height direction
            if (x[1][j] < -boundary_height_half) {
                x[1][j] += 2 * boundary_height_half;
            } else if (x[1][j] > boundary_height_half) {
                x[1][j] -= 2 * boundary_height_half;
            }
        }
    } else {
        for (int j = 0; j < x[0].size(); ++j) {
            // Handle width direction
            if (x[0][j] < boundary_pos[0]) {
                x[0][j] += 2 * boundary_width_half;
            } else if (x[0][j] > boundary_pos[2]) {
                x[0][j] -= 2 * boundary_width_half;
            }
            // Handle height direction
            if (x[1][j] < boundary_pos[3]) {
                x[1][j] += 2 * boundary_height_half;
            } else if (x[1][j] > boundary_pos[1]) {
                x[1][j] -= 2 * boundary_height_half;
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
        }
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
    double boundary_width = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    double boundary_height = (boundary_pos[1] - boundary_pos[3]) / 2.0;

    // Loop to calculate sf_b2b_all
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < i; ++j) {
            Matrix delta = {
                {matrix_p[0][j] - matrix_p[0][i]},
                {matrix_p[1][j] - matrix_p[1][i]}
            };
            if (is_periodic) {
                _make_periodic(delta, boundary_width, boundary_height, boundary_pos, true);
            }

            double delta_x = delta[0][0] / matrix_d_b2b_center[i][j];
            double delta_y = delta[1][0] / matrix_d_b2b_center[i][j];
            sf_b2b_all[2 * i][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_x);
            sf_b2b_all[2 * i + 1][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_y);

            sf_b2b_all[2 * j][i] = -sf_b2b_all[2 * i][j];
            sf_b2b_all[2 * j + 1][i] = -sf_b2b_all[2 * i + 1][j];
            
            
        }
    }

    // Calculate sf_b2b
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

// Get target state
std::tuple<bool, std::vector<double>, std::vector<double>, std::vector<int>> _get_target_grid_state(int self_id, 
                                                                                                    const Matrix& p, 
                                                                                                    const Matrix& dp,
                                                                                                    const Matrix& grid_center, 
                                                                                                    const double l_cell, 
                                                                                                    const double d_sen) 
{

    std::vector<double> target_pos(2), target_vel(2);

    // Calculate relative position of self_id
    Matrix rel_pos(2, std::vector<double>(grid_center[0].size()));
    for (size_t j = 0; j < grid_center[0].size(); ++j) {
        rel_pos[0][j] = grid_center[0][j] - p[0][self_id];
        rel_pos[1][j] = grid_center[1][j] - p[1][self_id];
    }

    // Calculate distance
    std::vector<double> rel_pos_norm(grid_center[0].size());
    std::vector<double> rel_pos_j(2, 0.0);
    for (size_t j = 0; j < grid_center[0].size(); ++j) {
        rel_pos_j = _extract_column_one(rel_pos, j);
        rel_pos_norm[j] = _norm(rel_pos_j);
    }

    // Find the nearest grid center
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

    // Get grid indices within sensing range
    std::vector<int> in_sense_indices;
    for (size_t j = 0; j < rel_pos_norm.size(); ++j) {
        if (rel_pos_norm[j] < d_sen) {
            in_sense_indices.push_back(j);
        }
    }

    return std::make_tuple(in_flag, target_pos, target_vel, in_sense_indices);
}

// Concatenate two 2D arrays by row or by column
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis) {
    if (axis == 0) { // Concatenate by row
        // Create a new 2D array with the number of rows being the sum of the rows of the two arrays and the number of columns being the columns of the first array
        Matrix result(arr1.size() + arr2.size(), std::vector<double>(arr1[0].size()));

        // Copy arr1 to the result array
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // Copy arr2 to the result array
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[arr1.size() + i].begin());
        }

        return result;
    } else if (axis == 1) { // Concatenate by column
        // Create a new 2D array with the number of rows being the rows of the first array and the number of columns being the sum of the columns of the two arrays
        Matrix result(arr1.size(), std::vector<double>(arr1[0].size() + arr2[0].size()));

        // Copy arr1 to the result array
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // Copy arr2 to the result array
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[i].begin() + arr1[0].size());
        }

        return result;
    } else {
        // If the axis parameter is not 0 or 1, return an empty array
        return Matrix();
    }
}

// Extract the specified column of a 2D array and return a 2D array
Matrix _extract_column(const Matrix& arr, size_t col_index) {
    Matrix result;

    // Check if the index is valid
    if (col_index < arr[0].size()) {
        // Traverse each row of the 2D array and extract the data of the specified column as a new row
        for (const auto& row : arr) {
            result.push_back({row[col_index]});
        }
    }
    return result;
}

// Extract the specified column of a 2D array and return a 1D array
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index) {
    std::vector<double> result;

    // Check if the index is valid
    if (col_index < arr[0].size()) {
        // Traverse each row of the 2D array and extract the data of the specified column as a new row
        for (const auto& row : arr) {
            result.push_back(row[col_index]);
        }
    }
    return result;
}

Matrix _divide(const Matrix& matrix, double scalar) {
    // Check for division by zero
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero in _divide function.");
    }

    // Create the result matrix and calculate element-wise
    Matrix result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = matrix[i][j] / scalar;
        }
    }

    return result;
}

// Define a function to calculate _norm of a vector
double _norm(std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
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
        return (1.0 / 2.0) * (1.0 + std::cos(M_PI * (z / r - delta) / (1.0 - delta)));
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
    // Get the number of rows and columns of the original matrix
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // Create a transposed matrix with the number of rows and columns swapped
    Matrix transposed(cols, std::vector<double>(rows));

    // Perform the transpose operation
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void calculateActionPrior(double *p_input, 
                        double *dp_input, 
                        double *a_prior_input, 
                        double *grid_center_input,
                        int *neighbor_index_input,
                        double d_sen,
                        double r_avoid,
                        double l_cell, 
                        int topo_nei_max,
                        int n_a,
                        int n_g,
                        int dim) 
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    Matrix matrix_grid_center(dim, std::vector<double>(n_g));
    Matrix a_prior(dim, std::vector<double>(n_a, 0.0));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index[i][j] = neighbor_index_input[i * topo_nei_max + j];
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_g; ++j) {
            matrix_grid_center[i][j] = grid_center_input[i * n_g + j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        bool in_flag;
        std::vector<double> target_grid_pos(2), target_grid_vel(2);
        std::vector<int> sensed_indices;
        std::tie(in_flag, target_grid_pos, target_grid_vel, sensed_indices) = _get_target_grid_state(i, matrix_p, matrix_dp, matrix_grid_center, l_cell, d_sen);

        std::vector<double> policy_force = robotPolicy(i, target_grid_pos, matrix_p, matrix_dp, neighbor_index, r_avoid);

        // Write the result to a_prior
        for (int d = 0; d < dim; ++d) {
            a_prior[d][i] = policy_force[d];
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            a_prior_input[i * n_a + j] = a_prior[i][j];
        }
    }

}

// Policy function to calculate robot actions
std::vector<double> robotPolicy(int agent_id, 
                                const std::vector<double>& target_position, 
                                const Matrix& matrix_p, 
                                const Matrix& matrix_dp, 
                                const std::vector<std::vector<int>>& neighbor_index,
                                double r_avoid) {
    // Parameter settings
    double repulsion_strength = 3.0;
    double attraction_strength = 2.0;
    double sync_strength = 2.0;
    double force_limit_min = -1.0;
    double force_limit_max = 1.0;

    // Get robot position and velocity
    std::vector<double> position = {matrix_p[0][agent_id], matrix_p[1][agent_id]};
    std::vector<double> velocity = {matrix_dp[0][agent_id], matrix_dp[1][agent_id]};

    // Initialize total force
    std::vector<double> total_force(2, 0.0);

    // Calculate target attraction force
    std::vector<double> direction_to_target = {target_position[0] - position[0], target_position[1] - position[1]};
    double distance_to_target = std::sqrt(direction_to_target[0] * direction_to_target[0] +
                                          direction_to_target[1] * direction_to_target[1]);
    if (distance_to_target > 0) {
        total_force[0] += attraction_strength * direction_to_target[0] / distance_to_target;
        total_force[1] += attraction_strength * direction_to_target[1] / distance_to_target;
    }

    // Get neighbor list
    std::vector<int> neighbors = getNeighborIds(agent_id, neighbor_index);

    // Initialize neighbor velocity and repulsion force
    std::vector<double> average_neighbor_velocity(2, 0.0);
    int neighbor_count = 0;

    for (int neighbor_id : neighbors) {
        std::vector<double> neighbor_position = {matrix_p[0][neighbor_id], matrix_p[1][neighbor_id]};
        std::vector<double> neighbor_velocity = {matrix_dp[0][neighbor_id], matrix_dp[1][neighbor_id]};

        // Calculate direction and distance to neighbor
        std::vector<double> direction_to_neighbor = {position[0] - neighbor_position[0], position[1] - neighbor_position[1]};
        double distance_to_neighbor = _norm(direction_to_neighbor);

        // Calculate repulsion force
        if (distance_to_neighbor > 0 && distance_to_neighbor < r_avoid) {
            // Calculate unit direction vector
            double unit_direction_to_neighbor_x = direction_to_neighbor[0] / distance_to_neighbor;
            double unit_direction_to_neighbor_y = direction_to_neighbor[1] / distance_to_neighbor;

            double factor = repulsion_strength * (r_avoid / distance_to_neighbor - 1.0);
            total_force[0] += factor * unit_direction_to_neighbor_x;
            total_force[1] += factor * unit_direction_to_neighbor_y;
        }

        // Accumulate neighbor velocity
        average_neighbor_velocity[0] += neighbor_velocity[0];
        average_neighbor_velocity[1] += neighbor_velocity[1];
        neighbor_count++;
    }

    // Calculate synchronization force
    if (neighbor_count > 0) {
        average_neighbor_velocity[0] /= neighbor_count;
        average_neighbor_velocity[1] /= neighbor_count;

        total_force[0] += sync_strength * (average_neighbor_velocity[0] - velocity[0]);
        total_force[1] += sync_strength * (average_neighbor_velocity[1] - velocity[1]);
    }

    // Limit the range of the force
    total_force[0] = clamp(total_force[0], force_limit_min, force_limit_max);
    total_force[1] = clamp(total_force[1], force_limit_min, force_limit_max);

    return total_force;
}

// Get the position and velocity of the specified robot ID
std::tuple<std::vector<double>, std::vector<double>> getPositionAndVelocity(
    int robot_id, 
    const Matrix& p, 
    const Matrix& dp) 
{
    // Get position [x, y]
    std::vector<double> position = {p[0][robot_id], p[1][robot_id]};
    
    // Get velocity [vx, vy]
    std::vector<double> velocity = {dp[0][robot_id], dp[1][robot_id]};
    
    return std::make_tuple(position, velocity);
}

// Get the neighbors of the specified robot (assuming neighbor index data is provided)
std::vector<int> getNeighborIds(int agent_id, const std::vector<std::vector<int>>& neighbor_index) {
    std::vector<int> neighbors;
    for (int id : neighbor_index[agent_id]) {
        if (id != -1) {
            neighbors.push_back(id);
        }
    }
    return neighbors;
}

Matrix getUnoccupiedCellsPosition(
    int robot_id, 
    const std::vector<std::vector<int>>& sensed_index, 
    const Matrix& grid_center) 
{
    // Get the grid indices sensed by the robot
    std::vector<int> list_grid;
    for (int i = 0; i < sensed_index[robot_id].size(); ++i) {
        if (sensed_index[robot_id][i] != -1) {
            list_grid.push_back(sensed_index[robot_id][i]);
        }
    }

    // Initialize sensed_grid with shape 2 x list_grid.size()
    Matrix sensed_grid(2, std::vector<double>(list_grid.size()));
    for (size_t j = 0; j < list_grid.size(); ++j) {
        sensed_grid[0][j] = grid_center[0][list_grid[j]];
        sensed_grid[1][j] = grid_center[1][list_grid[j]];
    }

    return sensed_grid; // Return the positions of the sensed grid centers
}

// Determine if the robot is within the target region
bool isWithinTargetRegion(int robot_id, const std::vector<int>& in_flags) {
    return static_cast<bool>(in_flags[robot_id]);
}

std::vector<double> getTargetCellPosition(int robot_id, 
                                        const Matrix& p, 
                                        const Matrix& dp,
                                        const Matrix& grid_center, 
                                        double l_cell, 
                                        double d_sen) 
{
    // Get the target grid state
    bool in_flag;
    std::vector<double> target_pos(2), target_vel(2);
    std::vector<int> in_sense_indices;

    std::tie(in_flag, target_pos, target_vel, in_sense_indices) = 
        _get_target_grid_state(robot_id, p, dp, grid_center, l_cell, d_sen);

    // Return the target position
    return target_pos;
}
