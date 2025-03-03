#ifndef FORMATION_ENV_H
#define FORMATION_ENV_H

#include <vector>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

using Matrix = std::vector<std::vector<double>>;

struct Env {
    // Define env properties here
    // Example: position, velocity, heading, etc.
    Matrix p; // Position
    Matrix dp; // Velocity
    Matrix obs;
    std::vector<double> leader_state;
    double d_sen;
    int topo_nei_max;
    int n_a; // Number of agents
    int obs_dim_agent; // Observation dimension for each env
    int dim; // Dimensionality
    // Add other required properties here
};

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
                      bool *condition);
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
                 double *coefficients);
std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self);
void _sf_b2b_all(double *p_input,
                 double *sf_b2b_input, 
                 double *d_b2b_edge_input,
                 bool *is_collide_b2b_input,
                 double *boundary_pos_input,
                 double *d_b2b_center_input,
                 int n_a,
                 int dim,
                 double k_ball,
                 bool is_periodic);
void _make_periodic(Matrix& x, double boundary_width_half, double boundary_height_half, std::vector<double> boundary_pos, bool is_rel);
void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_a, 
                   double *boundary_L_half);
std::tuple<bool, std::vector<double>, std::vector<double>, std::vector<int>> _get_target_grid_state(int self_id, 
                                                                                                    const Matrix& p, 
                                                                                                    const Matrix& dp,
                                                                                                    const Matrix& grid_center, 
                                                                                                    const double l_cell, 
                                                                                                    const double d_sen);
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis);
Matrix _extract_column(const Matrix& arr, size_t col_index);
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index);
Matrix _divide(const Matrix& matrix, double scalar);
double _norm(std::vector<double>& v);
bool _all_elements_greater_than_(std::vector<int>& arr, int n_l);
static double _rho_cos_dec(double z, double delta, double r);
Matrix _vector_to_matrix(const std::vector<double>& vec);
std::vector<double> _matrix_to_vector(const Matrix& matrix);
Matrix _transpose(const Matrix& matrix);
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
                            int dim);
std::vector<double> robotPolicy(int agent_id, 
                                const std::vector<double>& target_position, 
                                const Matrix& matrix_p, 
                                const Matrix& matrix_dp, 
                                const std::vector<std::vector<int>>& neighbor_index,
                                double r_avoid);
std::tuple<std::vector<double>, std::vector<double>> getPositionAndVelocity(
    int robot_id, 
    const Matrix& p, 
    const Matrix& dp);
std::vector<int> getNeighborIds(int agent_id, const std::vector<std::vector<int>>& neighbor_index);
Matrix getUnoccupiedCellsPosition(
    int robot_id, 
    const std::vector<std::vector<int>>& sensed_index, 
    const Matrix& grid_center);
bool isWithinTargetRegion(int robot_id, const std::vector<int>& in_flags);
std::vector<double> getTargetCellPosition(int robot_id, 
                                        const Matrix& p, 
                                        const Matrix& dp,
                                        const Matrix& grid_center, 
                                        double l_cell, 
                                        double d_sen);

#ifdef __cplusplus
}
#endif

#endif
