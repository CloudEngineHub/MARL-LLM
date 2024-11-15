"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

# TASK_DES = """
# Each robot's goal is to enter the target area, avoid collisions with neighboring robots, and position itself as close as possible to the average position of unoccupied cells. 
# During this process, each robot only relies on its local observation vector to make decisions. 
# The robot's local observation vector is composed of four parts, totaling 192 dimensions:
# first, the first 4 dimensions correspond to the robot's position and velocity vector;
# second, the 5th to 28th dimensions correspond to the relative position and velocity vectors of 6 neighboring robots. 
# If the magnitude of any of these relative position vectors is less than twice the collision radius, a collision is considered to have occurred;
# third, the 29th to 32nd dimensions correspond to the relative position and velocity vectors of the target cell. 
# When the magnitude of the relative position vector of the target cell is zero, it indicates that the robot has entered the target area;
# fourth, the 33rd to 192nd dimensions correspond to the relative position vectors of unoccupied cells. 
# If the average of these relative position vectors is less than a threshold of 0.05, it indicates that the robot's position is very close to the average position of the unoccupied cells.
# """.strip()

TASK_DES = """
An observation vector matrix has been input with a shape of 192 x n, where n represents the number of robots, and 192 represents the length of each robot's observation vector. 
Each robot's observation vector consists of four parts:
1. The first 4 dimensions represent the robot's own position and velocity vectors, with each being 2-dimensional, totaling 4 dimensions.
2. Dimensions 5 to 28 represent the relative position and velocity vectors of 6 neighbors with respect to the robot. 
   Specifically, dimensions 5-8 represent the relative position and velocity of Neighbor 1, dimensions 9-12 for Neighbor 2, and dimensions 25-28 for Neighbor 6. 
   Therefore, dimensions 5-6, 9-10, 13-14, 17-18, 21-22, and 25-26 indicate the relative position vectors of Neighbors 1-6. 
   The norms of these six position vectors indicate the distances of Neighbors 1-6 from the robot. If all six distances are greater than 0.2, Condition 1 is satisfied.
3. Dimensions 29 to 32 represent the relative position and velocity of the target cell with respect to the robot, where dimensions 29-30 represent the relative position vector. 
   If the norm of this relative position vector is 0, Condition 2 is satisfied.
4. Dimensions 33 to 192 represent the relative position vectors of 80 unoccupied cells with respect to the robot, 
   where dimensions 33-34, 35-36, ..., 191-192 indicate the relative positions of the 1st, 2nd, ..., and 80th unoccupied cells. 
   If the norm of the mean of these 80 position vectors is less than 0.07, Condition 3 is satisfied.
""".strip()
