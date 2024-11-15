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

TASK_DES = """
Each robot's goal is to enter the target area, avoid collisions with neighboring robots, and position itself as close as possible to the average position of unoccupied cells. 
During this process, each robot only relies on its local observation vector to make decisions. 
The robot's local observation vector is composed of four parts, totaling 192 dimensions:
first, the first 4 dimensions correspond to the robot's position and velocity vector;
second, the 5th to 28th dimensions correspond to the relative position and velocity vectors of 6 neighboring robots. 
If the magnitude of any of these relative position vectors is less than twice the collision radius, a collision is considered to have occurred;
third, the 29th to 32nd dimensions correspond to the relative position and velocity vectors of the target cell. 
When the magnitude of the relative position vector of the target cell is zero, it indicates that the robot has entered the target area;
fourth, the 33rd to 192nd dimensions correspond to the relative position vectors of unoccupied cells. 
If the average of these relative position vectors is less than a threshold of 0.05, it indicates that the robot's position is very close to the average position of the unoccupied cells.
""".strip()
