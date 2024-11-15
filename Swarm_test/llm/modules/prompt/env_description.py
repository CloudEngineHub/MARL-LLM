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

ENV_DES = """
Consider a plane with a target area and a group of robots. The target area is discretized into a set of cells. 
Each robot is represented by a small disc, with a sensing radius and a collision radius. 
The robot can obtain the relative position and velocity of neighboring robots and the relative positions of the cells within its sensing radius. 
When the distance between two robots is less than twice the sensing radius, they are considered to have collided.
Additionally, when the distance between the center of a cell and the center of a robot is less than the robot’s collision radius, that cell is considered to be occupied by the robot.
""".strip()
