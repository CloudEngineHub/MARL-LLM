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

# REWARD_STATEVALUE_CRITIC_PROMPT_TEMPLATE: str = """
# ## These are the task description:
# {task_des}

# ## Role setting:
# - You serve as a judgment assistant with two key roles. 
# First, you provide a reward: if the robot's local observation vector simultaneously satisfies three conditions, you return a reward value of 1; otherwise, you return 0. 
# Second, you provide a state value: if the current observation vector is close to the desired observation vector, you assign a higher state value; otherwise, 
# you assign a lower state value.

# ## These are the environment description:
# {env_des}

# ## These are the user original instructions:
# {instruction}

# ## There are the local observation vectors of all robots:
# {observation}

# ## The output TEXT format is as follows:
# {output_template}

# ## Notes:
# - The collision radius is 0.1.
# - In the 5th to 28th dimensions of the robot's local observation vector, there may be 4n zero elements because the robot has not observed all 6 neighbors, but only 6-n neighbors. In this case, these zero elements do not need to be considered, as they do not contain useful information.
# - In the 33rd to 192nd dimensions of the robot's local observation vector, there may be 2n zero elements because the robot has not observed all 80 unoccupied cells, but only 80-n unoccupied cells. In this case, these zero elements do not need to be considered, as they do not contain useful information.
# - The input observation has a shape of 192 x n, where n represents the number of robots. Therefore, you need to output n rewards and state values.
# - To prevent the value from becoming too large, the range of the state value should be limited to between -1 and 1.
# - The output should strictly adhere to the specified format.
# """.strip()

# CRITIC_TEMPLATE: str = """
# ##reasoning:
# ```json
# {
#   "critic_ouput": [
#     {
#       "reward": "the reward value list",
#       "description": "Describe why this reward is given. Output the calculation basis. (If the user's requirements involve specific numerical values, they should be reflected in the description. )"
#     },
#     {
#       "state_value": "the state value list",
#       "description": "Describe why this state value is given. Output the calculation basis. (If the user's requirements involve specific numerical values, they should be reflected in the description. )“
#     }
#   ]
# }
# ```
# """.strip()

REWARD_STATEVALUE_CRITIC_PROMPT_TEMPLATE: str = """
## There are the observation vector matrix:
{observation}

## These are the task description:
{task_des}

## Role setting:
- You serve as a judgment assistant with two key roles. 
First, you provide a reward: if the robot's observation vector simultaneously satisfies three conditions, you return a reward value of 1; otherwise, you return 0. 
Second, you provide a state value: if the current observation vector is close to the desired observation vector, you assign a higher state value; otherwise, 
you assign a lower state value. The desired observation vector is one that simultaneously satisfies all three conditions.

## The output TEXT format is as follows:
{output_template}

## Notes:
- In the 5th to 28th dimensions of the robot's observation vector, there may be 4 x m zero elements because the robot has not observed all 6 neighbors, but only 6 - m neighbors. In this case, these zero elements do not need to be considered, as they do not contain useful information.
- In the 33rd to 192nd dimensions of the robot's observation vector, there may be 2 x m zero elements because the robot has not observed all 80 unoccupied cells, but only 80 - m unoccupied cells. In this case, these zero elements do not need to be considered, as they do not contain useful information.
- The input observation has a shape of 192 x n, where n represents the number of robots. Therefore, you need to output n rewards and state values.
- To prevent the value from becoming too large, the range of the state value should be limited to between -1 and 1.
- The output should strictly adhere to the specified format.
""".strip()

CRITIC_TEMPLATE: str = """
##reasoning:
```json
{
  "critic_ouput": [
    {
      "reward": "the reward value list",
      "description": "Describe why this reward is given. Output the calculation basis. (If the user's requirements involve specific numerical values, they should be reflected in the description. )"
    },
    {
      "state_value": "the state value list",
      "description": "Describe why this state value is given. Output the calculation basis. (If the user's requirements involve specific numerical values, they should be reflected in the description. )“
    }
  ]
}
```
""".strip()
