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

import os
import traceback

import yaml
from modules.framework.parser import CodeParser

robot_api_prompt = """
def get_self_id():
    '''
    Description: Get the unique ID of the robot itself.
    Returns:
    - int: The unique ID of the robot itself.ID is a unique identifier for each robot in the environment.
    '''

def get_all_robots_id():
    '''
    Description: Get the unique IDs of all robots in the environment.(Including the robot itself)
    Returns:
    - list: A list of integers, each representing the unique ID of a robot in the environment.
    '''

def get_self_position():
    '''
    Description: Get the current position of the robot itself in real-time.
    Returns:
    - numpy.ndarray: The current, real-time position of the robot.
    '''

def set_self_velocity(velocity):
    '''
    Description: Set the velocity of the robot itself in real-time.
    Input:
    - velocity (numpy.ndarray): The new velocity to be set immediately.
    '''

def get_self_velocity():
    '''
    Get the current velocity of the robot itself in real-time.
    Returns:
    - numpy.ndarray: The current, real-time velocity of the robot.
    '''

def get_self_radius():
    '''
    Description: Get the radius of the robot itself.
    Returns:
    - float: The radius of the robot itself.
    '''

def stop_self():
    '''
    Description: Stop the robot itself.
    '''

def get_environment_range():
    '''
    Description: Get the x and y range of the environment.
    Returns:
    - dict: A dictionary containing the x and y range of the environment.Robot should keep within the range.
        - x_min (float): The minimum x value of the environment.
        - x_max (float): The maximum x value of the environment.
        - y_min (float): The minimum y value of the environment.
        - y_max (float): The maximum y value of the environment.
    '''

def get_surrounding_environment_info():
    '''
    Get real-time information of the surrounding robots and obstacles within the perception range.
    Returns:
    - list: A list of dictionaries, each containing the type, position, and other relevant information of surrounding objects (robots or obstacles).
        - "Type": A string indicating whether the object is a 'robot' or 'obstacle'.
        - "position": The current position of the object.
        - "velocity": The current velocity of the object. If the object is an obstacle, the velocity is [0, 0].
        - "radius": The radius of the object.
    '''

def get_prey_position():
    '''
    Description: Prey will keep moving in the environment.Get the real-time position of the prey in the environment.
    Returns:
    - numpy.ndarray: The position of the prey.
    '''

def get_lead_position():
    '''
    Description: Lead will keep moving in the environment.Get the real-time position of the prey in the environment.
    Returns:
    - numpy.ndarray: The position of the lead.
    '''

def get_target_position():
    '''
    Description: Get the target position for the robot to reach.
    Returns:
    - numpy.ndarray: The current position of the target.
    '''

def get_target_formation_points():
    '''
    Description: Get the target formation points for the robot to reach.
    Returns:
    - list: A list of numpy.ndarray, all representing target points for robots.
    '''

def get_surrounding_unexplored_area():
    '''
    Description: Get the unexplored area in the Robot's perception range.
    Returns:
    - list: A list of dictionaries containing the id and position of an unexplored area.
        - id (int): The unique ID of the unexplored area.
        - position (numpy.ndarray): The position of the unexplored area.
    '''

def get_quadrant_target_position():
    '''
    Description: Get the target position for the robot in the quadrant.
    Returns:
    - dict: A dictionary where the keys are quadrant indices and values are target positions.
        - key (int): The quadrant index.
        - value (numpy.ndarray): The target position for the robot in the quadrant.
    '''

def get_all_robots_initial_position():
    '''
    Description: Get the initial position of all robots in the environment.
    Returns:
    -dict: A dictionary where the keys are robot IDs and values are the initial positions of the robots.
        - key: int, the robot id.
        - value: numpy.ndarray, the initial position of the robot.
    '''

def get_prey_initial_position():
    '''
    Description: Get the initial position of the prey.Prey will move in the environment.
    Note: The prey will move in the environment, and the position will be updated in real-time.So your policy should consider the dynamic position of the prey.
    Returns:
    - list: A list of float, representing the initial position of the prey.
    '''

def get_initial_unexplored_areas():
    '''
    Description: Get the initial unexplored areas in the environment.
    Returns:
    -list: A list of numpy.ndarray, each representing the center position of an unexplored area.
    '''
""".strip()


class RobotApi:
    def __init__(self, content):
        self.content = content
        code_obj = CodeParser()
        code_obj.parse_code(self.content)
        self.apis = code_obj.function_defs

        self.base_apis = [
            "get_all_robots_id",
            "get_self_position",
            "set_self_velocity",
            "get_self_radius",
            "get_surrounding_environment_info",
            "get_all_robots_initial_position",
        ]

        # Updated API scope mapping to support multiple scopes per API
        self.api_scope = {
            "get_self_position": ["local"],
            "set_self_velocity": ["local"],
            "get_self_velocity": ["local"],
            "stop_self": ["local"],
            "get_environment_range": ["local", "global"],
            "get_self_radius": ["local"],
            "get_prey_position": ["local"],
            "get_lead_position": ["local"],
            "get_target_position": ["local"],
            "get_surrounding_robots_info": ["local"],
            "get_surrounding_obstacles_info": ["local"],
            "get_all_robots_id": ["global"],
            "get_all_robots_initial_position": ["global"],
            "get_target_formation_points": ["global"],
            "get_initial_unexplored_areas": ["global"],
            "get_prey_initial_position": ["global"],
            "get_surrounding_unexplored_area": ["local"],
            "get_quadrant_target_position": ["global", "local"],
            "get_surrounding_environment_info": ["local"],
            # Example of APIs with multiple scopes
            # 'example_api': ['local', 'global'],
        }

        self.base_prompt = [self.apis[api] for api in self.base_apis]
        self.task_apis = {
            "bridging": ["stop_self"],
            "circling": ["stop_self"],
            "covering": ["get_environment_range", "stop_self"],
            "crossing": ["stop_self"],
            "encircling": ["get_prey_position", "get_prey_initial_position"],
            "exploration": [
                "get_initial_unexplored_areas",
                "get_environment_range",
                "stop_self",
            ],
            "flocking": ["get_environment_range", "get_self_velocity"],
            "clustering": ["get_quadrant_target_position"],
            "shaping": ["get_target_formation_points", "stop_self"],
            "pursuing": ["get_lead_position"],
        }

    def get_api_prompt(
        self, task_name: str = None, scope: str = None, only_names: bool = False
    ) -> str | list:
        """
        Get the prompt of the robot API.
        Parameters:
            task_name: str, the name of the task to get the prompt.
            scope: str, optional, 'global' or 'local' to filter APIs by scope.
                   Default is None, which means to get all the prompts.
            only_names: bool, optional, if True, only return the API names instead of full text.
                        Default is False, which returns the full text.
        Returns:
            str or list: The prompt of the robot API.
        """
        if task_name is None:
            all_apis = self.apis.keys()
            if scope:
                filtered_apis = [
                    api for api in all_apis if scope in self.api_scope.get(api, [])
                ]
            else:
                filtered_apis = all_apis

            if only_names:
                return "\n\n".join(filtered_apis)
            return "\n\n".join([self.apis[api] for api in filtered_apis])

        try:
            task_prompt = self.base_prompt.copy()
            specific_apis = [
                self.apis[api] for api in self.task_apis.get(task_name, [])
            ]
            task_prompt.extend(specific_apis)

            if scope:
                task_prompt = [
                    api
                    for api in task_prompt
                    if scope in self.api_scope.get(self.get_api_name(api), [])
                ]

            if only_names:
                return [self.get_api_name(api) for api in task_prompt]
            return "\n\n".join(task_prompt)
        except KeyError:
            traceback.print_exc()
            raise SystemExit(
                f"Error in get_api_prompt: Task name '{task_name}' not found. Current existing tasks: {list(self.task_apis.keys())}."
            )
        except Exception as e:
            raise SystemExit(
                f"Error in get_api_prompt: {e}, current existing apis: {self.apis.keys()},"
                f"input task name: {task_name}, expected name: {self.task_apis.get(task_name, [])}"
            )

    def get_api_name(self, api: str) -> str:
        """
        Helper method to retrieve API name from the function definition.
        """
        return next(
            (name for name, content in self.apis.items() if content == api), None
        )


robot_api = RobotApi(content=robot_api_prompt)

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "../../config/", "experiment_config.yaml")

with open(yaml_file_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
task_name = data["arguments"]["--run_experiment_name"]["default"][0]

GLOBAL_ROBOT_API = robot_api.get_api_prompt(task_name, scope="global")
LOCAL_ROBOT_API = robot_api.get_api_prompt(task_name, scope="local")

global_import_list = robot_api.get_api_prompt(
    task_name, scope="global", only_names=True
)
local_import_list = robot_api.get_api_prompt(task_name, scope="local", only_names=True)
local_import_list = (
    local_import_list.split("\n\n")
    if isinstance(local_import_list, str)
    else local_import_list
)
local_import_list.append("get_assigned_task")

ALLOCATOR_TEMPLATE = """

def get_assigned_task():
    '''
    Description: Get the task assigned to the robot by the global allocator. This function is executed only once when the global allocator runs,
    providing each robot with a specific task based on the global context. The global allocator gathers all the information about the environment
    and robots to determine the most efficient and collision-free task allocation. The task information varies depending on the type of task.
    Returns:
        {template}
    '''
"""

if __name__ == "__main__":
    for task_name in robot_api.task_apis.keys():
        print(f"Task name: {task_name}")
        print("Global APIs:")
        print(robot_api.get_api_prompt(task_name, scope="global", only_names=True))
        print("Local APIs:")
        print(robot_api.get_api_prompt(task_name, scope="local", only_names=True))
        print("=" * 50)
        print()
