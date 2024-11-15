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

import json
import numpy as np

from modules.file import logger
from modules.framework.action import ActionNode
from modules.llm import GPT
from modules.prompt import (
    CRITIC_TEMPLATE,
    ENV_DES,
    TASK_DES,
)
from modules.framework.parser import *
from modules.prompt.user_requirements import get_user_commands
from modules.utils import root_manager


class RLCritic(ActionNode):
    def __init__(self, next_text, node_name=""):
        super().__init__(next_text, node_name)
        self._interaction_mode = False
        if (
            hasattr(self.context.args, "interaction_mode")
            and self.context.args.interaction_mode is True
        ):
            self.__llm = GPT(memorize=True)
            self._interaction_mode = True
        else:
            self.__llm = GPT()
        self.reward_list = []
        self.state_value_list = []

    def _build_prompt(self, observation_input):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        observation_str = str(observation_input)
        self.prompt = self.prompt.format(
            observation=observation_str,
            task_des=TASK_DES,
            output_template=CRITIC_TEMPLATE,
        )

    async def _process_response(self, response: str) -> str:
        content = parse_text(response, "json")
        self.reward_list = eval(content)["critic_output"][0]["reward"]
        self.state_value_list = eval(content)["critic_output"][1]["state_value"]
        # logger.log(f"Analyze Constraints Success", "success")


if __name__ == "__main__":
    import asyncio
    from modules.framework.context import WorkflowContext
    import argparse

    obs = np.random.rand(192, 10)

    root_manager.update_root("./workspace/test")

    parser = argparse.ArgumentParser(
        description="Run simulation with custom parameters."
    )

    parser.add_argument(
        "--interaction_mode",
        type=bool,
        default=False,
        help="Whether to run in interaction mode in analyze constraints.",
    )
    context = WorkflowContext()
    task = get_user_commands("formation")[0]

    context.command = task
    args = parser.parse_args()
    context.args = args
    rl_critic = RLCritic("rl critic")

    asyncio.run(rl_critic.run(obs))
    context.save_to_file("./workspace/test/constraint.pkl")
