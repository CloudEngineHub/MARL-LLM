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
from modules.file import logger
from modules.framework.action import ActionNode
from modules.framework.code import FunctionTree
from modules.framework.code_error import CodeError, Bug, Bugs
from modules.framework.parser import parse_text, CodeParser
from modules.llm import GPT
from modules.prompt import (
    DEBUG_PROMPT,
    CONTINUE_DEBUG_PROMPT,
    ALLOCATOR_TEMPLATE,
    GLOBAL_ROBOT_API,
    LOCAL_ROBOT_API,
    ENV_DES,
    TASK_DES,
)


class DebugError(ActionNode):
    def __init__(self, next_text="", node_name=""):
        super().__init__(next_text, node_name)
        self.__llm = GPT(memorize=True)
        self.error = None
        self.error_func = None
        self._skill_tree = None

    def setup(self, error: CodeError | Bugs | Bug):
        self.error = error.error_msg
        if len(self.error) > 1000:
            self.error = self.error[:1000]
            logger.log("error message is too long, only show the first 1000 characters", level="error")
        self.error_func = error.error_code
        self._skill_tree = (
            self.context.local_skill_tree
            if self.context.scoop == "local"
            else self.context.global_skill_tree
        )

    def _build_prompt(self):
        if len(self.context.global_skill_tree.layers) == 0:
            local_api_prompt = LOCAL_ROBOT_API
        else:
            local_api_prompt = LOCAL_ROBOT_API + ALLOCATOR_TEMPLATE.format(
                template=self.context.global_skill_tree.output_template
            )
        robot_api = (
            GLOBAL_ROBOT_API if self.context.scoop == "global" else local_api_prompt
        )
        # if self._call_times == 0:
        self.prompt = DEBUG_PROMPT.format(
            task_des=TASK_DES,
            instruction=self.context.command,
            robot_api=robot_api,
            env_des=ENV_DES,
            mentioned_functions=self.error_func,
            error_message=self.error,
        )
        # else:
        #     self.prompt = CONTINUE_DEBUG_PROMPT.format(
        #         error_message=self.error,
        #     )
        # self._call_times += 1

    async def _process_response(self, response: str, **kwargs) -> str:
        code = parse_text(text=response)
        parser = CodeParser()
        parser.parse_code(code)
        self._skill_tree.update_from_parser(parser.imports, parser.function_dict)
        self._skill_tree.save_functions_to_file()

        return str(code)
