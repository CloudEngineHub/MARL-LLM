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

# from sympy.printing.dot import template

from modules.framework.action import ActionNode
from modules.framework.code import FunctionTree, State
from modules.framework.parser import SingleFunctionParser, parse_text
from modules.prompt import (
    GLOBAL_ROBOT_API,
    LOCAL_ROBOT_API,
    ALLOCATOR_TEMPLATE,
    GLOBAL_RUN_OUTPUT_TEMPLATE,
    ENV_DES,
    TASK_DES,
)
from modules.utils import root_manager


class WriteRun(ActionNode):
    def __init__(self, skill_tree: FunctionTree, next_text="", node_name=""):
        super().__init__(next_text, node_name)
        self._skill_tree = skill_tree

    def _build_prompt(self):
        functions = "\n\n".join(self._skill_tree.function_valid_content)
        if len(self.context.global_skill_tree.layers) == 0:
            local_api_prompt = LOCAL_ROBOT_API
        else:
            local_api_prompt = LOCAL_ROBOT_API + ALLOCATOR_TEMPLATE.format(
                template=self.context.global_skill_tree.output_template
            )
        robot_api = (
            GLOBAL_ROBOT_API if self.context.scoop == "global" else local_api_prompt
        )
        self.desired_function_name = (
            "allocate_run" if self.context.scoop == "global" else "run_loop"
        )
        self.prompt = self.prompt.format(
            task_des=TASK_DES,
            instruction=self.context.command,
            env_des=ENV_DES,
            robot_api=robot_api,
            functions=functions,
            template=GLOBAL_RUN_OUTPUT_TEMPLATE,
        )

    async def _process_response(self, response: str) -> str:
        code = parse_text(text=response)
        parser = SingleFunctionParser()
        parser.parse_code(code)
        parser.check_function_name(self.desired_function_name)

        self._skill_tree.update_from_parser(parser.imports, parser.function_dict)
        self._skill_tree.save_code([self.desired_function_name])
        self._skill_tree[self.desired_function_name].state = State.WRITTEN

        if self._skill_tree.name == "global_skill":
            template = eval(parse_text(response, lang="json"))["value"]

            self._skill_tree.output_template = (
                f"type: {template['type']}\n"
                f"description: {template['description']}\n"
            )


if __name__ == "__main__":
    import asyncio
    from modules.framework.context import WorkflowContext

    context = WorkflowContext()
    path = "../../../workspace/test"
    context.load_from_file(f"{path}/reviewed_function.pkl")
    root_manager.update_root("../../../workspace/test")

    code_reviewer = WriteRun(
        context.global_skill_tree,
    )
    asyncio.run(code_reviewer.run())
    context.save_to_file("../../../workspace/test/write_run.pkl")
