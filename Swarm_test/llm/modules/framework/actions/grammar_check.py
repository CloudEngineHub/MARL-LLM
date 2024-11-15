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
from modules.framework.action import ActionNode, AsyncNode
from modules.framework.actions import DebugError
from modules.framework.code import State, FunctionNode, FunctionTree
from modules.framework.code_error import Bug, Bugs
from modules.framework.context import WorkflowContext
from modules.framework.handler import BugLevelHandler
from modules.framework.parser import GrammarParser
from modules.utils import root_manager


class GrammarCheck(ActionNode):
    def __init__(
        self, skill_tree: FunctionTree, next_text: str = "", node_name: str = ""
    ):
        super().__init__(next_text, node_name)
        self.function_name = None
        self.function = None
        self._grammar_checker = GrammarParser()
        self._skill_tree = skill_tree

    def _build_prompt(self):
        pass

    def setup(self, function: FunctionNode):
        self.function_name = function.name
        self.function = function

    async def _run(self) -> str:
        if not self.function_name:
            logger.log("Error in GrammarCheck: function_name is not set", "error")
            raise SystemExit
        else:
            self._skill_tree.save_by_function(function=self.function_name)
            errors = self._grammar_checker.check_code_errors(
                file_path=f"{root_manager.workspace_root}/{self._skill_tree.name}.py"
            )
            return self._process_response(str(errors))

    def _process_response(self, response: str) -> str | Bugs | Bug:
        if eval(response):
            if self.function.grammar_check_times > 3:
                logger.log("Grammar check failed more than 3 times", "error")
                self.error_handler = None
                return ""

            error_code = self._skill_tree.save_by_function(
                function=self.function_name, save=False
            )

            bug_list = [
                Bug(
                    error_msg=e["error_message"],
                    error_function=e["function_name"],
                    error_code=error_code,
                )
                for e in eval(response)
            ]
            logger.log(
                f"Grammar check failed for function: {self.function_name}", "error"
            )
            return Bugs(bug_list, error_code=error_code)
        else:
            logger.log(
                f"Grammar check passed for function: {self.function_name}", "warning"
            )
            return response


class GrammarCheckAsync(AsyncNode):
    def __init__(
        self,
        skill_tree,
        run_mode="layer",
        start_state=State.REVIEWED,
        end_state=State.CHECKED,
    ):
        super().__init__(skill_tree, run_mode, start_state, end_state)

    def _build_prompt(self):
        pass

    async def operate(self, function: FunctionNode):
        action = GrammarCheck(self.skill_tree)
        action.error_handler = self.error_handler
        action.setup(function)
        function.grammar_check_times += 1
        await action.run()
        function.state = self._end_state

    async def _run_layer_mode(self):
        layer_index = self.skill_tree.get_min_layer_index_by_state(self._start_state)
        if layer_index == -1:
            logger.log(f"No functions in {self._start_state} state", "error")
            raise SystemExit

        for function_node in self.skill_tree.layers[layer_index].functions:
            await self.operate(function_node)

    async def _run_sequential_mode(self):
        for function in self.skill_tree.nodes:
            if function.state == self._start_state:
                await self.operate(function)

    async def _run(self):
        if self._run_mode == "layer":
            await self._run_layer_mode()
        elif self._run_mode == "sequential":
            await self._run_sequential_mode()
        elif self._run_mode == "parallel":
            await self._run_sequential_mode()
        else:
            logger.log("Unknown generate_mode", "error")
            raise SystemExit


if __name__ == "__main__":
    import asyncio

    context = WorkflowContext()
    path = "../../../workspace/test"
    context.load_from_file(f"{path}/reviewed_function.pkl")
    root_manager.update_root(path)
    debug_error = DebugError(context.global_skill_tree)

    grammar_check = GrammarCheckAsync(context.global_skill_tree)
    bug_handler = BugLevelHandler()
    bug_handler.next_action = debug_error
    grammar_check.error_handler = bug_handler
    asyncio.run(grammar_check.run())
    grammar_check.context.save_to_file(f"{path}/CodeReview.pkl")
