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

import time

from modules.file import logger
from modules.framework.action import ActionNode, ActionLinkedList
from modules.framework.code import FunctionTree, State
from modules.framework.context import WorkflowContext
from modules.framework.handler import BugLevelHandler
from modules.utils import root_manager


class GenerateFunctions(ActionNode):
    def __init__(self, next_text: str = "", node_name: str = "", run_mode="layer"):
        from modules.framework.actions import (
            DesignFunctionAsync,
            WriteFunctionsAsync,
            CodeReviewAsync,
            WriteRun,
            DebugError,
            GrammarCheckAsync,
        )

        super().__init__(next_text, node_name)
        self._cost_time = 0
        self.skill_tree = (
            self.context.global_skill_tree
            if self.context.scoop == "global"
            else self.context.local_skill_tree
        )
        logger.log(f"Generate {self.context.scoop} functions", "warning")
        design_functions = DesignFunctionAsync(self.skill_tree, run_mode)
        write_functions = WriteFunctionsAsync(self.skill_tree, run_mode)
        grammar_check = GrammarCheckAsync(self.skill_tree, run_mode)

        code_review = CodeReviewAsync(self.skill_tree, run_mode)
        bug_handler = BugLevelHandler()
        write_run = WriteRun(self.skill_tree)
        debug_error = DebugError(
            self.skill_tree,
        )
        bug_handler.next_action = debug_error
        grammar_check.error_handler = bug_handler
        debug_error._next = grammar_check
        # link actions
        self._actions = ActionLinkedList("Generate Functions", design_functions)
        self._actions.add(write_functions)
        self._actions.add(code_review)
        self._actions.add(grammar_check)

        self._write_run = ActionLinkedList("Write Run", write_run)

        # self._write_run.add(code_review)
        # self._write_run.add(grammar_check)

    def _build_prompt(self):
        pass

    def _process_response(self, response: str) -> str:
        pass

    async def _run(self) -> str:
        start_time = time.time()
        finish = False
        while not finish:
            time.sleep(1)
            await self._actions.run_internal_actions()
            finish = all(node.state == State.CHECKED for node in self.skill_tree.nodes)
        await self._write_run.run_internal_actions()
        end_time = time.time()
        self._cost_time = end_time - start_time
        # code_analyzer = CodeAnalyzer(code=self._function_pool.file.read())
        # self._average_complexity, self._mi_score = code_analyzer.analyze()
        logger.log(
            f"Generate {self.context.scoop} functions cost time: {self._cost_time}",
            "warning",
        )
        # logger.log(f"average complexity: {self._average_complexity}, mi score: {self._mi_score}", "warning")
        logger.log(f"All {self.context.scoop} functions are generated", "warning")
        if self.context.scoop == "global":
            self.context.scoop = "local"
            self._next = GenerateFunctions()


if __name__ == "__main__":
    import asyncio

    context = WorkflowContext()
    path = "../../../workspace/test"
    root_manager.update_root("../../../workspace/test")
    context.load_from_file(f"{path}/AnalyzeSkills.pkl")
    generate_function = GenerateFunctions(context.global_skill_tree)

    asyncio.run(generate_function.run())
    generate_function.context.save_to_file(f"{path}/analyze_functions.pkl")
