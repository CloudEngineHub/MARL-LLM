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

from os import listdir

from modules.file import logger
from modules.framework.action import ActionNode
from modules.framework.code import FunctionTree
from modules.framework.code_error import Feedback
from modules.framework.constraint import ConstraintPool
from modules.framework.parser import parse_text
from modules.prompt import (
    VIDEO_PROMPT_TEMPLATE,
    OUTPUT_TEMPLATE,
    TASK_DES,
)
from modules.utils import process_video, create_video_from_frames


class VideoCriticize(ActionNode):
    def __init__(self, next_text: str = "", node_name: str = ""):
        super().__init__(next_text, node_name)

        self._frames: list
        self._function_pool = FunctionTree()
        self._constraint_pool = ConstraintPool()

    def _build_prompt(self):
        self.setup()

        self.prompt = [
            VIDEO_PROMPT_TEMPLATE.format(
                task_des=TASK_DES,
                command=self.context.command,
                feedback="/n".join(self.context.feedbacks),
                constraint=str(self._constraint_pool),
                out_put=OUTPUT_TEMPLATE,
            ),
            *map(
                lambda x: {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"},
                },
                self._frames,
            ),
        ]
        pass

    def setup(self):
        from modules.utils import root_manager

        number = len(listdir(f"{root_manager.data_root}/frames")) - 1
        video_path = f"{root_manager.data_root}/output{number}.mp4"
        self._frames = process_video(video_path, end_time=10, seconds_per_frame=1)
        create_video_from_frames(
            self._frames, output_path=f"{root_manager.data_root}/extra{number}.mp4"
        )

    async def _process_response(self, response: str) -> str | Feedback:
        response = parse_text(text=response, lang="json")
        result = eval(response)
        if result["result"].strip().lower() == "success":
            # HumanFeedback
            if_feedback = input("If task is done? Press y/n")
            if if_feedback == "y":
                logger.log("run code:success", "warning")
                return "NONE"
            else:
                if self.context.args.feedback == "None":
                    logger.log("run code:fail", "warning")
                    return "NONE"
                feedback = input("Please provide feedback:")
                self.context.feedbacks.append(feedback)
                return Feedback(feedback)

        elif result["result"].strip().lower() == "fail":
            return Feedback(result["feedback"])
        else:
            logger.log(f"Invalid result: {result}", "error")
            raise Exception("Invalid result")

    async def _run(self) -> str:
        sim = """```json
        {
        "result": "SUCCESS",
        "feedback": "..."
        }```
        """
        res = await self._process_response(response=sim)
        return res


if __name__ == "__main__":
    import asyncio
    from modules.utils import root_manager

    path = "../../../workspace/2024-05-26_17-19-56"
    root_manager.update_root(path)
    function_analyser = VideoCriticize("analyze constraints")

    function_analyser.context.load_from_file(f"{path}/RunCodeAsync.pkl")
    asyncio.run(function_analyser.run())
    function_analyser.context.save_to_file(f"{path}/analyze_functions.pkl")
