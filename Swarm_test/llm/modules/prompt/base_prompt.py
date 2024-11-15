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

from modules.prompt.analyze_stage_prompt import (
    ANALYZE_CONSTRAINT_PROMPT_TEMPLATE,
    CONSTRAIN_TEMPLATE,
    ANALYZE_SKILL_PROMPT_TEMPLATE,
    FUNCTION_TEMPLATE,
)
from modules.prompt.design_stage_prompt import (
    DESIGN_GLOBAL_FUNCTION_PROMPT_TEMPLATE,
    DESIGN_LOCAL_FUNCTION_PROMPT_TEMPLATE,
)
from modules.prompt.coding_stage_prompt import (
    WRITE_GLOBAL_FUNCTION_PROMPT_TEMPLATE,
    WRITE_LOCAL_FUNCTION_PROMPT_TEMPLATE, 
)
from modules.prompt.code_review_stage_prompt import (
    GLOBAL_CODE_REVIEW_PROMPT_TEMPLATE,
    LOCAL_CODE_REVIEW_PROMPT_TEMPLATE,
)
from modules.prompt.coding_stage_prompt import (
    WRITE_GLOBAL_RUN_PROMPT_TEMPLATE,
    WRITE_LOCAL_RUN_PROMPT_TEMPLATE,
    GLOBAL_RUN_OUTPUT_TEMPLATE,
)

from modules.prompt.rl_critic_prompt import (
    REWARD_STATEVALUE_CRITIC_PROMPT_TEMPLATE,
    CRITIC_TEMPLATE,
)


class Prompt:
    def __init__(
        self,
    ):
        self.action_map: dict = {
            "AnalyzeConstraints": ANALYZE_CONSTRAINT_PROMPT_TEMPLATE,
            "AnalyzeSkills": ANALYZE_SKILL_PROMPT_TEMPLATE,
            "DesignFunction": {
                "global": DESIGN_GLOBAL_FUNCTION_PROMPT_TEMPLATE,
                "local": DESIGN_LOCAL_FUNCTION_PROMPT_TEMPLATE,
            },
            "WriteFunction": {
                "global": WRITE_GLOBAL_FUNCTION_PROMPT_TEMPLATE,
                "local": WRITE_LOCAL_FUNCTION_PROMPT_TEMPLATE,
            },
            "CodeReview": {
                "global": GLOBAL_CODE_REVIEW_PROMPT_TEMPLATE,
                "local": LOCAL_CODE_REVIEW_PROMPT_TEMPLATE,
            },
            "WriteRun": {
                "global": WRITE_GLOBAL_RUN_PROMPT_TEMPLATE,
                "local": WRITE_LOCAL_RUN_PROMPT_TEMPLATE,
            },
            "RLCritic": REWARD_STATEVALUE_CRITIC_PROMPT_TEMPLATE
        }

    def get_prompt(self, action: str, scope: str) -> str:
        if action not in self.action_map:
            return ""
        if action == "AnalyzeSkills" or action == "AnalyzeConstraints" or action == "RLCritic":
            return self.action_map[action]

        return self.action_map[action][scope]
