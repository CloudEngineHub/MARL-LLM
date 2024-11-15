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

from .analyze_stage_prompt import (
    ANALYZE_SKILL_PROMPT_TEMPLATE,
    ANALYZE_CONSTRAINT_PROMPT_TEMPLATE,
    CONSTRAIN_TEMPLATE,
    FUNCTION_TEMPLATE,
)
from .env_description import ENV_DES
from .robot_api_prompt import (
    robot_api,
    GLOBAL_ROBOT_API,
    LOCAL_ROBOT_API,
    ALLOCATOR_TEMPLATE,
    global_import_list,
    local_import_list,
)
from .run_code_prompt import (
    DEBUG_PROMPT,
    CONTINUE_DEBUG_PROMPT,
    FEEDBACK_PROMPT_TEMPLATE,
    CONTINUE_FEEDBACK_PROMPT_TEMPLATE,
    GRAMMAR_CHECK_PROMPT_TEMPLATE,
)
from .task_description import TASK_DES
from .user_requirements import tasks, get_user_commands
from .video_critic_prompt import VIDEO_PROMPT_TEMPLATE, OUTPUT_TEMPLATE
from .base_prompt import Prompt
from .coding_stage_prompt import GLOBAL_RUN_OUTPUT_TEMPLATE

__all__ = [
    "ANALYZE_SKILL_PROMPT_TEMPLATE",
    "ANALYZE_CONSTRAINT_PROMPT_TEMPLATE",
    "CONSTRAIN_TEMPLATE",
    "FUNCTION_TEMPLATE",
    "ENV_DES",
    "robot_api",
    "GLOBAL_ROBOT_API",
    "LOCAL_ROBOT_API",
    "ALLOCATOR_TEMPLATE",
    "DEBUG_PROMPT",
    "CONTINUE_DEBUG_PROMPT",
    "Prompt",
    "global_import_list",
    "global_import_list",
    "FEEDBACK_PROMPT_TEMPLATE",
    "CONTINUE_FEEDBACK_PROMPT_TEMPLATE",
    "GRAMMAR_CHECK_PROMPT_TEMPLATE",
    "TASK_DES",
    "tasks",
    "VIDEO_PROMPT_TEMPLATE",
    "OUTPUT_TEMPLATE",
    "get_user_commands",
    "GLOBAL_RUN_OUTPUT_TEMPLATE",
]
