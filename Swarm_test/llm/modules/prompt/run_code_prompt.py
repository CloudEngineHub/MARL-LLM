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

DEBUG_PROMPT = """
## Background:
{task_des}

## Role setting:
-It's now the phase to run the code, your task is to find the erroneous part based on the compiler's traceback feedback, and modify it.

## These are the User original instructions:
{instruction}

## These are the environment description:
{env_des}


## These are the basic Robot APIs:
```python
{robot_api}
```
## These are the functions that mentioned in the error message:
{mentioned_functions}

## These are the error messages:
{error_message}

## Task
According to the error message, make modifications based on the existing foundation, and output the modified function in its entirety.
The output TEXT format is as follows:
Reasoning: what caused the error, and how did you fix it? Provide serval fixes and choose the best one.
code:
```python
import ...(if necessary else remove this line)


function_name(...):
    ...


...
```

## Notes:
- Only allowed to modify errors, not allowed to modify function names as well as the input and output of the function.
- Output the complete code of the entire function, not just a part of it that's been omitted.
- Rewrite all functions that need modifications.
- Keep the original code in the function as unchanged as possible, only modifying the parts that are incorrect.
- The provided basic Robot APIs have already been implemented. You cannot modify these functions; you can only call them.
- Do not raising errors or using assertions in the function body.
- Do not assume any part of the code,because the code will be executed directly without any modification and human intervention.
- All APIs and written functions can be called directly without any import.
- The output should be in the specified format.
""".strip()

CONTINUE_DEBUG_PROMPT = """
Based on the modifications you made earlier, issues still persist after execution. Here is the feedback result information:
"{error_message}"

## Task
According to the error message, make modifications based on the existing foundation, and output the modified function in its entirety.
The output TEXT format is as follows:
Reasoning:Why the previous modifications did not work? what caused the error, and how did you fix it? Provide serval fixes and choose the best one.
code:
```python
import ...(if necessary)


function_name(...):
    ...


...
```

## Notes:
- Only allowed to modify errors, not allowed to modify function names as well as the input and output of the function.
- Output the complete code of the entire function, not just a part of it that's been omitted.
- Rewrite all functions that need modifications.
- Keep the original code in the function as unchanged as possible, only modifying the parts that are incorrect.
- The provided basic Robot APIs have already been implemented. You cannot modify these functions; you can only call them.
- The output should be in the specified format.
""".strip()

FEEDBACK_PROMPT_TEMPLATE = """
## Background:
{task_des}

## These are the User original instructions:
{instruction}

## Role setting:
- After running the code, there are some issues. You need to modify based on the feedback from users.

## These are the environment description:
{env_des}


## These are the basic Robot APIs:
There are two types of APIs: local and global.
where local APIs can only be called by the robot itself, and global APIs can be called by an centralized allocator.

### local APIs:
```python
{local_api}
```

### global APIs:
```python
{global_api}
```

## These are the functions that can be modified:
There are two types of functions: local and global.
where local functions can only be called by the robot itself, and global functions can be called by an centralized allocator.

### local functions:
```python
{local_functions}
```

### global functions:
```python
{global_functions}
```

## These are the user's feedback:
{feedback}

## Output format:
### Reasoning: you should analyze in the step:
    - What is the problem?
    - Which part of the code is problematic?
    - Any global functions that need modification?
    - Any local functions that need modification?
    - What is the solution?
### Code:
```python
function_name(...):
    import ...(if necessary)
    ...
# possible other functions
...
```
## Notes:
- Do not modify the function name.
- The input and output of the function could be modified.But you should make sure other functions that call this function can still work.
- Global functions can't call local functions,and local functions can't call global functions.
- If any third-party libraries are needed, you can import them in the function.Don't import them outside the function.
- Apis and written functions can be called directly.
- Output the complete code of the entire function, not just a part of it that's been omitted.
- Don't create new functions.
- Only output the functions that need modification, with as few changes as possible.
- Rewrite all functions that need modifications.
- Keep the original code in the function as unchanged as possible, only modifying the parts that are incorrect.
- The output should be in the specified format.
""".strip()

CONTINUE_FEEDBACK_PROMPT_TEMPLATE = """
Based on the modifications you made earlier, issues still persist after execution. Here is the feedback result information:
"{feedback}"
Continue making modifications.

## Output format:
### Reasoning: you should analyze in the step:
    - Why previous modifications did not work?
    - What is the problem?
    - Which part of the code is problematic?
    - What is the solution?
### Code:
```python
import ...(if necessary)

function_name(...):
    ...

## Notes:
- Do not modify the function name.
- The input and output of the function could be modified.But you should make sure other functions that call this function can still work.
- Output the complete code of the entire function, not just a part of it that's been omitted.
- Rewrite all functions that need modifications.
- Keep the original code in the function as unchanged as possible, only modifying the parts that are incorrect.
- The output should be in the specified format.
""".strip()

GRAMMAR_CHECK_PROMPT_TEMPLATE = """
## Background:
{task_des}

## Role setting:
- The code you wrote encountered issues during syntax checking. Now, you need to modify the code according to the problems identified to meet the requirements.

## These are the environment description:
{env_des}

## These are the basic Robot APIs:
```python
{robot_api}
```

## These are the functions:
```python
{functions}
```

## These are the error messages:
{error}

## Output format:
### Reasoning: you should analyze in the step:
    - What is the problem?
    - How was the problem generated?
    - What is the solution?

### Code:
```python
import ...(if necessary)

function_name(...):
    ...

## Notes:
- Do not modify the function name.
- The input and output of the function could be modified.But you should make sure other functions that call this function can still work.
- Output the complete code of the entire function, not just a part of it that's been omitted.
- Rewrite all functions that need modifications.
- Keep the original code in the function as unchanged as possible, only modifying the parts that are incorrect.
- The output should be in the specified format.
""".strip()
