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

import radon.complexity as cc
from radon.metrics import mi_visit


class CodeAnalyzer:
    """
    CodeAnalyzer 类用于分析给定代码的复杂度和可维护性。
    """

    def __init__(self, code: str):
        self.code = code
        from modules.framework.parser import CodeParser

        self.parser = CodeParser()
        self.parser.parse_code(code)

    def calculate_cyclomatic_complexity(self):
        blocks = cc.cc_visit(self.code)
        complexities = [block.complexity for block in blocks]
        if complexities:
            average_complexity = sum(complexities) / len(complexities)
        else:
            average_complexity = 0
        return average_complexity

    def calculate_maintainability_index(self):
        mi_score = mi_visit(self.code, False)
        return mi_score

    def count_lines_and_comments(self):
        total_lines = 0
        comment_lines = 0
        code_lines = 0
        in_multiline_comment = False
        lines = self.code.split("\n")
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            total_lines += 1
            if in_multiline_comment:
                comment_lines += 1
                if stripped_line.endswith("'''") or stripped_line.endswith('"""'):
                    in_multiline_comment = False
            elif stripped_line.startswith("#"):
                comment_lines += 1
            elif stripped_line.startswith("'''") or stripped_line.startswith('"""'):
                comment_lines += 1
                if not (
                    stripped_line.endswith("'''")
                    or stripped_line.endswith('"""')
                    and len(stripped_line) > 3
                ):
                    in_multiline_comment = True
            else:
                code_lines += 1

        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        return total_lines, code_lines, comment_lines, comment_ratio

    def analyze_functions(self):
        function_stats = {}
        for func_name, func_code in self.parser.function_dict.items():
            func_analyzer = CodeAnalyzer(func_code)
            (
                total_lines,
                code_lines,
                comment_lines,
                comment_ratio,
            ) = func_analyzer.count_lines_and_comments()
            function_stats[func_name] = {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "comment_ratio": comment_ratio,
            }
        return function_stats

    def analyze(self):
        avg_complexity = self.calculate_cyclomatic_complexity()
        mi_score = self.calculate_maintainability_index()
        (
            total_lines,
            code_lines,
            comment_lines,
            comment_ratio,
        ) = self.count_lines_and_comments()
        function_stats = self.analyze_functions()
        return {
            "average_complexity": avg_complexity,
            "maintainability_index": mi_score,
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "comment_ratio": comment_ratio,
            "function_stats": function_stats,
        }


if __name__ == "__main__":
    # Example usage
    code = """
def example_function(x):
    # This is a comment
    '''

    '''
    if x > 0:
        return x
    else:
        return -x

def another_function(y):
    \"\"\"This is a docstring comment\"\"\"
    for i in range(y):
        print(i)
    """
    analyzer = CodeAnalyzer(code)
    analysis_result = analyzer.analyze()
    print(analysis_result)
