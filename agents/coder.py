from __future__ import annotations

import re

from core.config import MODEL_OPTIONS, MODEL_REGISTRY
from core.executor import ask_with_fallback
from core.prompts import CODER_SYSTEM_PROMPT


def _clean_code_block(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    if "def solve" in stripped:
        start = stripped.index("def solve")
        prefix = stripped[:start]
        if "import " in prefix:
            import_start = prefix.rfind("import ")
            return (prefix[import_start:] + stripped[start:]).strip()
    return stripped


def coding_agent(reasoning: str, focus: str | None = None, metric_v3_compliant: bool = False, minimize_graph: bool = True) -> str:
    optimization_hints = ""
    if metric_v3_compliant:
        optimization_hints += '''\n\n# METRIC V3 COMPLIANCE - STATIC SHAPES REQUIRED:
# - All tensor shapes must be explicitly defined at graph creation
# - No dynamic shape inference or symbolic dimensions allowed
# - Use fixed-size tensors only - no runtime shape calculations
'''
    if minimize_graph:
        optimization_hints += '''\n\n# MINIMAL COMPUTATIONAL GRAPH - OPTIMIZE FOR SCORE:
# Score formula: points = 25.0 - log(MACs + memory + params)
# To maximize score, MINIMIZE: MACs + memory + parameters\n#
# OPTIMIZATION STRATEGIES:
# 1. Use smallest possible weight matrices (reduce params)
# 2. Minimize intermediate tensor sizes (reduce memory)
# 3. Use fewest operations possible (reduce MACs)
# 4. Prefer simple operations: Add, Mul, Reshape over Conv, MatMul
# 5. Avoid unnecessary padding or redundant calculations
# 6. Use integer operations where possible (smaller than float32)
# 7. Eliminate dead code and unused variables
# 8. Prefer element-wise ops over reduction ops (fewer MACs)
# 9. Fuse operations when possible (e.g., Conv+ReLU together)
# 10. Use numpy efficiently - vectorized ops over loops\n#
# KEY INSIGHT: Small, correct networks score higher than large, complex ones!
# The 9538 exploit used 1.37M hidden constants - now impossible.
# Build TINY networks that just barely solve the task correctly.
'''
    extra_focus = f"\nExtra coding focus:\n{focus}\n" if focus else ""
    response = ask_with_fallback(
        MODEL_REGISTRY["coder"],
        f"""Convert this ARC reasoning into Python.

Requirements:
- use numpy
- expose a function named solve(grid)
- return a grid compatible with ARC JSON format
- OPTIMIZE FOR MINIMAL COMPUTATIONAL GRAPH (score = 25 - log(MACs + memory + params))
- return code only, with no markdown fences and no explanation

Reasoning:
{reasoning}
{optimization_hints}
{extra_focus}
""",
        system=CODER_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["coder"],
        fallback_models=[MODEL_REGISTRY["operator_fast"], MODEL_REGISTRY["orchestrator"]],
    )
    return _clean_code_block(response)
