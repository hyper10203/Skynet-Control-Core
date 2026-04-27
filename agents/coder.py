from __future__ import annotations

import re

from core.config import MODEL_OPTIONS, MODEL_REGISTRY
from core.executor import ask
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


def coding_agent(reasoning: str, focus: str | None = None) -> str:
    extra_focus = f"\nExtra coding focus:\n{focus}\n" if focus else ""
    response = ask(
        MODEL_REGISTRY["coder"],
        f"""Convert this ARC reasoning into Python.

Requirements:
- use numpy
- expose a function named solve(grid)
- return a grid compatible with ARC JSON format
- keep the code short and readable
- return code only, with no markdown fences and no explanation

Reasoning:
{reasoning}
{extra_focus}
""",
        system=CODER_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["coder"],
    )
    return _clean_code_block(response)
