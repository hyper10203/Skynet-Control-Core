from __future__ import annotations

from core.arc_loader import serialize_task
from core.config import MODEL_OPTIONS, MODEL_REGISTRY
from core.executor import ask_with_fallback
from core.prompts import CRITIC_SYSTEM_PROMPT


def critic_agent(code: str, task: dict | None = None, focus: str | None = None) -> str:
    task_block = f"\nTask:\n{serialize_task(task)}\n" if task else ""
    extra_focus = f"\nExtra critic focus:\n{focus}\n" if focus else ""
    return ask_with_fallback(
        MODEL_REGISTRY["critic"],
        f"""Find weaknesses in this ARC solution.

Focus on:
- hidden edge cases
- wrong assumptions
- shapes, colors, symmetry, object grouping
- how to make the rule more robust

{task_block}
Code:
{code}
{extra_focus}
""",
        system=CRITIC_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["critic"],
        fallback_models=[MODEL_REGISTRY["orchestrator"], MODEL_REGISTRY["operator_fast"]],
    )
