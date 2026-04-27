from __future__ import annotations

from core.arc_loader import serialize_task
from core.config import MODEL_OPTIONS, MODEL_REGISTRY
from core.executor import ask
from core.prompts import (
    PRIMARY_REASONER_SYSTEM_PROMPT,
    SECONDARY_REASONER_SYSTEM_PROMPT,
    TERTIARY_REASONER_SYSTEM_PROMPT,
)


def reasoning_primary(task: dict, focus: str | None = None) -> str:
    extra_focus = f"\nExtra focus:\n{focus}\n" if focus else ""
    return ask(
        MODEL_REGISTRY["reasoning_primary"],
        f"""Solve this ARC task step-by-step.

Task:
{serialize_task(task)}

Give a concise transformation rule and mention reusable pattern ideas.
{extra_focus}
""",
        system=PRIMARY_REASONER_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["reasoning_primary"],
    )


def reasoning_secondary(task: dict, focus: str | None = None) -> str:
    extra_focus = f"\nExtra focus:\n{focus}\n" if focus else ""
    return ask(
        MODEL_REGISTRY["reasoning_secondary"],
        f"""Find a different transformation rule for this ARC task.

Task:
{serialize_task(task)}

Prefer an alternative explanation, not a restatement.
{extra_focus}
""",
        system=SECONDARY_REASONER_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["reasoning_secondary"],
    )


def reasoning_tertiary(task: dict, focus: str | None = None) -> str:
    extra_focus = f"\nExtra focus:\n{focus}\n" if focus else ""
    return ask(
        MODEL_REGISTRY["reasoning_tertiary"],
        f"""Provide a third-pass ARC explanation that complements the other reasoners.

Task:
{serialize_task(task)}

Look for edge cases, hidden constraints, and alternative implementation shortcuts.
{extra_focus}
""",
        system=TERTIARY_REASONER_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["reasoning_tertiary"],
    )
