from __future__ import annotations

from core.arc_loader import serialize_task
from core.config import MODEL_OPTIONS, MODEL_REGISTRY
from core.executor import ask


def choose_reasoning(task: dict, r1: str, r2: str, memory_context: list[dict]) -> str:
    return ask(
        MODEL_REGISTRY["orchestrator"],
        f"""Task:
{serialize_task(task)}

Memory patterns:
{memory_context}

Option A:
{r1}

Option B:
{r2}

Pick the best reasoning or merge them.
Prefer a reusable, concrete ARC rule that can be turned into Python.
""",
        system="You are an ARC orchestrator that selects or merges candidate rules.",
        options=MODEL_OPTIONS["orchestrator"],
    )
