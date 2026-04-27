from __future__ import annotations

import json
import re

from core.arc_loader import serialize_task
from core.config import MODEL_OPTIONS, MODEL_REGISTRY, PREFER_DIRECT_ORCHESTRATION
from core.executor import ask
from core.prompts import ORCHESTRATOR_SYSTEM_PROMPT


def _memory_digest(memory_context: list[dict]) -> str:
    if not memory_context:
        return "[]"
    digest = []
    for item in memory_context[:3]:
        digest.append(
            {
                "keywords": item.get("keywords", []),
                "reasoning": str(item.get("reasoning", ""))[:220],
                "success": item.get("success", True),
            }
        )
    return json.dumps(digest, indent=2)


def _extract_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value is None:
        return default
    return bool(value)


def build_plan(task: dict, memory_context: list[dict], critique_rounds: int = 3) -> dict:
    raw = ask(
        MODEL_REGISTRY["orchestrator"],
        f"""You are the top-level ARC orchestrator. The user talks only to you.

Task:
{serialize_task(task)}

Memory patterns:
{_memory_digest(memory_context)}

Return JSON only with these keys:
- task_summary: short string
- use_memory: true/false
- call_primary_reasoner: true/false
- call_secondary_reasoner: true/false
- call_tertiary_reasoner: true/false
- call_coder: true/false
- call_critic: true/false
- critic_rounds: integer from 0 to {critique_rounds}
- primary_focus: short string
- secondary_focus: short string
- tertiary_focus: short string
- coder_focus: short string
- critic_focus: short string
- direct_reasoning: short string

Rules:
- If the task looks straightforward, you may skip extra reasoners and provide direct_reasoning.
- Only wake extra models when they add value.
- Keep the plan compact and practical.
""",
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["orchestrator"],
    )
    parsed = _extract_json_object(raw)
    direct_reasoning = str(parsed.get("direct_reasoning", "")).strip()
    if not parsed and raw.strip():
        direct_reasoning = raw.strip()
    plan = {
        "task_summary": str(parsed.get("task_summary", task.get("id", "task"))),
        "use_memory": _as_bool(parsed.get("use_memory", True), True),
        "call_primary_reasoner": _as_bool(parsed.get("call_primary_reasoner", False), False),
        "call_secondary_reasoner": _as_bool(parsed.get("call_secondary_reasoner", False), False),
        "call_tertiary_reasoner": _as_bool(parsed.get("call_tertiary_reasoner", False), False),
        "call_coder": _as_bool(parsed.get("call_coder", True), True),
        "call_critic": _as_bool(parsed.get("call_critic", True), True),
        "critic_rounds": max(0, min(int(parsed.get("critic_rounds", critique_rounds or 0)), critique_rounds)),
        "primary_focus": str(parsed.get("primary_focus", "")).strip(),
        "secondary_focus": str(parsed.get("secondary_focus", "")).strip(),
        "tertiary_focus": str(parsed.get("tertiary_focus", "")).strip(),
        "coder_focus": str(parsed.get("coder_focus", "")).strip(),
        "critic_focus": str(parsed.get("critic_focus", "")).strip(),
        "direct_reasoning": direct_reasoning,
        "raw_plan": raw,
    }
    if not plan["call_primary_reasoner"] and not plan["call_secondary_reasoner"] and not plan["call_tertiary_reasoner"] and not plan["direct_reasoning"]:
        plan["call_primary_reasoner"] = True
    if PREFER_DIRECT_ORCHESTRATION and plan["direct_reasoning"]:
        plan["call_primary_reasoner"] = False
        plan["call_secondary_reasoner"] = False
        plan["call_tertiary_reasoner"] = False
    if not plan["call_coder"]:
        plan["call_coder"] = True
    return plan


def synthesize_reasoning(
    task: dict,
    memory_context: list[dict],
    plan: dict,
    primary_reasoning: str = "",
    secondary_reasoning: str = "",
    tertiary_reasoning: str = "",
) -> str:
    return ask(
        MODEL_REGISTRY["orchestrator"],
        f"""You are finalizing the reasoning for one ARC task.

Task:
{serialize_task(task)}

Task summary:
{plan.get("task_summary", "")}

Relevant memory:
{_memory_digest(memory_context)}

Your own direct reasoning hint:
{plan.get("direct_reasoning", "")}

Primary reasoner:
{primary_reasoning or "[not used]"}

Secondary reasoner:
{secondary_reasoning or "[not used]"}

Third reasoner:
{tertiary_reasoning or "[not used]"}

Write one final, concrete transformation rule for the coder.
Merge complementary evidence from all useful voices. Make it exact, concise, and implementation-ready.
""",
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["orchestrator"],
    )


def refine_reasoning(task: dict, current_reasoning: str, critique: str) -> str:
    return ask(
        MODEL_REGISTRY["orchestrator"],
        f"""Revise the ARC reasoning after critique.

Task:
{serialize_task(task)}

Current reasoning:
{current_reasoning}

Critique:
{critique}

Return a corrected reasoning only.
""",
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        options=MODEL_OPTIONS["orchestrator"],
    )
