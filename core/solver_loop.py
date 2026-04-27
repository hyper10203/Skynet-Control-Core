from __future__ import annotations

from agents.coder import coding_agent
from agents.critic import critic_agent
from agents.orchestrator import build_plan, refine_reasoning, synthesize_reasoning
from agents.reasoning import reasoning_primary, reasoning_secondary, reasoning_tertiary
from core.memory import load_memory, retrieve_patterns, store_pattern


def solve_task(task: dict, critique_rounds: int = 3) -> dict:
    memory = load_memory()
    patterns = retrieve_patterns(task, memory)

    try:
        plan = build_plan(task, patterns, critique_rounds=critique_rounds)
        routed_memory = patterns if plan.get("use_memory", True) else []

        r1 = ""
        if plan.get("call_primary_reasoner", True):
            r1 = reasoning_primary(task, focus=plan.get("primary_focus"))

        r2 = ""
        if plan.get("call_secondary_reasoner", False):
            r2 = reasoning_secondary(task, focus=plan.get("secondary_focus"))

        r3 = ""
        if plan.get("call_tertiary_reasoner", False):
            r3 = reasoning_tertiary(task, focus=plan.get("tertiary_focus"))

        if not r1 and not r2 and not r3 and plan.get("direct_reasoning"):
            reasoning = plan["direct_reasoning"]
        else:
            reasoning = synthesize_reasoning(
                task,
                routed_memory,
                plan,
                primary_reasoning=r1,
                secondary_reasoning=r2,
                tertiary_reasoning=r3,
            )

        critiques: list[str] = []
        best_code = coding_agent(reasoning, focus=plan.get("coder_focus"))

        for _ in range(min(plan.get("critic_rounds", 0), critique_rounds)):
            if not plan.get("call_critic", True):
                break
            critique = critic_agent(best_code, task=task, focus=plan.get("critic_focus"))
            critiques.append(critique)
            reasoning = refine_reasoning(task, reasoning, critique)
            best_code = coding_agent(reasoning, focus=plan.get("coder_focus"))

        store_pattern(task, reasoning, success=bool(best_code), code=best_code)

        return {
            "task_id": task.get("id"),
            "plan": plan,
            "patterns": patterns,
            "reasoning_primary": r1,
            "reasoning_secondary": r2,
            "reasoning_tertiary": r3,
            "merged_reasoning": reasoning,
            "critiques": critiques,
            "code": best_code,
        }
    except Exception as exc:
        store_pattern(task, f"failure: {exc}", success=False, code="")
        raise
