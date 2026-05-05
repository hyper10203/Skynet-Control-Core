"""Teacher-to-student workflow for NeuroGolf graph distillation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import SKYNET_DISTILLATION_PLAN_PATH


DEFAULT_STAGES = [
    {
        "id": "teacher_rule_search",
        "status": "ready",
        "goal": "Use the largest local/reasoning model to infer symbolic task rules from train, test, and ARC-GEN examples.",
    },
    {
        "id": "candidate_graph_generation",
        "status": "pending",
        "goal": "Emit multiple legal static ONNX candidate graphs per task: identity, color map, affine geometry, lookup, and tiny conv fallbacks.",
    },
    {
        "id": "student_distillation",
        "status": "pending",
        "goal": "Distill teacher behavior into the smallest static student graph, preferring rule graphs over learned weights.",
    },
    {
        "id": "local_validation",
        "status": "pending",
        "goal": "Run local ONNX validation, static shape inference, banned-op checks, and cost profiling before any submit.",
    },
    {
        "id": "cost_regression",
        "status": "pending",
        "goal": "Keep the lower-cost valid graph per task and reject changes that increase MACs, memory, or params without solving more examples.",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_distillation_plan() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _now(),
        "mode": "teacher_search_then_static_student",
        "objective": "Solve each NeuroGolf task with the smallest correct static ONNX graph.",
        "teacher_model": "",
        "student_target": "static_onnx_graph_per_task",
        "fine_tune_policy": {
            "enabled": False,
            "purpose": "Improve rule inference and graph-template selection, not submit a large model.",
            "dataset_path": "",
            "output_adapter_path": "",
        },
        "distillation_policy": {
            "prefer_symbolic_graphs": True,
            "allow_tiny_conv_fallback": True,
            "max_candidate_graphs_per_task": 12,
            "reject_dynamic_shapes": True,
            "reject_banned_ops": True,
            "optimize_for": ["correctness", "params", "memory", "macs"],
        },
        "stages": DEFAULT_STAGES,
        "teacher_signals": [],
    }


def load_distillation_plan() -> dict[str, Any]:
    if not SKYNET_DISTILLATION_PLAN_PATH.exists():
        return default_distillation_plan()
    try:
        payload = json.loads(SKYNET_DISTILLATION_PLAN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_distillation_plan()
    if not isinstance(payload, dict):
        return default_distillation_plan()
    payload.setdefault("stages", DEFAULT_STAGES)
    payload.setdefault("teacher_signals", [])
    return payload


def save_distillation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan = dict(plan)
    plan["updated_at"] = _now()
    SKYNET_DISTILLATION_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKYNET_DISTILLATION_PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def initialize_distillation_plan(*, teacher_model: str = "", dataset_path: str = "") -> dict[str, Any]:
    plan = default_distillation_plan()
    plan["teacher_model"] = teacher_model.strip()
    plan["fine_tune_policy"]["dataset_path"] = dataset_path.strip()
    return save_distillation_plan(plan)


def set_fine_tune_policy(*, enabled: bool, dataset_path: str, output_adapter_path: str, teacher_model: str) -> dict[str, Any]:
    plan = load_distillation_plan()
    plan["teacher_model"] = teacher_model.strip()
    plan["fine_tune_policy"] = {
        "enabled": bool(enabled),
        "purpose": "Improve rule inference and graph-template selection, not submit a large model.",
        "dataset_path": dataset_path.strip(),
        "output_adapter_path": output_adapter_path.strip(),
    }
    return save_distillation_plan(plan)


def record_teacher_signal(task_id: str, rule_summary: str, graph_hint: str, source: str = "operator") -> dict[str, Any]:
    plan = load_distillation_plan()
    signal = {
        "created_at": _now(),
        "task_id": task_id.strip(),
        "rule_summary": rule_summary.strip(),
        "graph_hint": graph_hint.strip(),
        "source": source.strip() or "operator",
    }
    plan.setdefault("teacher_signals", []).append(signal)
    plan["teacher_signals"] = plan["teacher_signals"][-500:]
    return save_distillation_plan(plan)


def distillation_status() -> dict[str, Any]:
    plan = load_distillation_plan()
    stages = plan.get("stages", [])
    ready = sum(1 for stage in stages if str(stage.get("status")) in {"ready", "done"})
    return {
        "plan_path": str(SKYNET_DISTILLATION_PLAN_PATH),
        "teacher_model": plan.get("teacher_model", ""),
        "fine_tune_enabled": bool((plan.get("fine_tune_policy") or {}).get("enabled")),
        "stage_count": len(stages),
        "ready_or_done_stages": ready,
        "teacher_signal_count": len(plan.get("teacher_signals", [])),
        "updated_at": plan.get("updated_at"),
        "plan": plan,
    }
