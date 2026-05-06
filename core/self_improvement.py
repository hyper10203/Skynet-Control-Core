"""Adaptive learning loop for the ARC/NeuroGolf agent.

This module keeps the autonomy system learning from recent cycle results
without letting a weak local model rewrite source files blindly.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from core.config import (
    NEUROGOLF_AUTONOMY_REPORTS_DIR,
    NEUROGOLF_OPERATOR_AUTO_NOTE_PATH,
)

SELF_IMPROVEMENT_LOG_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement.jsonl"
SELF_IMPROVEMENT_STATE_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement_state.json"
SELF_IMPROVEMENT_ENABLED_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement_enabled"
MAX_SELF_IMPROVEMENT_CYCLES = 20
MIN_PERFORMANCE_THRESHOLD = 0.7


def _daemon_log(message: str) -> None:
    """Write to daemon log for debugging."""
    log_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] self_improvement:{message}\n")


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _plan_key(report: dict) -> str:
    plan = report.get("plan") if isinstance(report.get("plan"), dict) else {}
    action = str(plan.get("action", "")).strip() or "-"
    target = str(plan.get("target") or plan.get("seed_label") or "").strip() or "-"
    mode = str(plan.get("mode") or plan.get("variant_hint") or "").strip() or "-"
    return f"{action}|{target}|{mode}"


def _normalized_reason(report: dict) -> str:
    submit_result = report.get("submit_result") if isinstance(report.get("submit_result"), dict) else {}
    return str(submit_result.get("reason", "")).strip().lower()


def _latest_manifest_score(report: dict) -> float | None:
    manifest = report.get("build_manifest") if isinstance(report.get("build_manifest"), dict) else {}
    return (
        _safe_float(manifest.get("estimated_live_score"))
        or _safe_float(manifest.get("known_score"))
        or _safe_float(manifest.get("total_score"))
    )


def _top_focus_tasks(report: dict, limit: int = 6) -> list[dict[str, object]]:
    manifest = report.get("build_manifest") if isinstance(report.get("build_manifest"), dict) else {}
    selected = manifest.get("selected")
    if not isinstance(selected, list):
        return []
    ranked: list[dict[str, object]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        if not task:
            continue
        score = _safe_float(item.get("score"))
        if score is None:
            continue
        params = _safe_int(item.get("params"))
        memory = _safe_int(item.get("memory"))
        macs = _safe_int(item.get("macs"))
        ranked.append(
            {
                "task": task,
                "score": round(score, 6),
                "source": str(item.get("source", "")).strip(),
                "params": params,
                "memory": memory,
                "macs": macs,
                "cost_hint": params + memory + macs,
            }
        )
    ranked.sort(key=lambda row: (float(row["score"]), -int(row["cost_hint"])))
    return ranked[:limit]


def _load_improvement_state() -> dict:
    if not SELF_IMPROVEMENT_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(SELF_IMPROVEMENT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _persist_improvement_state(state: dict) -> None:
    SELF_IMPROVEMENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    SELF_IMPROVEMENT_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    note = str(state.get("operator_note", "")).strip()
    if note:
        NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.write_text(note + "\n", encoding="utf-8")
    elif NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.exists():
        NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.unlink()


def _state_signature(payload: dict) -> str:
    if not payload:
        return ""
    comparable = {key: value for key, value in payload.items() if key != "updated_at"}
    return json.dumps(comparable, sort_keys=True)


def is_self_improvement_enabled() -> bool:
    """Check if self-improvement is enabled."""
    return SELF_IMPROVEMENT_ENABLED_PATH.exists()


def enable_self_improvement() -> None:
    """Enable the self-improvement system."""
    SELF_IMPROVEMENT_ENABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELF_IMPROVEMENT_ENABLED_PATH.touch()
    _daemon_log("adaptive learning enabled")


def disable_self_improvement() -> None:
    """Disable the self-improvement system."""
    if SELF_IMPROVEMENT_ENABLED_PATH.exists():
        SELF_IMPROVEMENT_ENABLED_PATH.unlink()
    if NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.exists():
        NEUROGOLF_OPERATOR_AUTO_NOTE_PATH.unlink()
    _daemon_log("adaptive learning disabled")


def load_improvement_history(limit: int = 50) -> list[dict]:
    """Load self-improvement history from log."""
    if not SELF_IMPROVEMENT_LOG_PATH.exists():
        return []
    lines = SELF_IMPROVEMENT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    records = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def log_improvement_attempt(
    analysis: str,
    proposed_changes: list[dict],
    validation_result: dict | None,
    applied: bool,
    reason: str | None = None,
) -> None:
    """Log an improvement attempt."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
        "proposed_changes": proposed_changes,
        "validation_result": validation_result,
        "applied": applied,
        "reason": reason,
    }
    SELF_IMPROVEMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SELF_IMPROVEMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def analyze_recent_performance(reports: list[dict]) -> dict[str, Any]:
    """Analyze recent performance to identify learning opportunities."""
    if not reports:
        return {"status": "no_data", "recommendation": "Need more reports to analyze"}

    total = len(reports)
    submissions = [report for report in reports if (report.get("submit_result") or {}).get("submitted")]
    successes = len(submissions)
    failures = total - successes
    branch_counts = Counter(_plan_key(report) for report in reports if _plan_key(report))
    reason_counts = Counter(_normalized_reason(report) for report in reports if _normalized_reason(report))
    latest_report = reports[0]
    latest_branch = _plan_key(latest_report)
    latest_reason = _normalized_reason(latest_report)
    latest_manifest = latest_report.get("build_manifest") if isinstance(latest_report.get("build_manifest"), dict) else {}
    latest_files = _safe_int(latest_manifest.get("files"))
    latest_score = _latest_manifest_score(latest_report)

    recent_streak = 0
    for report in reports:
        if _plan_key(report) != latest_branch:
            break
        recent_streak += 1

    focus_tasks = _top_focus_tasks(latest_report)
    repeated_seed_loops = sum(
        count
        for key, count in branch_counts.items()
        if key.startswith("direct_seed_pack|") and count >= 2
    )
    recoverable_reasons = {
        "experiment already submitted",
        "candidate matches current active pack",
        "submission message already exists in recent history",
        "delta_sum_local below threshold",
    }
    blocked_repeat_count = sum(reason_counts.get(reason, 0) for reason in recoverable_reasons)

    needs_improvement = (
        (successes / total if total else 0.0) < MIN_PERFORMANCE_THRESHOLD
        or recent_streak >= 2
        or blocked_repeat_count > 0
        or repeated_seed_loops > 0
        or latest_files >= 400
        or bool(focus_tasks)
    )

    recommendation = "Hold steady and keep the current branch mix."
    if latest_files >= 400 and latest_score is not None:
        recommendation = (
            "Treat the latest 400/400 pack as the baseline and chase per-task graph reductions "
            "instead of restarting from raw seeds."
        )
    elif blocked_repeat_count > 0 or repeated_seed_loops > 0:
        recommendation = "Stop replaying blocked seed families and rotate to materially new builds."
    elif latest_files < 400:
        recommendation = "Increase valid task coverage before spending a submission slot."

    return {
        "status": "needs_improvement" if needs_improvement else "performing_well",
        "patterns": {
            "total_cycles": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "failures": failures,
            "latest_branch": latest_branch,
            "latest_reason": latest_reason,
            "latest_files": latest_files,
            "latest_score": latest_score,
            "recent_branch_streak": recent_streak,
            "blocked_repeat_count": blocked_repeat_count,
            "repeated_seed_loops": repeated_seed_loops,
            "top_branches": branch_counts.most_common(6),
            "top_reasons": reason_counts.most_common(6),
            "focus_tasks": focus_tasks,
        },
        "recommendation": recommendation,
    }


def _format_operator_note(state: dict) -> str:
    hints = state.get("planner_hints", [])
    focus_tasks = state.get("focus_tasks", [])
    lines = [
        "Adaptive guidance:",
    ]
    for hint in hints:
        lines.append(f"- {hint}")
    if focus_tasks:
        formatted = ", ".join(
            f"{item['task']} ({float(item['score']):.2f}, {item['source'] or 'unknown source'})"
            for item in focus_tasks
        )
        lines.append(f"- Weakest current tasks to rebuild first: {formatted}.")
    return "\n".join(lines)


def build_improvement_state(reports: list[dict], analysis: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic learning package for the next autonomy cycle."""
    latest_report = reports[0]
    latest_manifest = latest_report.get("build_manifest") if isinstance(latest_report.get("build_manifest"), dict) else {}
    latest_branch = _plan_key(latest_report)
    latest_score = _latest_manifest_score(latest_report)
    latest_files = _safe_int(latest_manifest.get("files"))
    latest_reason = _normalized_reason(latest_report)
    latest_mode = str((latest_report.get("plan") or {}).get("mode") or "").strip()
    latest_scope = str(latest_manifest.get("score_scope", "")).strip()
    focus_tasks = analysis.get("patterns", {}).get("focus_tasks", [])
    top_branches = analysis.get("patterns", {}).get("top_branches", [])
    top_reasons = dict(analysis.get("patterns", {}).get("top_reasons", []))

    hints: list[str] = []
    if latest_files >= 400 and latest_score is not None:
        hints.append(
            f"Current 400/400 baseline is {latest_branch} at an estimated local score of {latest_score:.2f}; improve from that pack instead of resetting to seed-only submissions."
        )
    if latest_mode == "skip_known_dynamic":
        hints.append(
            "Keep skip_known_dynamic as the default fresh-build mode until another validated mode beats it on score or task quality."
        )
    if latest_scope == "profile_only":
        hints.append(
            "Profile-only scorer blends are the current intended baseline; do not drop back to strict-known unless validation or score evidence improves."
        )
    if analysis.get("patterns", {}).get("recent_branch_streak", 0) >= 2:
        hints.append(
            f"Avoid rebuilding {latest_branch} unchanged; require a new source mix, repaired tasks, or a stronger validated candidate before repeating it."
        )
    if top_reasons.get("experiment already submitted", 0):
        hints.append(
            "Never resubmit the same experiment fingerprint. Change the artifact meaningfully before using a submission slot."
        )
    if top_reasons.get("submission message already exists in recent history", 0):
        hints.append(
            "Treat duplicate submission messages as a planner bug. Change both the artifact and the description together."
        )
    if analysis.get("patterns", {}).get("repeated_seed_loops", 0):
        hints.append(
            "Direct imported seed packs are reference material only. Keep the loop on fresh builds, validated repair packs, or targeted per-task replacements."
        )
    if latest_reason == "submission disabled" and latest_files >= 400:
        hints.append(
            "When submissions are enabled again, submit only after the pack differs from the active manifest and still validates 400/400 locally."
        )
    if latest_manifest.get("build_strategy") == "validated_repair":
        hints.append(
            "Prefer validated repair outputs over raw packs when their score stays close, because they preserve local validity evidence."
        )
    if top_branches:
        dominant_key, dominant_count = top_branches[0]
        if dominant_count >= 3:
            hints.append(
                f"The dominant recent branch is {dominant_key} ({dominant_count} cycles). Put new effort into weaker tasks rather than replaying the same branch."
            )

    summary = (
        "Adaptive learning is steering the planner toward gradual, competition-intended progress: "
        "keep the strongest 400/400 baseline, avoid repeated seed loops, and rebuild the weakest per-task graphs first."
    )

    state: dict[str, Any] = {
        "mode": "adaptive_learning",
        "analysis_summary": summary,
        "recommendation": analysis.get("recommendation", ""),
        "latest_branch": latest_branch,
        "latest_reason": latest_reason,
        "latest_mode": latest_mode,
        "latest_files": latest_files,
        "baseline_estimated_score": latest_score,
        "planner_hints": hints,
        "focus_tasks": focus_tasks,
    }
    state["operator_note"] = _format_operator_note(state)
    return state


def run_self_improvement_cycle(reports: list[dict]) -> dict:
    """Run one adaptive learning cycle."""
    if not is_self_improvement_enabled():
        return {"status": "disabled", "reason": "Self-improvement not enabled"}

    analysis = analyze_recent_performance(reports)
    if analysis.get("status") == "no_data":
        return {"status": "no_data", "reason": "Need more reports to analyze", "analysis": analysis}

    proposal = build_improvement_state(reports, analysis)
    current_state = _load_improvement_state()
    changed = _state_signature(current_state) != _state_signature(proposal)
    if not changed:
        return {
            "status": "no_action",
            "reason": "Adaptive guidance is unchanged",
            "analysis": analysis,
            "proposal": proposal,
            "applied_changes": [],
            "failed_changes": [],
        }

    _persist_improvement_state(proposal)
    applied_changes = [
        {
            "file": str(SELF_IMPROVEMENT_STATE_PATH),
            "reason": "Updated adaptive learning state",
        },
        {
            "file": str(NEUROGOLF_OPERATOR_AUTO_NOTE_PATH),
            "reason": "Updated automatic operator guidance",
        },
    ]
    log_improvement_attempt(
        analysis=proposal.get("analysis_summary", ""),
        proposed_changes=[{"type": "planner_hint", "text": hint} for hint in proposal.get("planner_hints", [])],
        validation_result={
            "artifacts": applied_changes,
            "focus_tasks": proposal.get("focus_tasks", []),
            "recommendation": proposal.get("recommendation", ""),
        },
        applied=True,
        reason="Updated adaptive planner guidance from recent cycle evidence",
    )
    _daemon_log("adaptive guidance updated")
    return {
        "status": "learned",
        "applied_changes": applied_changes,
        "failed_changes": [],
        "analysis": analysis,
        "proposal": proposal,
    }


def get_improvement_status() -> dict:
    """Get current self-improvement status for UI."""
    history = load_improvement_history(limit=20)
    recent_applied = [item for item in history if item.get("applied")]

    return {
        "enabled": is_self_improvement_enabled(),
        "total_attempts": len(history),
        "recent_improvements": len(recent_applied),
        "max_cycles": MAX_SELF_IMPROVEMENT_CYCLES,
        "last_attempt": history[-1] if history else None,
        "state": _load_improvement_state(),
    }
