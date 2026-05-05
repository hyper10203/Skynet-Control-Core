from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from core.config import (
    MODEL_OPTIONS,
    MODEL_REGISTRY,
    NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT,
    NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS,
    NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD,
    NEUROGOLF_AUTONOMY_LOG_PATH,
    NEUROGOLF_AUTONOMY_LOOP_SECONDS,
    NEUROGOLF_AUTONOMY_MAX_HISTORY,
    NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS,
    NEUROGOLF_AUTONOMY_PID_PATH,
    NEUROGOLF_AUTONOMY_RECENT_REPORT_LIMIT,
    NEUROGOLF_AUTONOMY_REPORTS_DIR,
    NEUROGOLF_AUTONOMY_STATUS_PATH,
    NEUROGOLF_OPERATOR_NOTE_PATH,
    NEUROGOLF_PACKAGE_SCRIPT,
    NEUROGOLF_PROJECT_ROOT,
)
from core.executor import ask
from core.neurogolf_context import summarize_neurogolf_workspace, sync_neurogolf_state
from core.prompts import NEUROGOLF_AUTONOMY_SYSTEM_PROMPT
from core.self_improvement import (
    enable_self_improvement,
    disable_self_improvement,
    is_self_improvement_enabled,
    run_self_improvement_cycle,
    get_improvement_status,
)
from utils.submission_validation import validate_submission_zip, write_validation_report


NEUROGOLF_LEARNING_STATE_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "learning_state.json"
RECOVERABLE_SKIP_REASONS = {
    "experiment already submitted",
    "candidate matches current active pack",
    "submission message already exists in recent history",
    "delta_sum_local below threshold",
}


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


def _daemon_log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    NEUROGOLF_AUTONOMY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NEUROGOLF_AUTONOMY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _write_status(phase: str, message: str, *, progress: float, extra: dict | None = None) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "phase": phase,
        "message": message,
        "progress": max(0.0, min(1.0, progress)),
    }
    if extra:
        payload.update(extra)
    NEUROGOLF_AUTONOMY_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUROGOLF_AUTONOMY_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_operator_note() -> str:
    if not NEUROGOLF_OPERATOR_NOTE_PATH.exists():
        return ""
    return NEUROGOLF_OPERATOR_NOTE_PATH.read_text(encoding="utf-8").strip()


def _compact_summary(summary: dict) -> str:
    payload = {
        "best_completed_public_score": summary.get("best_completed_public_score"),
        "pending_submission_count": summary.get("pending_submission_count"),
        "current_team_name": summary.get("current_team_name"),
        "current_manifest": summary.get("current_manifest", {}),
        "recent_submissions": summary.get("recent_submissions", [])[:6],
        "recent_output_manifests": summary.get("recent_output_manifests", [])[:6],
        "available_cached_candidates": _available_cached_candidates(summary)[:8],
        "imported_sources": summary.get("imported_sources", [])[:10],
        "campaign_progress": summary.get("campaign_progress", {}),
        "public_seed_hints": summary.get("public_seed_hints", {}),
        "buildable_seed_hints": summary.get("buildable_seed_hints", {}),
        "available_seed_labels": summary.get("available_seed_labels", []),
        "highest_claimed_imported_public_score": summary.get("highest_claimed_imported_public_score"),
        "metric_v3_rules": summary.get("metric_v3_rules", {}),
        "exploit_audit": summary.get("exploit_audit", {}),
        "competition_data": summary.get("competition_data", {}),
        "recent_cycle_patterns": summary.get("recent_cycle_patterns", {}),
        "autonomy_learning": summary.get("autonomy_learning", {}),
        "lessons": summary.get("lessons", []),
    }
    return json.dumps(payload, indent=2)[:8000]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one autonomous NeuroGolf optimization cycle.")
    parser.add_argument("--zip-name", default="submission_autonomous.zip")
    parser.add_argument("--allow-submit", action="store_true")
    parser.add_argument("--min-local-delta", type=float, default=NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=NEUROGOLF_AUTONOMY_LOOP_SECONDS)
    parser.add_argument("--max-pending-submissions", type=int, default=NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever when --loop is enabled.")
    return parser


def _recent_reports(limit: int = NEUROGOLF_AUTONOMY_RECENT_REPORT_LIMIT) -> list[dict]:
    reports: list[dict] = []
    if not NEUROGOLF_AUTONOMY_REPORTS_DIR.exists():
        return reports
    for path in sorted(NEUROGOLF_AUTONOMY_REPORTS_DIR.glob("cycle_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_path"] = str(path)
        reports.append(payload)
    return reports


def _submitted_fingerprints(reports: list[dict]) -> set[str]:
    fingerprints: set[str] = set()
    for report in reports:
        submit_result = report.get("submit_result") or {}
        if submit_result.get("submitted"):
            fingerprint = str(report.get("experiment_fingerprint", "")).strip()
            if fingerprint:
                fingerprints.add(fingerprint)
    return fingerprints


def _recent_seed_modes(reports: list[dict], limit: int = 6) -> set[tuple[str, str]]:
    used: set[tuple[str, str]] = set()
    for report in reports[:limit]:
        plan = report.get("plan") or {}
        seed = str(plan.get("seed_label", "")).strip()
        mode = str(plan.get("mode", "")).strip()
        if seed and mode:
            used.add((seed, mode))
    return used


def _available_cached_candidates(summary: dict) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for item in summary.get("recent_output_manifests", []):
        zip_path = str(item.get("zip_path", "")).strip()
        if not zip_path:
            continue
        path = Path(zip_path)
        if not path.exists():
            continue
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        score_hint = item.get("estimated_live_score")
        if score_hint is None:
            score_hint = item.get("known_score")
        if score_hint is None:
            score_hint = item.get("seed_claimed_public_score")
        try:
            numeric_hint = float(score_hint) if score_hint is not None else 0.0
        except (TypeError, ValueError):
            numeric_hint = 0.0
        candidates.append(
            {
                "target": path.name,
                "zip_path": str(path),
                "seed_label": item.get("seed_label"),
                "variant": item.get("mode") or item.get("build_strategy") or "",
                "score_hint": numeric_hint,
                "updated_at": item.get("updated_at", ""),
                "manifest_name": item.get("name", ""),
            }
        )
    # Sort by updated_at (newest first), then by score_hint (highest first)
    # Prioritizes recent submissions over historical high scores
    candidates.sort(key=lambda row: (row["updated_at"], row["score_hint"]), reverse=True)
    return candidates


def _seed_variant_label(variant: str) -> str:
    value = str(variant or "").strip()
    mapping = {
        "processable_best": "safe splice",
        "hybrid_priority": "open splice",
        "fill_invalid_priority": "repair splice",
    }
    return mapping.get(value, value or "safe splice")


def _blend_variant_label(variant: str) -> str:
    value = str(variant or "").strip()
    mapping = {
        "skip_known_dynamic": "static replacement blend",
        "strict_known": "strict known-only blend",
        "skip_known_dynamic_sanitized": "sanitized static blend",
    }
    return mapping.get(value, value or "strict known-only blend")


def _display_target(action: str, target: str, seed_label: str, variant: str) -> str:
    if action == "direct_seed_pack":
        return f"direct imported seed {seed_label or target}".strip()
    if action == "cached_candidate_submit":
        return f"cached candidate {target}".strip()
    if action == "scorer_blend_build":
        return _blend_variant_label(variant)
    if action == "seed_preserving_build":
        return f"{seed_label} with {_seed_variant_label(variant)}".strip()
    return (target or seed_label or variant or "next experiment").strip()


def _available_seed_candidates(summary: dict) -> list[tuple[str, float]]:
    buildable = summary.get("buildable_seed_hints", {})
    candidates: list[tuple[str, float]] = []
    seen: set[str] = set()
    if isinstance(buildable, dict):
        for label, score in buildable.items():
            clean_label = str(label).strip()
            if not clean_label or clean_label in seen:
                continue
            try:
                candidates.append((clean_label, float(score)))
                seen.add(clean_label)
            except (TypeError, ValueError):
                continue
    imported_scores = _direct_seed_score_map(summary)
    public_scores = summary.get("public_seed_hints", {})
    available_labels = summary.get("available_seed_labels", [])
    if isinstance(available_labels, list):
        for label in available_labels:
            clean_label = str(label).strip()
            if not clean_label or clean_label in seen:
                continue
            try:
                score = float(imported_scores.get(clean_label, public_scores.get(clean_label, 0.0) if isinstance(public_scores, dict) else 0.0))
            except (TypeError, ValueError):
                score = 0.0
            candidates.append((clean_label, score))
            seen.add(clean_label)
    if not candidates:
        # NO FALLBACK TO OLD V2 SEEDS - force fresh builds
        # Return empty list to trigger fresh build mode
        pass
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _seed_score_map(summary: dict) -> dict[str, float]:
    return {label: score for label, score in _available_seed_candidates(summary)}


# V2-era seeds with dynamic shapes/invalid files - EXCLUDE ALL OF THESE
# DELETE ALL OLD SEEDS - only use freshly imported V3-compliant sources
V2_INVALID_SEEDS = {
    # Artem family - all have dynamic shapes/invalid files
    "latest_artem_part4",
    "latest_artem_part2",
    "latest_artem_part1",
    "latest_artem_part3",
    # Konbu family
    "latest_konbu_5344",
    "latest_konbu_5343",
    # Rocker family
    "latest_rocker_5353",
    "latest_rocker_5350",
    "latest_rocker_5340",
    # Other V2-era seeds with exploits
    "latest_amanatar_optimal",
    "latest_karnak_logic_tasks",
    "latest_jon_ngc26_top",
    "latest_jon_ngc26_blend",
    "latest_afr1ste_5177",
    "latest_afr1ste_5178",
    "latest_magmacot_5550",
    "latest_magmacot_5540",
    # Any seed with "needless" prefix (old V2)
    "needless_4250",
    "needless_4200",
    "needless_4300",
    # Any seed with "legacy" or "old" prefix
    "legacy_artem",
    "old_konbu",
    # Block ALL seeds that aren't freshly imported today
    # Only allow seeds with today's valid 400/400 submission
}

# FORCE FRESH BUILDS: Delete all seeds, rebuild only what passes V3 validation
DELETE_ALL_OLD_SEEDS = True  # Set to True to force fresh builds from scratch

# Known invalid tasks per V2 seed (these have dynamic shapes/symbolic dims)
# Use this to filter out only invalid tasks, keeping valid ones from good seeds
SEED_INVALID_TASKS = {
    "latest_artem_part4": [
        "task004", "task012", "task015", "task016", "task023", "task024", "task025", "task032",
        "task041", "task047", "task048", "task050", "task053", "task055", "task063", "task067",
        "task073", "task077", "task081", "task084", "task085", "task086", "task092", "task098",
        "task100", "task101", "task110", "task112", "task117", "task120", "task122", "task125",
        "task127", "task129", "task132", "task135", "task139", "task141", "task147", "task150",
        "task151", "task155", "task157", "task160", "task161", "task162", "task166", "task168",
        "task171", "task180", "task181", "task182", "task185", "task187", "task190", "task192",
        "task193", "task196", "task198", "task199", "task202", "task205", "task207", "task220",
        "task222", "task224", "task225", "task226", "task229", "task230", "task232", "task239",
        "task244", "task248", "task252", "task253", "task254", "task256", "task257", "task262",
        "task266", "task267", "task268", "task272", "task276", "task278", "task280", "task282",
        "task283", "task284", "task288", "task293", "task294", "task296", "task298", "task299",
        "task303", "task305", "task309", "task314", "task320", "task322", "task323", "task325",
        "task331", "task332", "task334", "task336", "task337", "task340", "task341", "task342",
        "task344", "task346", "task352", "task376", "task379", "task385", "task387", "task389",
        "task397", "task399",
    ],
}


def _available_direct_seed_candidates(summary: dict) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    for row in summary.get("imported_sources", []):
        if not row.get("usable_as_seed"):
            continue
        label = str(row.get("seed_label", "")).strip()
        if not label:
            continue
        # Skip V2-era invalid seeds - only use today's valid sources
        if label in V2_INVALID_SEEDS:
            continue
        try:
            score = float(row.get("claimed_public_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        candidates.append((label, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _direct_seed_score_map(summary: dict) -> dict[str, float]:
    return {label: score for label, score in _available_direct_seed_candidates(summary)}


def _default_seed_label(summary: dict) -> str | None:
    # First try imported sources (which are sorted by score, highest first)
    direct_candidates = _available_direct_seed_candidates(summary)
    if direct_candidates:
        # Use highest scoring valid imported source
        return direct_candidates[0][0]
    # Fall back to buildable seeds
    candidates = _available_seed_candidates(summary)
    if candidates:
        return candidates[0][0]
    # NO FALLBACK TO OLD V2 SEEDS
    # Return None to trigger fresh build mode - all old seeds deleted
    return None


def _top_buildable_seed(summary: dict) -> tuple[str, float] | None:
    # First try imported sources (sorted by score, highest first)
    direct_candidates = _available_direct_seed_candidates(summary)
    if direct_candidates:
        return direct_candidates[0]
    # Fall back to buildable seeds
    candidates = _available_seed_candidates(summary)
    return candidates[0] if candidates else None


def _fallback_candidates(summary: dict) -> list[tuple[str, str]]:
    modes = ("processable_best", "hybrid_priority", "fill_invalid_priority")
    return [(label, mode) for label, _score in _available_seed_candidates(summary) for mode in modes]


def _should_bypass_local_delta_gate(action: str, plan: dict, summary: dict) -> bool:
    if action in {"scorer_blend_build", "direct_seed_pack", "cached_candidate_submit"}:
        return True
    if action != "seed_preserving_build":
        return False

    seed_label = str(plan.get("seed_label", "")).strip()
    variant = str(plan.get("mode", "")).strip()
    if not seed_label:
        return False

    seed_scores = _seed_score_map(summary)
    claimed = float(seed_scores.get(seed_label, 0.0))
    best_completed = float(summary.get("best_completed_public_score") or 0.0)
    if claimed <= 0 or claimed + 5.0 < best_completed:
        return False

    recent_submissions = summary.get("recent_submissions", [])
    normalized_seed = seed_label.lower()
    if variant == "processable_best":
        for item in recent_submissions:
            description = str(item.get("description", "")).lower()
            score = item.get("public_score")
            if score is None:
                continue
            if "6285" in normalized_seed and "6285" in description and "processable-best" in description:
                return False

    return variant in {"hybrid_priority", "fill_invalid_priority"}


def _experiment_key(plan: dict) -> tuple[str, str, str]:
    action = str(plan.get("action", "")).strip() or "seed_preserving_build"
    target = str(plan.get("target", "")).strip() or str(plan.get("seed_label", "")).strip()
    variant = str(plan.get("mode", "")).strip() or str(plan.get("variant_hint", "")).strip()
    if action == "seed_preserving_build":
        variant = _normalize_seed_variant(variant)
    elif action == "scorer_blend_build":
        target = ""
        variant = _normalize_blend_variant(variant)
    elif action in {"direct_seed_pack", "cached_candidate_submit"}:
        variant = str(plan.get("variant_hint", "")).strip()
    return (action, target, variant)


def _recent_experiment_keys(reports: list[dict], limit: int = NEUROGOLF_AUTONOMY_RECENT_REPORT_LIMIT) -> set[tuple[str, str, str]]:
    used: set[tuple[str, str, str]] = set()
    for report in reports[:limit]:
        plan = report.get("plan") or {}
        if isinstance(plan, dict):
            used.add(_experiment_key(plan))
    return used


def _key_text(key: tuple[str, str, str]) -> str:
    return "|".join(str(part or "-") for part in key)


def _reason_matches(reason: str, prefixes: set[str]) -> bool:
    lowered = str(reason or "").strip().lower()
    return any(lowered.startswith(prefix) for prefix in prefixes)


def _learning_from_reports(reports: list[dict]) -> dict:
    branch_counts: dict[str, int] = {}
    blocked_counts: dict[str, int] = {}
    last_reason_by_key: dict[str, str] = {}
    last_blocked_reason_by_key: dict[str, str] = {}
    submitted_fingerprints: set[str] = set()
    seen_signatures: set[str] = set()

    for report in reports:
        plan = report.get("plan") or {}
        if not isinstance(plan, dict):
            continue
        key = _key_text(_experiment_key(plan))
        branch_counts[key] = branch_counts.get(key, 0) + 1
        submit_result = report.get("submit_result") or {}
        reason = str(submit_result.get("reason", "")).strip()
        if reason:
            last_reason_by_key[key] = reason
        if _reason_matches(reason, RECOVERABLE_SKIP_REASONS):
            blocked_counts[key] = blocked_counts.get(key, 0) + 1
            last_blocked_reason_by_key.setdefault(key, reason)
        fingerprint = str(report.get("experiment_fingerprint", "")).strip()
        if fingerprint and (submit_result.get("submitted") or reason == "experiment already submitted"):
            submitted_fingerprints.add(fingerprint)
        signature = str(report.get("candidate_signature", "")).strip()
        if signature:
            seen_signatures.add(signature)

    streak_key = ""
    streak_count = 0
    for report in reports:
        plan = report.get("plan") or {}
        if not isinstance(plan, dict):
            continue
        key = _key_text(_experiment_key(plan))
        if not streak_key:
            streak_key = key
            streak_count = 1
            continue
        if key != streak_key:
            break
        streak_count += 1

    blocked = [
        {
            "key": key,
            "count": count,
            "last_reason": last_blocked_reason_by_key.get(key, last_reason_by_key.get(key, "")),
        }
        for key, count in sorted(blocked_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    recommendations: list[str] = []
    if streak_count >= 3 and streak_key:
        recommendations.append(f"Break the repeated branch streak: {streak_key} appeared {streak_count} times in a row.")
    if blocked:
        recommendations.append("Do not build branches whose last result was duplicate/current/message-colliding unless every fresh branch is exhausted.")
    if any(item["last_reason"] == "delta_sum_local below threshold 40.00" for item in blocked):
        recommendations.append("When local delta blocks a splice, switch seed or mode instead of lowering the bar blindly.")

    return {
        "branch_counts": branch_counts,
        "blocked_branches": blocked[:12],
        "recent_streak": {"key": streak_key, "count": streak_count} if streak_key else {},
        "submitted_fingerprints": sorted(submitted_fingerprints)[:50],
        "seen_candidate_signatures": sorted(seen_signatures)[:50],
        "recommendations": recommendations,
    }


def _persist_learning_state(learning: dict) -> None:
    NEUROGOLF_LEARNING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(learning)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    NEUROGOLF_LEARNING_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _branch_block_count(summary: dict, key: tuple[str, str, str]) -> int:
    learning = summary.get("autonomy_learning", {}) if isinstance(summary.get("autonomy_learning"), dict) else {}
    key_text = _key_text(key)
    for item in learning.get("blocked_branches", []):
        if str(item.get("key", "")) == key_text:
            try:
                return int(item.get("count", 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _branch_repeat_count(summary: dict, key: tuple[str, str, str]) -> int:
    learning = summary.get("autonomy_learning", {}) if isinstance(summary.get("autonomy_learning"), dict) else {}
    branch_counts = learning.get("branch_counts", {})
    if not isinstance(branch_counts, dict):
        return 0
    try:
        return int(branch_counts.get(_key_text(key), 0))
    except (TypeError, ValueError):
        return 0


def _branch_recent_streak_count(summary: dict, key: tuple[str, str, str]) -> int:
    learning = summary.get("autonomy_learning", {}) if isinstance(summary.get("autonomy_learning"), dict) else {}
    streak = learning.get("recent_streak", {})
    if not isinstance(streak, dict):
        return 0
    if str(streak.get("key", "")) != _key_text(key):
        return 0
    try:
        return int(streak.get("count", 0))
    except (TypeError, ValueError):
        return 0


def _should_avoid_branch(summary: dict, key: tuple[str, str, str], recent_keys: set[tuple[str, str, str]] | None = None) -> bool:
    if _branch_block_count(summary, key) >= 2:
        return True
    if _branch_recent_streak_count(summary, key) >= 2:
        return True
    if recent_keys is not None and key in recent_keys:
        return True
    return _branch_repeat_count(summary, key) >= 4


def _choose_fresh_fallback_plan(summary: dict, reports: list[dict], tried_keys: set[tuple[str, str, str]]) -> dict | None:
    recent_keys = _recent_experiment_keys(reports)
    candidates = [dict(candidate) for candidate in _fallback_experiments(summary)]

    for candidate in candidates:
        key = _experiment_key(candidate)
        if key in tried_keys or _should_avoid_branch(summary, key, recent_keys):
            continue
        candidate["rationale"] = "Selected by deterministic learning guard: fresh branch after recent failures."
        return candidate

    for candidate in candidates:
        key = _experiment_key(candidate)
        if key in tried_keys or _branch_block_count(summary, key) >= 3 or _branch_repeat_count(summary, key) >= 4:
            continue
        candidate["rationale"] = "Selected by deterministic learning guard: least-repeated fallback branch."
        return candidate

    for candidate in candidates:
        key = _experiment_key(candidate)
        if key not in tried_keys:
            candidate["rationale"] = "Selected by deterministic learning guard: all known branches have history, using next available."
            return candidate
    return None


def _coerce_plan_schema(plan: dict, summary: dict) -> dict:
    if not isinstance(plan, dict):
        return {}
    coerced = dict(plan)
    experiment_type = str(coerced.get("experiment_type", "")).strip().lower()
    action = str(coerced.get("action", "")).strip().lower()
    token = action or experiment_type or "seed_preserving_build"
    if token in {"direct_seed", "direct_seed_pack", "use_direct_seed"}:
        coerced["action"] = "direct_seed_pack"
        coerced["should_submit"] = False
        if not str(coerced.get("seed_label", "")).strip():
            coerced["seed_label"] = str(coerced.get("target", "")).strip()
    elif token in {"cached_candidate", "cached_candidate_submit", "reuse_candidate", "submit_cached_candidate"}:
        coerced["action"] = "cached_candidate_submit"
        coerced["should_submit"] = False
        if not str(coerced.get("target", "")).strip():
            coerced["target"] = str(coerced.get("seed_label", "")).strip()
    elif token in {"open_blend", "blend", "scorer_blend_build"}:
        coerced["action"] = "scorer_blend_build"
        if not str(coerced.get("mode", "")).strip():
            coerced["mode"] = str(coerced.get("variant_hint", "")).strip()
        coerced["seed_label"] = ""
    elif token in {"fresh_build", "build_from_scratch", "new_build"}:
        coerced["action"] = "scorer_blend_build"
        coerced["mode"] = "strict_known"
        coerced["seed_label"] = ""
        coerced["should_submit"] = False  # Build first, submit after validation
    elif token in {"sync_only"}:
        coerced["action"] = "sync_only"
    else:
        coerced["action"] = "seed_preserving_build"
        if not str(coerced.get("seed_label", "")).strip():
            coerced["seed_label"] = str(coerced.get("target", "")).strip()
        if not str(coerced.get("mode", "")).strip():
            coerced["mode"] = str(coerced.get("variant_hint", "")).strip()
    coerced.pop("experiment_type", None)
    return coerced


def _normalize_seed_variant(text: str) -> str:
    value = str(text or "").strip().lower()
    if value in {"processable_best", "fill_invalid_priority", "hybrid_priority"}:
        return value
    if value in {"safe", "safer", "conservative", "surgical", "minimal"}:
        return "processable_best"
    if value in {"aggressive", "hybrid", "mixed", "balanced", "open"}:
        return "hybrid_priority"
    if value in {"repair", "repair_heavy", "fill", "fill_invalid", "invalid", "recovery"}:
        return "fill_invalid_priority"
    return "processable_best"


def _normalize_blend_variant(text: str) -> str:
    value = str(text or "").strip().lower()
    if value in {"skip_known_dynamic", "strict_known", "skip_known_dynamic_sanitized"}:
        return value
    if value in {"dynamic", "exploit", "fast", "open", "strict", "safe", "known", "fresh", "scratch"}:
        return "strict_known"
    if value in {"sanitized", "sanitize", "patched"}:
        return "skip_known_dynamic_sanitized"
    return "strict_known"


def _fallback_experiments(summary: dict) -> list[dict]:
    experiments: list[dict] = []
    for mode in ("strict_known", "skip_known_dynamic_sanitized", "skip_known_dynamic"):
        experiments.append({"action": "scorer_blend_build", "mode": mode})
    for label, _score in _available_seed_candidates(summary)[:4]:
        for mode in ("processable_best", "fill_invalid_priority", "hybrid_priority"):
            experiments.append({"action": "seed_preserving_build", "seed_label": label, "mode": mode})
    for label, _score in _available_direct_seed_candidates(summary)[:4]:
        experiments.append({"action": "direct_seed_pack", "seed_label": label, "variant_hint": "", "should_submit": False})
    for candidate in _available_cached_candidates(summary)[:6]:
        experiments.append(
            {
                "action": "cached_candidate_submit",
                "target": candidate["target"],
                "variant_hint": str(candidate.get("variant", "")),
                "should_submit": False,
            }
        )
    return experiments


def _run_external_build(command: list[str], *, label: str, target: str, variant: str) -> None:
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=NEUROGOLF_PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate(timeout=NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    except subprocess.TimeoutExpired as exc:
        timeout_seconds = int(NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS)
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                process.kill()
            try:
                process.communicate(timeout=15)
            except Exception:
                pass
        message = f"{label} timed out after {timeout_seconds} seconds"
        _daemon_log(f"build:timeout target={target} variant={variant} timeout={timeout_seconds}")
        _write_status(
            "error",
            message,
            progress=1.0,
            extra={"target": target, "variant": variant, "build_strategy": label},
        )
        raise RuntimeError(message) from exc
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "").strip()
        stdout = str(exc.output or "").strip()
        detail = stderr or stdout or f"exit status {exc.returncode}"
        message = f"{label} failed: {detail[:400]}"
        _daemon_log(f"build:failed target={target} variant={variant} returncode={exc.returncode} detail={detail[:400]}")
        _write_status(
            "error",
            message,
            progress=1.0,
            extra={"target": target, "variant": variant, "build_strategy": label},
        )
        raise RuntimeError(message) from exc


def _safe_model_response(role: str, prompt: str, system: str, *, fallback_role: str = "orchestrator") -> str:
    # Validate inputs
    if not role or not isinstance(role, str):
        _daemon_log(f"model:error invalid_role type={type(role)}")
        return f"[error] Invalid role: {role}"
    
    if role not in MODEL_REGISTRY:
        _daemon_log(f"model:error unknown_role={role}")
        return f"[error] Unknown role: {role}"
    
    if fallback_role not in MODEL_REGISTRY:
        _daemon_log(f"model:error unknown_fallback_role={fallback_role}, using orchestrator")
        fallback_role = "orchestrator"
    
    primary_model = MODEL_REGISTRY[role]
    timeout_by_role = {
        "reasoning_primary": 90,
        "reasoning_secondary": 90,
        "reasoning_tertiary": 90,
        "coder": 120,
        "critic": 90,
        "orchestrator": 120,
    }
    if role == "reasoning_primary" and "deepseek" in primary_model.lower():
        fallback_model = MODEL_REGISTRY[fallback_role]
        _daemon_log(
            f"model:quarantine role={role} model={primary_model} "
            f"fallback_model={fallback_model} reason=runtime instability"
        )
        _write_status("reasoning", f"{role} quarantined, using {fallback_model}", progress=0.2, extra={"role": role, "model": fallback_model})
        try:
            response = ask(
                fallback_model,
                prompt,
                system=f"{system}\nThe primary strategist is temporarily unavailable, so cover its role fully.",
                timeout=timeout_by_role.get(fallback_role, 120),
                options=MODEL_OPTIONS[fallback_role],
            )
            _daemon_log(f"model:ok role={role} fallback_model={fallback_model}")
            return response
        except Exception as exc:
            _daemon_log(f"model:failed role={role} quarantined_fallback={type(exc).__name__}: {exc}")
            return f"[{role} quarantined] fallback failed: {type(exc).__name__}: {exc}"
    role_timeout = timeout_by_role.get(role, 120)
    _daemon_log(f"model:start role={role} model={primary_model}")
    _write_status("reasoning", f"Querying {role}: {primary_model}", progress=0.2, extra={"role": role, "model": primary_model})
    try:
        response = ask(
            primary_model,
            prompt,
            system=system,
            timeout=role_timeout,
            options=MODEL_OPTIONS[role],
        )
        _daemon_log(f"model:ok role={role} model={primary_model}")
        return response
    except Exception as exc:
        fallback_model = MODEL_REGISTRY[fallback_role]
        _daemon_log(f"model:fallback role={role} model={primary_model} error={type(exc).__name__}: {exc}")
        fallback_prompt = (
            f"The intended specialist model '{primary_model}' failed with:\n{type(exc).__name__}: {exc}\n\n"
            f"Please cover this role instead.\n\n{prompt}"
        )
        try:
            response = ask(
                fallback_model,
                fallback_prompt,
                system=f"{system}\nIf another model fails, step in and keep the NeuroGolf pipeline moving.",
                timeout=timeout_by_role.get(fallback_role, 120),
                options=MODEL_OPTIONS[fallback_role],
            )
            _daemon_log(f"model:ok role={role} fallback_model={fallback_model}")
            return response
        except Exception as fallback_exc:
            _daemon_log(
                f"model:failed role={role} primary={type(exc).__name__}: {exc} "
                f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
            )
            return (
                f"[{role} unavailable] primary={type(exc).__name__}: {exc}; "
                f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
            )


def _team_strategy_briefs(summary: dict) -> dict:
    state = _compact_summary(summary)
    primary = _safe_model_response(
        "reasoning_primary",
        f"""Analyze the current NeuroGolf competition state and propose 3 concrete next experiments.

APRIL 28 2026 METRIC UPDATE: Dynamic shapes and symbolic dimensions now YIELD ZERO POINTS. Constant values now correctly count.
METRIC V3 SCORING: points = 25.0 - log(MACs + memory + params). Minimize the sum for maximum score.

State:
{state}

Focus on:
- Treat old V2 seeds and imported packs as references until local validation proves each task file
- If NO valid seeds exist: Use "fresh_build" mode to create 400 new minimal networks from scratch
- Use big models as teacher/search engines, then compile or distill into tiny static ONNX graphs
- BUILDING MINIMAL computational graphs: small + correct = high score, large + complex = low/zero score
- The 9538 exploit used 1.37M hidden constants + 373 dynamic-shape tasks. Only 19 clean tasks. INVALID now.
- Prefer simple numpy operations over deep neural networks. Geometric transforms > learned weights.
- Use smallest possible weight matrices, minimize intermediate tensors, fewest operations.
- Prefer element-wise ops (Add, Mul) over reduction ops (Sum, Mean). Avoid Conv/MatMul when possible.
- Build -> Test -> Validate -> Keep only what passes V3 validation -> Submit complete 400/400 pack
- avoiding bad local-scorer traps
- finding efficient, leaderboard-safe task swaps
- using imported V3-compliant notebooks as source material when they produce smaller valid graphs

Return a compact numbered list.
""",
        system="You are the primary NeuroGolf strategist. Think like a careful competition engineer. The April 28 2026 metric update requires all models to have statically-defined shapes and minimal computational graphs.",
        fallback_role="orchestrator",
    )
    secondary = _safe_model_response(
        "reasoning_secondary",
        f"""Propose alternative NeuroGolf experiments that differ from the obvious seed-preserving plan.

APRIL 28 2026 METRIC UPDATE: The old 9k+ exploit family used dynamic shapes and now scores ZERO. Focus on valid static-shape alternatives.
METRIC V3 OPTIMIZATION: Build MINIMAL networks. Score = 25 - log(MACs + memory + params). Smaller is better.

State:
{state}

Look for:
- identifying tasks without valid V3 submissions and proposing new static-shape network builds
- overlooked seed choices with clean static shapes
- safe vs risky experiment splits
- paths that create new V3-compliant files rather than reusing invalid V2 files
- extracting teacher insights from valid imported sources, then rebuilding smaller compliant graphs
- EFFICIENCY OPPORTUNITIES: Simple numpy ops > neural networks. Geometric transforms > learned weights.
- Challenge: Can we achieve the same accuracy with 10x fewer parameters and MACs?
- The 9538 exploit used 1.37M hidden constants. Build tiny networks instead.

Return a compact numbered list.
""",
        system="You are the alternate NeuroGolf strategist. Generate distinct ideas, not paraphrases. The April 28 metric update closed dynamic-shape exploits. Focus on minimal computational graphs.",
        fallback_role="orchestrator",
    )
    tertiary = _safe_model_response(
        "reasoning_tertiary",
        f"""Provide a third NeuroGolf strategy pass that focuses on resource fusion and efficiency.

APRIL 28 2026 METRIC UPDATE: Dynamic-shape and Constant-node exploits are CLOSED. All models need static shapes and correct parameter counting.
METRIC V3 SCORING: points = 25 - log(MACs + memory + params). Build MINIMAL networks for maximum score.

State:
{state}

Your job:
- identify tasks that need NEW V3-compliant network files (not present in current packs)
- find gaps in current submissions where static-shape networks must be built from scratch
- combine Kaggle imports, local manifests, prior lessons, and public seed behavior
- identify which imported sources are V3 compliant vs invalid V2-era files
- suggest strategies to build missing V3 files rather than rely on old submissions
- EFFICIENCY FOCUS: Find ways to minimize MACs, memory, and params while maintaining correctness
- INSIGHT: The 9538 exploit had 1.37M constants. Build tiny networks. Simple > complex.
- Question everything: Can we solve this with geometric transforms instead of learned weights?

Return a compact numbered list.
""",
        system="You are the third NeuroGolf strategist. Fuse external Kaggle signals with local evidence. The April 28 metric update requires static-shape compliance and minimal computational graphs.",
        fallback_role="orchestrator",
    )
    coder_brief = _safe_model_response(
        "coder",
        f"""Turn these NeuroGolf strategy notes into an implementation checklist.

State:
{state}

Primary ideas:
{primary}

Secondary ideas:
{secondary}

Third strategist ideas:
{tertiary}

Return plain text with:
- candidate experiment branches
- likely files/scripts to touch
- safe automation steps
- top 3 experiments to run next
""",
        system="You are the implementation strategist for NeuroGolf automation. Write concrete experiment and patch checklists.",
        fallback_role="orchestrator",
    )
    critique = _safe_model_response(
        "critic",
        f"""Critique the following NeuroGolf strategy bundle.

State:
{state}

Primary ideas:
{primary}

Secondary ideas:
{secondary}

Third strategist ideas:
{tertiary}

Implementation checklist:
{coder_brief}

Return the 3 biggest risks or likely mistakes.
""",
        system="You are the NeuroGolf critic. Attack risky assumptions and repetitive experiment choices.",
        fallback_role="orchestrator",
    )
    return {
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
        "coder": coder_brief,
        "critic": critique,
    }


def _normalize_plan(plan: dict, reports: list[dict], summary: dict) -> dict:
    plan = _coerce_plan_schema(plan, summary)
    action = str(plan.get("action", "seed_preserving_build")).strip() or "seed_preserving_build"
    recent_keys = _recent_experiment_keys(reports)
    available_labels = {label for label, _score in _available_seed_candidates(summary)}
    default_seed = _default_seed_label(summary)
    cached_targets = {item["target"] for item in _available_cached_candidates(summary)}

    if action == "direct_seed_pack":
        direct_candidates = _available_direct_seed_candidates(summary)
        direct_scores = _direct_seed_score_map(summary)
        direct_labels = set(direct_scores)
        direct_default = direct_candidates[0][0] if direct_candidates else default_seed
        seed = str(plan.get("seed_label", "")).strip() or direct_default
        if seed not in direct_labels and direct_labels:
            plan = dict(plan)
            plan["seed_label"] = direct_default
            plan["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to strongest buildable direct seed"
            seed = direct_default
        elif direct_candidates:
            top_seed, top_score = direct_candidates[0]
            chosen_score = float(direct_scores.get(seed, 0.0))
            if top_seed != seed and top_score >= chosen_score + 50.0:
                plan = dict(plan)
                plan["seed_label"] = top_seed
                plan["target"] = top_seed
                plan["rationale"] = str(plan.get("rationale", "")).strip() + " | upgraded to the strongest imported direct seed"
                seed = top_seed
        # Imported seeds are reference material. They must pass the same local ONNX
        # validation as generated builds before a later cycle can submit them.
        plan = dict(plan)
        plan["should_submit"] = False
        plan["rationale"] = str(plan.get("rationale", "")).strip() + " | validation-first: imported seed staged for inspection"
        key = _experiment_key({"action": action, "seed_label": seed, "variant_hint": ""})
        if not _should_avoid_branch(summary, key, recent_keys):
            return plan
        for candidate in _fallback_experiments(summary):
            candidate_key = _experiment_key(candidate)
            if not _should_avoid_branch(summary, candidate_key, recent_keys):
                candidate = dict(candidate)
                candidate["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to a fresh experiment branch"
                return candidate
        return plan

    if action == "cached_candidate_submit":
        target = str(plan.get("target", "")).strip()
        if not target or target not in cached_targets:
            best_cached = _available_cached_candidates(summary)
            if best_cached:
                plan = dict(plan)
                plan["target"] = best_cached[0]["target"]
                plan["variant_hint"] = str(best_cached[0].get("variant", ""))
                plan["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to the strongest cached candidate"
                target = plan["target"]
        plan = dict(plan)
        plan["should_submit"] = False
        plan["rationale"] = str(plan.get("rationale", "")).strip() + " | validation-first: cached candidate staged for inspection"
        key = _experiment_key({"action": action, "target": target, "variant_hint": str(plan.get("variant_hint", "")).strip()})
        if not _should_avoid_branch(summary, key, recent_keys):
            return plan
        for candidate in _fallback_experiments(summary):
            candidate_key = _experiment_key(candidate)
            if not _should_avoid_branch(summary, candidate_key, recent_keys):
                candidate = dict(candidate)
                candidate["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to a fresh experiment branch"
                return candidate
        return plan

    if action == "scorer_blend_build":
        mode = _normalize_blend_variant(plan.get("mode", "strict_known"))
        plan["mode"] = mode
        plan["variant_hint"] = mode
        key = _experiment_key(plan)
        if not _should_avoid_branch(summary, key, recent_keys):
            plan["seed_label"] = ""
            return plan
        for candidate_mode in ("strict_known", "skip_known_dynamic_sanitized", "skip_known_dynamic"):
            candidate_key = _experiment_key({"action": action, "mode": candidate_mode})
            if not _should_avoid_branch(summary, candidate_key, recent_keys):
                plan = dict(plan)
                plan["mode"] = candidate_mode
                plan["variant_hint"] = candidate_mode
                plan["seed_label"] = ""
                plan["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to a fresh static-build branch"
                return plan
        return plan

    seed = str(plan.get("seed_label", default_seed)).strip() or default_seed
    mode = _normalize_seed_variant(plan.get("mode", "processable_best"))
    plan["mode"] = mode
    plan["variant_hint"] = mode
    if seed not in available_labels and available_labels:
        plan = dict(plan)
        plan["seed_label"] = default_seed
        plan["mode"] = "processable_best"
        plan["variant_hint"] = "processable_best"
        plan["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to the strongest currently buildable seed"
        return plan
    key = _experiment_key(plan)
    if not _should_avoid_branch(summary, key, recent_keys):
        return plan
    for candidate in _fallback_experiments(summary):
        candidate_key = _experiment_key(candidate)
        if not _should_avoid_branch(summary, candidate_key, recent_keys):
            candidate = dict(candidate)
            candidate["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to a fresh experiment branch"
            return candidate
    return plan


def _experiment_fingerprint(plan: dict, build_manifest: dict) -> str:
    payload = {
        "action": plan.get("action"),
        "seed_label": plan.get("seed_label"),
        "mode": plan.get("mode"),
        "replaced_count": build_manifest.get("replaced_count"),
        "replacements": build_manifest.get("replacements", []),
        "zip_size_bytes": build_manifest.get("zip_size_bytes"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _candidate_signature(manifest: dict) -> str:
    tasks: list[str] = []
    for item in manifest.get("replacements", []):
        task = str(item.get("task", "")).strip()
        if task:
            tasks.append(task)
    for item in manifest.get("replaced", []):
        task = str(item.get("task", "")).strip()
        if task:
            tasks.append(task)
    payload = {
        "build_strategy": manifest.get("build_strategy"),
        "base_zip": manifest.get("seed_zip") or manifest.get("base_zip"),
        "mode": manifest.get("mode"),
        "preserved_task000": manifest.get("preserved_task000"),
        "replaced_count": manifest.get("replaced_count"),
        "tasks": sorted(set(tasks)),
        "zip_size_bytes": manifest.get("zip_size_bytes"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _lookup_imported_seed(summary: dict, seed_label: str) -> dict | None:
    for row in summary.get("imported_sources", []):
        if str(row.get("seed_label", "")).strip() == seed_label and row.get("usable_as_seed"):
            return row
    return None


def _find_cached_seed_manifest(seed_label: str, mode: str) -> dict | None:
    outputs_dir = NEUROGOLF_PROJECT_ROOT / "outputs"
    for path in sorted(outputs_dir.glob("*manifest*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("seed_label") != seed_label or payload.get("mode") != mode:
            continue
        zip_path = Path(str(payload.get("zip_path", "")).strip())
        if not zip_path.exists():
            continue
        payload["_manifest_path"] = str(path)
        return payload
    return None


def _find_cached_blend_manifest(mode: str) -> dict | None:
    outputs_dir = NEUROGOLF_PROJECT_ROOT / "outputs"
    for path in sorted(outputs_dir.glob("*manifest*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("build_strategy") != "scorer_blend":
            continue
        if payload.get("mode") != mode:
            continue
        zip_path = Path(str(payload.get("zip_path", "")).strip())
        if not zip_path.exists():
            continue
        payload["_manifest_path"] = str(path)
        return payload
    return None


def _find_cached_candidate(summary: dict, target: str) -> dict | None:
    clean = str(target or "").strip().lower()
    for item in _available_cached_candidates(summary):
        path = Path(str(item.get("zip_path", "")))
        if clean in {item["target"].lower(), path.name.lower(), str(path).lower()}:
            return item
    return None


def plan_next_action(summary: dict) -> dict:
    operator_note = _load_operator_note()
    team = _team_strategy_briefs(summary)
    available_seed_labels = [label for label, _score in _available_seed_candidates(summary)]
    available_direct_seed_labels = [label for label, _score in _available_direct_seed_candidates(summary)]
    default_seed = _default_seed_label(summary)
    _daemon_log("planner:start orchestrator merge")
    _write_status("planning", "Merging strategist outputs", progress=0.6)
    raw = _safe_model_response(
        "orchestrator",
        f"""Decide the next NeuroGolf experiment.

Current state:
{_compact_summary(summary)}

Operator note:
{operator_note or "None"}

Primary strategist:
{team["primary"]}

Alternate strategist:
{team["secondary"]}

Third strategist:
{team["tertiary"]}

Implementation strategist:
{team["coder"]}

Critic:
{team["critic"]}

Return JSON only with:
- experiment_type: one of ["direct_seed", "cached_candidate", "surgical_splice", "open_blend", "sync_only", "fresh_build"]
- target:
  - for direct_seed: one of {json.dumps(available_direct_seed_labels)}
  - for surgical_splice: one of {json.dumps(available_seed_labels)}
  - for cached_candidate: a cached zip name if you want to reuse a previous build
  - for open_blend, sync_only, or fresh_build: empty string
- variant_hint:
  - for surgical_splice: optional hint like "safer", "aggressive", or "repair-heavy"
  - for open_blend: optional hint like "strict", "known", or "sanitized" (NOTE: "dynamic" exploit no longer works)
  - otherwise: empty string
- should_submit: true/false
- rationale: short string
- submission_message: short string

APRIL 28 2026 METRIC UPDATE - CRITICAL:
Dynamic shapes and symbolic dimensions now YIELD ZERO POINTS. The old 9k+ exploit family is INVALID.
Constant values now correctly count toward parameters. Hidden constant exploitation NO LONGER WORKS.
All models MUST have statically-defined shapes. Invalid networks (failed shape inference) score ZERO.

INTENDED-COMPETITION MODE:
- Treat imported notebooks and seed zips as references, not trusted submissions
- The system should build or compile new V3-compliant static graphs whenever possible
- Use "fresh_build" experiment_type to build minimal computational graphs for all 400 tasks
- Build the smallest networks that correctly solve each task (score = 25 - log(MACs + memory + params))
- Test each build - only keep tasks that pass V3 validation
- Assemble valid tasks into a complete 400/400 submission

SUBMISSION VALIDATION - HARD BLOCKS:
- Every submission zip must pass local ONNX validation before Kaggle submission
- Direct seeds and cached candidates are never exempt from validation
- Do not chase metric bugs, hidden constants, symbolic dimensions, or dynamic-shape behavior

TARGET REQUIREMENTS DISABLED:
- Target score and consecutive best streak requirements are DISABLED
- The system will NO LONGER wait for specific scores or streaks before submitting
- All valid V3-compliant submissions are allowed immediately
- No more blocking due to "target not reached" or "streak not achieved"

KAGGLE ERROR HANDLING - CRITICAL:
- When submissions fail, ALWAYS fetch error details from Kaggle
- Kaggle error messages often specify which tasks failed (e.g., "task042 failed validation")
- Parse these errors to identify failed tasks and add them to invalid_tasks list
- Replace failed tasks in subsequent builds using skip_known_dynamic mode
- Use error details to guide which tasks need rebuilding

CRITICAL ACTIONS:
1. If NO valid seeds exist: Use "fresh_build" mode to create new minimal networks from scratch
2. Use big models only as teachers/searchers that infer rules; distill into tiny static ONNX graphs
3. Build minimal computational graphs - small + correct = high score
4. Test and validate each task - only keep what passes V3 validation
5. When you have 400 locally valid tasks, set should_submit: true and submit
6. On submission failure: fetch Kaggle error details, identify failed tasks, fix and resubmit

Default behavior:
- choose the strongest next branch from all available material instead of defaulting to a single seed family
- treat autonomy_learning.blocked_branches and autonomy_learning.recent_streak as hard warnings from prior mistakes
- if the recent streak shows repeated duplicate/already-submitted cycles, choose a different action, seed, or normalized variant
- prefer strict fresh/static graph building over direct imported seed submission
- keep task000
- use imported Kaggle assets only as source material when they materially improve graph quality AND pass static validation
- reuse earlier candidate zips ONLY if they are V3-compliant (static shapes)
- AVOID submissions with dynamic shapes or heavy constant nodes - these now score ZERO
- prioritize metric v3 compliant seeds (static shapes, valid parameter counts) over historically high-scoring but now-invalid submissions
- scorer_blend_build should focus on combining VALID sources, not exploiting metric quirks
- BUILD new files for missing tasks rather than submitting incomplete V2 packs
- avoid repeating the same branch just because it is easy to express as a seed plus variant
""",
        system=NEUROGOLF_AUTONOMY_SYSTEM_PROMPT,
        fallback_role="operator_fast",
    )
    _daemon_log("planner:done orchestrator merge")
    _write_status("planning", "Planner selected next experiment", progress=0.72)
    parsed = _extract_json_object(raw)
    if not parsed:
        direct_candidates = _available_direct_seed_candidates(summary)
        top_seed = direct_candidates[0] if direct_candidates else _top_buildable_seed(summary)
        best_completed = summary.get("best_completed_public_score")
        seed_label = default_seed
        if top_seed is not None:
            seed_label = top_seed[0]
            top_score = float(top_seed[1])
            should_submit = best_completed is None or top_score >= float(best_completed) + 100.0
            submission_message = f"Autonomy fallback tries seed {seed_label}."
            return {
                "action": "direct_seed_pack",
                "seed_label": seed_label,
                "target": seed_label,
                "variant_hint": "",
                "should_submit": should_submit,
                "rationale": "Fallback plan with score-aware direct-seed preference",
                "submission_message": submission_message,
                "raw_plan": raw,
                "team_strategy": team,
            }
        else:
            # NO VALID SEEDS - Use fresh_build to create 400 new minimal networks from scratch
            return {
                "action": "scorer_blend_build",
                "seed_label": "",
                "target": "",
                "variant_hint": "strict_known",
                "mode": "strict_known",
                "should_submit": False,  # Build first, validate, then submit
                "rationale": "FRESH BUILD: No valid seeds available. Building 400 new minimal networks from scratch.",
                "submission_message": "Fresh build - creating new V3-compliant networks from scratch",
                "raw_plan": raw,
                "team_strategy": team,
            }
    parsed["raw_plan"] = raw
    parsed["team_strategy"] = team
    return parsed


def _zip_dir_of_onnx(source_dir: Path, output_zip: Path) -> tuple[int, bool]:
    task_files = sorted(source_dir.glob("task*.onnx"))
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in task_files:
            archive.write(path, path.name)
    preserved_task000 = any(path.name == "task000.onnx" for path in task_files)
    return len(task_files), preserved_task000


def _inspect_zip(zip_path: Path) -> tuple[int, bool]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [Path(name).name for name in archive.namelist()]
    task_names = [name for name in names if re.fullmatch(r"task\d{3}\.onnx", name)]
    return len(task_names), "task000.onnx" in task_names


def run_direct_seed_pack(summary: dict, zip_name: str, seed_label: str) -> dict:
    source = _lookup_imported_seed(summary, seed_label)
    if source is None:
        raise SystemExit(f"Direct seed artifact not available for {seed_label}")
    artifact_path = Path(str(source["artifact_path"]))
    output_zip = NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name
    manifest_path = NEUROGOLF_PROJECT_ROOT / "outputs" / f"{Path(zip_name).stem}_manifest.json"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if str(source.get("seed_kind")) == "zip":
        shutil.copyfile(artifact_path, output_zip)
        file_count, preserved_task000 = _inspect_zip(output_zip)
    else:
        file_count, preserved_task000 = _zip_dir_of_onnx(artifact_path, output_zip)
    manifest = {
        "build_strategy": "direct_seed_pack",
        "seed_label": seed_label,
        "target": seed_label,
        "variant_hint": "",
        "seed_kind": source.get("seed_kind"),
        "seed_claimed_public_score": source.get("claimed_public_score"),
        "artifact_path": str(artifact_path),
        "zip_path": str(output_zip),
        "zip_size_bytes": output_zip.stat().st_size,
        "files": file_count,
        "preserved_task000": preserved_task000,
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _daemon_log(f"build:direct-seed seed={seed_label} files={file_count} zip={output_zip.name}")
    _write_status("building", f"Using direct imported seed {seed_label}", progress=0.88, extra={"seed_label": seed_label, "target": seed_label, "variant": "", "build_strategy": "direct_seed_pack"})
    return manifest


def run_cached_candidate_submit(summary: dict, zip_name: str, target: str) -> dict:
    cached = _find_cached_candidate(summary, target)
    if cached is None:
        raise SystemExit(f"Cached candidate not found: {target}")
    source_zip = Path(str(cached["zip_path"]))
    output_zip = NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name
    manifest_path = NEUROGOLF_PROJECT_ROOT / "outputs" / f"{Path(zip_name).stem}_manifest.json"
    if source_zip.resolve() != output_zip.resolve():
        shutil.copyfile(source_zip, output_zip)
    file_count, preserved_task000 = _inspect_zip(output_zip)
    manifest = {
        "build_strategy": "cached_candidate_submit",
        "target": cached["target"],
        "variant_hint": cached.get("variant", ""),
        "seed_label": cached.get("seed_label"),
        "source_zip": str(source_zip),
        "zip_path": str(output_zip),
        "zip_size_bytes": output_zip.stat().st_size,
        "files": file_count,
        "preserved_task000": preserved_task000,
        "score_hint": cached.get("score_hint", 0.0),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _daemon_log(f"build:cached-candidate target={cached['target']} zip={output_zip.name}")
    _write_status("building", f"Reusing cached candidate {cached['target']}", progress=0.88, extra={"target": cached["target"], "variant": cached.get("variant", ""), "build_strategy": "cached_candidate_submit"})
    return manifest


def run_seed_preserving_build(zip_name: str, seed_label: str, mode: str) -> dict:
    cached = _find_cached_seed_manifest(seed_label, mode)
    output_zip = NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name
    manifest_path = NEUROGOLF_PROJECT_ROOT / "outputs" / f"{Path(zip_name).stem}_manifest.json"
    if cached is not None:
        cached_zip = Path(str(cached["zip_path"]))
        if cached_zip.resolve() != output_zip.resolve():
            shutil.copyfile(cached_zip, output_zip)
        manifest = dict(cached)
        manifest["build_strategy"] = "seed_preserving"
        manifest["zip_path"] = str(output_zip)
        manifest["zip_size_bytes"] = output_zip.stat().st_size
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _daemon_log(
            "build:cache-hit "
            f"seed={seed_label} mode={mode} source={cached.get('_manifest_path')} zip={output_zip.name}"
        )
        _write_status("building", f"Reusing cached splice from {seed_label} ({_seed_variant_label(mode)})", progress=0.84, extra={"seed_label": seed_label, "target": seed_label, "variant": mode, "build_strategy": "seed_preserving"})
        return manifest

    _daemon_log(f"build:start seed={seed_label} mode={mode} zip={zip_name}")
    _write_status("building", f"Building splice from {seed_label} ({_seed_variant_label(mode)})", progress=0.8, extra={"seed_label": seed_label, "target": seed_label, "variant": mode, "build_strategy": "seed_preserving"})
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(NEUROGOLF_PACKAGE_SCRIPT),
        "-BuildStrategy",
        "seed_preserving",
        "-SeedLabel",
        seed_label,
        "-SeedMode",
        mode,
        "-ZipName",
        zip_name,
    ]
    _run_external_build(command, label="seed_preserving", target=seed_label, variant=mode)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_strategy"] = "seed_preserving"
    manifest["manifest_path"] = str(manifest_path)
    _daemon_log(
        "build:done "
        f"seed={seed_label} mode={mode} replaced={manifest.get('replaced_count')} "
        f"delta={manifest.get('delta_sum_local')} files={manifest.get('files')}"
    )
    _write_status("building", f"Finished splice from {seed_label} ({_seed_variant_label(mode)})", progress=0.88, extra={"seed_label": seed_label, "target": seed_label, "variant": mode, "build_strategy": "seed_preserving"})
    return manifest


def run_scorer_blend_build(zip_name: str, mode: str) -> dict:
    cached = _find_cached_blend_manifest(mode)
    output_zip = NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name
    manifest_path = NEUROGOLF_PROJECT_ROOT / "outputs" / f"{Path(zip_name).stem}_manifest.json"
    if cached is not None:
        cached_zip = Path(str(cached["zip_path"]))
        if cached_zip.resolve() != output_zip.resolve():
            shutil.copyfile(cached_zip, output_zip)
        manifest = dict(cached)
        manifest["build_strategy"] = "scorer_blend"
        manifest["zip_path"] = str(output_zip)
        manifest["zip_size_bytes"] = output_zip.stat().st_size
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _daemon_log(
            "build:cache-hit "
            f"strategy=scorer_blend mode={mode} source={cached.get('_manifest_path')} zip={output_zip.name}"
        )
        _write_status("building", f"Reusing cached open blend ({_blend_variant_label(mode)})", progress=0.84, extra={"variant": mode, "build_strategy": "scorer_blend"})
        return manifest

    include_sanitized = mode == "skip_known_dynamic_sanitized"
    skip_known = mode in {"skip_known_dynamic", "skip_known_dynamic_sanitized"}
    _daemon_log(f"build:start strategy=scorer_blend mode={mode} zip={zip_name}")
    _write_status("building", f"Building open blend ({_blend_variant_label(mode)})", progress=0.8, extra={"variant": mode, "build_strategy": "scorer_blend"})
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(NEUROGOLF_PACKAGE_SCRIPT),
        "-BuildStrategy",
        "scorer_blend",
        "-ZipName",
        zip_name,
    ]
    if skip_known:
        command.append("-SkipKnownValidation")
    if include_sanitized:
        command.append("-IncludeSanitized")
    _run_external_build(command, label="scorer_blend", target="open_blend", variant=mode)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_strategy"] = "scorer_blend"
    manifest["mode"] = mode
    manifest["manifest_path"] = str(manifest_path)
    _daemon_log(
        "build:done "
        f"strategy=scorer_blend mode={mode} known={manifest.get('known_score')} "
        f"live={manifest.get('estimated_live_score')} files={manifest.get('files')}"
    )
    _write_status("building", f"Finished open blend ({_blend_variant_label(mode)})", progress=0.88, extra={"variant": mode, "build_strategy": "scorer_blend"})
    return manifest


def maybe_submit(zip_name: str, message: str, manifest: dict, min_local_delta: float, *, bypass_local_delta: bool = False) -> dict:
    if not bypass_local_delta and float(manifest.get("delta_sum_local", 0.0)) < min_local_delta:
        _write_status("submitting", "Submission skipped by local delta gate", progress=0.93)
        return {"submitted": False, "reason": f"delta_sum_local below threshold {min_local_delta:.2f}"}
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "submit_to_kaggle.py"),
        "--file",
        str(NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name),
        "--message",
        message,
        "--force",
    ]
    _daemon_log(f"submit:start zip={zip_name} message={message}")
    _write_status("submitting", f"Submitting {zip_name} to Kaggle", progress=0.94, extra={"message": message})
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "submitted": False,
            "reason": "submit command failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:] if completed.stdout else "",
            "stderr": completed.stderr[-4000:] if completed.stderr else "",
        }
    _daemon_log(f"submit:done zip={zip_name}")
    _write_status("submitting", f"Submission accepted for polling: {zip_name}", progress=0.97, extra={"message": message})
    return {
        "submitted": True,
        "stdout": completed.stdout[-4000:] if completed.stdout else "",
    }


def _materialize_plan(plan: dict, summary: dict, campaign: dict) -> dict:
    action = str(plan.get("action", "seed_preserving_build")).strip() or "seed_preserving_build"
    mode = str(plan.get("mode", "processable_best")).strip() or "processable_best"
    # Get seed_label from plan or use default (which may be None if no valid seeds)
    plan_seed = plan.get("seed_label")
    seed_label = str(plan_seed).strip() if plan_seed else ""
    target = str(plan.get("target", "")).strip()
    variant_hint = str(plan.get("variant_hint", "")).strip()
    display_variant = variant_hint or mode
    if action == "scorer_blend_build":
        seed_label = ""
    if not seed_label and action == "seed_preserving_build":
        default_seed = _default_seed_label(summary)
        if default_seed:
            seed_label = default_seed
        else:
            # No valid seeds available - force blend build to create fresh networks
            action = "scorer_blend_build"
            mode = "strict_known"
            seed_label = ""
            display_variant = mode

    top_seed = _top_buildable_seed(summary)
    if action == "seed_preserving_build" and top_seed is not None:
        top_seed_label, top_seed_score = top_seed
        best_completed = summary.get("best_completed_public_score")
        if (
            not bool(plan.get("should_submit", False))
            and seed_label == top_seed_label
            and (best_completed is None or float(top_seed_score) >= float(best_completed) + 100.0)
        ):
            plan["should_submit"] = False
            plan["rationale"] = str(plan.get("rationale", "")).strip() + " | held for local ONNX validation before any high-score seed submission"
            if not str(plan.get("submission_message", "")).strip():
                plan["submission_message"] = f"Validation-held imported high-score seed {seed_label}."
        elif (
            not bool(plan.get("should_submit", False))
            and seed_label == top_seed_label
            and mode in {"hybrid_priority", "fill_invalid_priority"}
            and float(top_seed_score) >= float(best_completed or 0.0)
        ):
            plan["should_submit"] = False
            plan["rationale"] = (
                str(plan.get("rationale", "")).strip()
                + " | held a high-claim splice branch for local ONNX validation before submission"
            ).strip()
            if not str(plan.get("submission_message", "")).strip():
                pretty_variant = "hybrid splice" if mode == "hybrid_priority" else "repair splice"
                plan["submission_message"] = f"Validation-held {pretty_variant} from imported top seed {seed_label}."

    # DISABLED: Campaign target and streak goals no longer block submissions
    # Previously checked: if campaign and not campaign.get("goal_reached"):
    # Now: Always allow submissions regardless of score or streak
    # The goal_reached is always True when targets are disabled (set to 0)

    submit_target = target or seed_label or "blend"
    submit_message = str(plan.get("submission_message", "")).strip() or f"autonomy {action} {submit_target} {display_variant}".strip()
    return {
        "action": action,
        "mode": mode,
        "seed_label": seed_label,
        "target": target,
        "display_variant": display_variant,
        "submit_target": submit_target,
        "submit_message": submit_message,
        "experiment_label": _display_target(action, target, seed_label, display_variant),
    }


def _build_from_plan(args: argparse.Namespace, summary: dict, context: dict) -> dict:
    action = context["action"]
    mode = context["mode"]
    seed_label = context["seed_label"]
    target = context["target"]
    if action == "scorer_blend_build":
        return run_scorer_blend_build(args.zip_name, mode)
    if action == "direct_seed_pack":
        return run_direct_seed_pack(summary, args.zip_name, seed_label or target or _default_seed_label(summary))
    if action == "cached_candidate_submit":
        return run_cached_candidate_submit(summary, args.zip_name, target)
    return run_seed_preserving_build(args.zip_name, seed_label, mode)


def _validate_v3_compliance(build_manifest: dict, plan: dict | None = None) -> dict:
    """Validate that submission is Metric V3 compliant (no dynamic shapes, valid static inference).
    
    Args:
        build_manifest: The manifest from the build
        plan: The plan that generated this build (to check for skip modes)
    
    Returns dict with valid=True/False and reason if invalid.
    """
    zip_path_text = str(build_manifest.get("zip_path", "")).strip()
    if zip_path_text:
        zip_path = Path(zip_path_text)
        validation_report = validate_submission_zip(zip_path)
        report_path = zip_path.with_name(f"{zip_path.stem}_v3_validation.json")
        try:
            write_validation_report(validation_report, report_path)
        except Exception as exc:
            validation_report.setdefault("warnings", []).append(f"could not write validation report: {exc}")
        if not validation_report["valid"]:
            missing_tasks = validation_report.get("missing_tasks", [])
            invalid_reports = validation_report.get("invalid_task_reports", [])
            invalid_tasks = [
                str(item.get("name") or item.get("archive_member") or "unknown")
                for item in invalid_reports
            ]
            reason_parts = list(validation_report.get("errors", []))
            if not reason_parts:
                reason_parts.append("submission zip failed local ONNX validation")
            return {
                "valid": False,
                "reason": "; ".join(reason_parts),
                "invalid_tasks": invalid_tasks,
                "dynamic_tasks": [],
                "missing_tasks": missing_tasks,
                "validation_report_path": str(report_path),
                "zip_validation": validation_report,
            }
        return {
            "valid": True,
            "reason": None,
            "invalid_tasks": [],
            "dynamic_tasks": [],
            "validation_report_path": str(report_path),
            "zip_validation": validation_report,
        }
    
    # Check for dynamic shapes in manifest
    task_statuses = build_manifest.get("task_statuses", [])
    invalid_tasks = []
    dynamic_shape_tasks = []
    
    for task in task_statuses:
        task_id = task.get("task_id", "unknown")
        # Check for invalid status
        if task.get("status") == "invalid":
            invalid_tasks.append(task_id)
        # Check for dynamic shapes (no shape inference or symbolic dims)
        onnx_info = task.get("onnx_info", {})
        if onnx_info.get("has_dynamic_shapes", False):
            dynamic_shape_tasks.append(task_id)
        if onnx_info.get("has_symbolic_dims", False):
            dynamic_shape_tasks.append(task_id)
        # Check if shape inference failed
        if not onnx_info.get("static_shape_inference_passed", True):
            invalid_tasks.append(task_id)
    
    # Without a zip, use manifest-only validation as a conservative fallback.
    invalid_base_tasks = build_manifest.get("invalid_base_tasks", [])
    if invalid_base_tasks:
        invalid_count = len(invalid_base_tasks)
        return {
            "valid": False,
            "reason": f"V2-era invalid base tasks present ({invalid_count} tasks). Replace them before submission.",
            "invalid_tasks": invalid_base_tasks,
            "dynamic_tasks": dynamic_shape_tasks,
        }
    
    if dynamic_shape_tasks:
        return {
            "valid": False,
            "reason": f"Dynamic shapes detected in {len(dynamic_shape_tasks)} tasks. Metric V3 requires static shapes.",
            "invalid_tasks": invalid_tasks,
            "dynamic_tasks": list(set(dynamic_shape_tasks)),
        }
    
    if invalid_tasks:
        return {
            "valid": False,
            "reason": f"Invalid tasks detected ({len(invalid_tasks)}). Shape inference failed or V2-era exploits present.",
            "invalid_tasks": invalid_tasks,
            "dynamic_tasks": dynamic_shape_tasks,
        }
    
    return {"valid": True, "reason": None, "invalid_tasks": [], "dynamic_tasks": []}


def _submission_decision(
    args: argparse.Namespace,
    summary: dict,
    reports: list[dict],
    plan: dict,
    build_manifest: dict,
    experiment_fingerprint: str,
    candidate_signature: str,
    submit_message: str,
) -> dict:
    current_manifest = summary.get("current_manifest") if isinstance(summary.get("current_manifest"), dict) else {}
    matches_current = bool(current_manifest) and candidate_signature == _candidate_signature(current_manifest)
    recent_descriptions = {
        str(item.get("description", "")).strip()
        for item in summary.get("recent_submissions", [])
        if str(item.get("description", "")).strip()
    }
    already_submitted = experiment_fingerprint in _submitted_fingerprints(reports)

    if args.allow_submit and bool(plan.get("should_submit", False)):
        if int(summary.get("pending_submission_count", 0)) >= args.max_pending_submissions:
            _write_status("idle", "Waiting: pending submission cap reached", progress=1.0)
            return {"submitted": False, "reason": "pending submission cap reached"}
        if matches_current:
            _write_status("idle", "Skipped: candidate matches active pack", progress=1.0)
            return {"submitted": False, "reason": "candidate matches current active pack"}
        if submit_message in recent_descriptions:
            _write_status("idle", "Skipped: recent submission already used this message", progress=1.0)
            return {"submitted": False, "reason": "submission message already exists in recent history"}
        if already_submitted:
            _write_status("idle", "Skipped: experiment already submitted", progress=1.0)
            return {"submitted": False, "reason": "experiment already submitted"}
        
        # CRITICAL: Validate Metric V3 compliance before submission
        v3_validation = _validate_v3_compliance(build_manifest, plan)
        if not v3_validation["valid"]:
            _write_status("idle", f"BLOCKED: {v3_validation['reason']}", progress=1.0)
            _daemon_log(f"submission:blocked_v2 {v3_validation['reason']}")
            return {
                "submitted": False, 
                "reason": f"V3_VALIDATION_FAILED: {v3_validation['reason']}",
                "v3_validation": v3_validation,
            }
        
        return maybe_submit(
            args.zip_name,
            submit_message,
            build_manifest,
            args.min_local_delta,
            bypass_local_delta=_should_bypass_local_delta_gate(str(plan.get("action", "")), plan, summary),
        )
    if args.allow_submit:
        _write_status("idle", "Planner held submission", progress=1.0)
        return {"submitted": False, "reason": "planner held submission"}
    _write_status("idle", "Submission disabled for this run", progress=1.0)
    return {"submitted": False, "reason": "submission disabled"}


def _health_check() -> dict:
    """Perform system health check before running cycle."""
    health = {"healthy": True, "issues": []}
    
    # Check if reports dir is writable
    try:
        NEUROGOLF_AUTONOMY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        test_file = NEUROGOLF_AUTONOMY_REPORTS_DIR / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
    except Exception as e:
        health["healthy"] = False
        health["issues"].append(f"Reports directory not writable: {e}")
    
    # Check if status file is writable
    try:
        NEUROGOLF_AUTONOMY_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        health["healthy"] = False
        health["issues"].append(f"Status directory not writable: {e}")
    
    return health


def run_cycle(args: argparse.Namespace) -> dict:
    # Health check first
    health = _health_check()
    if not health["healthy"]:
        _daemon_log(f"cycle:health_check_failed issues={health['issues']}")
        _write_status("error", f"Health check failed: {health['issues']}", progress=1.0)
        return {"error": "Health check failed", "issues": health["issues"]}
    
    summary = sync_neurogolf_state(limit_history=args.history)
    campaign = summary.get("campaign_progress", {}) if isinstance(summary.get("campaign_progress"), dict) else {}
    _daemon_log(
        f"cycle:start best_public={summary.get('best_completed_public_score')} "
        f"pending={summary.get('pending_submission_count')}"
    )
    _write_status(
        "syncing",
        "Refreshing NeuroGolf state from outputs and Kaggle",
        progress=0.05,
        extra={
            "best_completed_public_score": summary.get("best_completed_public_score"),
            "pending_submission_count": summary.get("pending_submission_count"),
            "campaign_progress": campaign,
        },
    )
    if args.sync_only:
        _write_status("idle", "Sync complete", progress=1.0)
        return {"synced": True, "best_completed_public_score": summary.get("best_completed_public_score")}

    reports = _recent_reports()
    learning = _learning_from_reports(reports)
    _persist_learning_state(learning)
    summary = dict(summary)
    summary["autonomy_learning"] = learning
    plan = plan_next_action(summary)
    plan = _normalize_plan(plan, reports, summary)
    action = str(plan.get("action", "seed_preserving_build")).strip() or "seed_preserving_build"
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "plan": plan,
        "best_completed_public_score": summary.get("best_completed_public_score"),
        "pending_submission_count": summary.get("pending_submission_count"),
        "campaign_progress": campaign,
        "learning": learning,
    }

    if action == "sync_only":
        report["result"] = {"synced": True}
        report["submit_result"] = {"submitted": False, "reason": "sync-only cycle"}
        _daemon_log("cycle:planner chose sync-only")
        _write_status("idle", "Planner chose sync-only", progress=1.0)
    else:
        attempts: list[dict] = []
        tried_keys: set[tuple[str, str, str]] = set()
        max_attempts = 2 if args.allow_submit else 1
        final_attempt: dict | None = None

        for attempt_index in range(max_attempts):
            plan = _normalize_plan(plan, reports, summary)
            plan_key = _experiment_key(plan)
            if plan_key in tried_keys:
                alternate = _choose_fresh_fallback_plan(summary, reports, tried_keys)
                if alternate is None:
                    break
                plan = _normalize_plan(alternate, reports, summary)
                plan_key = _experiment_key(plan)
            tried_keys.add(plan_key)

            context = _materialize_plan(plan, summary, campaign)
            _daemon_log(
                f"cycle:plan attempt={attempt_index + 1} action={context['action']} "
                f"target={context['submit_target']} variant={context['display_variant']} "
                f"submit={bool(plan.get('should_submit', False))}"
            )
            _write_status(
                "building",
                f"Selected experiment branch: {context['experiment_label']}",
                progress=0.76,
                extra={
                    "build_strategy": context["action"],
                    "target": context["submit_target"],
                    "variant": context["display_variant"],
                    "seed_label": context["seed_label"],
                    "attempt": attempt_index + 1,
                },
            )

            # Build with error handling
            try:
                build_manifest = _build_from_plan(args, summary, context)
            except Exception as build_exc:
                _daemon_log(f"cycle:build_error attempt={attempt_index + 1} error={type(build_exc).__name__}: {build_exc}")
                attempt_record = {
                    "attempt": attempt_index + 1,
                    "plan": dict(plan),
                    "plan_key": _key_text(plan_key),
                    "build_error": f"{type(build_exc).__name__}: {build_exc}",
                    "submit_result": {"submitted": False, "reason": f"build_failed: {type(build_exc).__name__}"},
                }
                attempts.append(attempt_record)
                if attempt_index + 1 >= max_attempts:
                    break
                alternate = _choose_fresh_fallback_plan(summary, reports, tried_keys)
                if alternate is None:
                    break
                plan = alternate
                continue
            
            experiment_fingerprint = _experiment_fingerprint(plan, build_manifest)
            candidate_signature = _candidate_signature(build_manifest)
            submit_result = _submission_decision(
                args,
                summary,
                reports,
                plan,
                build_manifest,
                experiment_fingerprint,
                candidate_signature,
                context["submit_message"],
            )
            attempt_record = {
                "attempt": attempt_index + 1,
                "plan": dict(plan),
                "plan_key": _key_text(plan_key),
                "build_manifest": build_manifest,
                "experiment_fingerprint": experiment_fingerprint,
                "candidate_signature": candidate_signature,
                "submit_result": submit_result,
            }
            attempts.append(attempt_record)
            final_attempt = attempt_record
            _daemon_log(
                f"cycle:submit attempt={attempt_index + 1} submitted={submit_result.get('submitted')} "
                f"reason={submit_result.get('reason', '')}"
            )

            reason = str(submit_result.get("reason", "")).strip()
            if submit_result.get("submitted") or not _reason_matches(reason, RECOVERABLE_SKIP_REASONS):
                break
            if attempt_index + 1 >= max_attempts:
                break
            alternate = _choose_fresh_fallback_plan(summary, reports, tried_keys)
            if alternate is None:
                break
            _daemon_log(
                f"learning:retry reason={reason} from={_key_text(plan_key)} "
                f"to={_key_text(_experiment_key(alternate))}"
            )
            plan = alternate

        if final_attempt is None:
            if attempts:
                last_attempt = attempts[-1]
                report["plan"] = last_attempt.get("plan", report.get("plan"))
                report["attempts"] = attempts
                report["submit_result"] = last_attempt.get(
                    "submit_result",
                    {"submitted": False, "reason": "no viable experiment branch"},
                )
            else:
                report["submit_result"] = {"submitted": False, "reason": "no viable experiment branch"}
        else:
            report["plan"] = final_attempt["plan"]
            report["build_manifest"] = final_attempt["build_manifest"]
            report["experiment_fingerprint"] = final_attempt["experiment_fingerprint"]
            report["candidate_signature"] = final_attempt["candidate_signature"]
            report["submit_result"] = final_attempt["submit_result"]
            report["attempts"] = attempts
            action = str(final_attempt["plan"].get("action", action))

    NEUROGOLF_AUTONOMY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist_learning_state(_learning_from_reports([report] + reports))
    _daemon_log(f"cycle:report {report_path}")

    # Run self-improvement cycle if enabled
    if is_self_improvement_enabled():
        _write_status("improving", "Analyzing performance for self-improvement", progress=0.95)
        improvement_result = run_self_improvement_cycle([report] + reports)
        _daemon_log(f"self_improvement:result status={improvement_result.get('status')} applied={len(improvement_result.get('applied_changes', []))}")
        report["self_improvement"] = improvement_result

    _write_status("idle", "Cycle complete", progress=1.0, extra={"latest_report": str(report_path)})
    return {"report_path": str(report_path), "action": action, "report": report}


def main() -> int:
    args = build_parser().parse_args()
    if args.loop:
        cycle = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Circuit breaker: stop after 5 consecutive errors
        last_success_time = datetime.now(timezone.utc)
        
        while True:
            cycle += 1
            try:
                result = run_cycle(args)
                print(json.dumps(result, indent=2))
                consecutive_errors = 0  # Reset on success
                last_success_time = datetime.now(timezone.utc)
                
                # Self-healing: check if we're stuck in a loop
                if _is_stuck_in_loop(cycle, last_success_time):
                    _daemon_log("circuit_breaker:stuck_in_loop detected, forcing reset")
                    _write_status("healing", "Detected stuck loop, applying recovery", progress=0.5)
                    _apply_recovery_measures()
                    
            except Exception as exc:
                consecutive_errors += 1
                error_msg = f"{type(exc).__name__}: {exc}"
                
                NEUROGOLF_AUTONOMY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                error_report = {
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "consecutive_errors": consecutive_errors,
                    "cycle": cycle,
                }
                error_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / f"cycle_error_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                error_path.write_text(json.dumps(error_report, indent=2), encoding="utf-8")
                _daemon_log(f"cycle:error {error_msg} consecutive={consecutive_errors} path={error_path}")
                _write_status("error", error_msg, progress=1.0, extra={
                    "error_report": str(error_path),
                    "consecutive_errors": consecutive_errors,
                })
                print(json.dumps({"error_report": str(error_path), "error": error_msg}, indent=2))
                
                # Circuit breaker: too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    _daemon_log(f"circuit_breaker:halting after {consecutive_errors} consecutive errors")
                    _write_status("halted", f"Circuit breaker triggered after {consecutive_errors} errors", progress=1.0)
                    print(json.dumps({"halted": True, "reason": f"{consecutive_errors} consecutive errors"}, indent=2))
                    return 1
                    
                # Progressive backoff on errors
                backoff_seconds = min(300, 30 * (2 ** (consecutive_errors - 1)))
                _daemon_log(f"recovery:backing_off for {backoff_seconds}s after error")
                time.sleep(backoff_seconds)
                continue  # Skip normal sleep, use backoff instead
                
            if args.max_cycles > 0 and cycle >= args.max_cycles:
                break
            time.sleep(max(30, args.sleep_seconds))
        return 0

    result = run_cycle(args)
    print(json.dumps(result, indent=2))
    return 0


def _is_stuck_in_loop(cycle: int, last_success_time: datetime) -> bool:
    """Detect if the system is stuck in a loop with no progress."""
    # Check if more than 2 hours passed since last success
    time_since_success = (datetime.now(timezone.utc) - last_success_time).total_seconds()
    return time_since_success > 7200  # 2 hours


def _apply_recovery_measures() -> None:
    """Apply self-healing measures when stuck."""
    # Clear learning state to force fresh strategies
    learning_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / "learning_state.json"
    if learning_path.exists():
        backup_path = learning_path.with_suffix(f".json.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(learning_path), str(backup_path))
        _daemon_log(f"recovery:reset learning state, backed up to {backup_path}")
    
    # Clear operator note to reset context
    if NEUROGOLF_OPERATOR_NOTE_PATH.exists():
        backup_note = NEUROGOLF_OPERATOR_NOTE_PATH.with_suffix(f".txt.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(NEUROGOLF_OPERATOR_NOTE_PATH), str(backup_note))
        _daemon_log("recovery:reset operator note")
    
    # Reset self-improvement state if enabled
    si_enabled_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement_enabled"
    if si_enabled_path.exists():
        si_enabled_path.unlink()
        _daemon_log("recovery:reset self-improvement to prevent over-optimization")


if __name__ == "__main__":
    raise SystemExit(main())
