from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from core.config import (
    MODEL_OPTIONS,
    MODEL_REGISTRY,
    NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD,
    NEUROGOLF_AUTONOMY_LOG_PATH,
    NEUROGOLF_AUTONOMY_LOOP_SECONDS,
    NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS,
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
        "imported_sources": summary.get("imported_sources", [])[:10],
        "public_seed_hints": summary.get("public_seed_hints", {}),
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


def _fallback_candidates() -> list[tuple[str, str]]:
    return [
        ("latest_artem_part4", "processable_best"),
        ("latest_konbu_5344", "processable_best"),
        ("latest_rocker_5353", "processable_best"),
        ("latest_afr1ste_5177", "processable_best"),
        ("latest_artem_part4", "hybrid_priority"),
        ("latest_konbu_5344", "hybrid_priority"),
        ("latest_rocker_5353", "hybrid_priority"),
        ("latest_afr1ste_5177", "hybrid_priority"),
        ("latest_artem_part4", "fill_invalid_priority"),
        ("latest_konbu_5344", "fill_invalid_priority"),
    ]


def _safe_model_response(role: str, prompt: str, system: str, *, fallback_role: str = "orchestrator") -> str:
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

State:
{state}

Focus on:
- preserving strong public seeds
- avoiding bad local-scorer traps
- finding efficient, leaderboard-safe task swaps
- identifying loopholes or efficiency families worth testing

Return a compact numbered list.
""",
        system="You are the primary NeuroGolf strategist. Think like a careful competition engineer.",
        fallback_role="orchestrator",
    )
    secondary = _safe_model_response(
        "reasoning_secondary",
        f"""Propose alternative NeuroGolf experiments that differ from the obvious seed-preserving plan.

State:
{state}

Look for:
- overlooked seed choices
- safe vs risky experiment splits
- potential submission/validator quirks
- unusual but testable leaderboard ideas

Return a compact numbered list.
""",
        system="You are the alternate NeuroGolf strategist. Generate distinct ideas, not paraphrases.",
        fallback_role="orchestrator",
    )
    tertiary = _safe_model_response(
        "reasoning_tertiary",
        f"""Provide a third NeuroGolf strategy pass that focuses on resource fusion.

State:
{state}

Your job:
- combine Kaggle imports, local manifests, prior lessons, and public seed behavior
- look for validator-safe loopholes or underused imported assets
- suggest experiments that the first two strategists might miss

Return a compact numbered list.
""",
        system="You are the third NeuroGolf strategist. Fuse external Kaggle signals with local evidence and search for overlooked leverage.",
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
- candidate seed/mode combos
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


def _normalize_plan(plan: dict, reports: list[dict]) -> dict:
    recent = _recent_seed_modes(reports)
    seed = str(plan.get("seed_label", "latest_artem_part4")).strip() or "latest_artem_part4"
    mode = str(plan.get("mode", "processable_best")).strip() or "processable_best"
    if (seed, mode) not in recent:
        return plan
    for candidate_seed, candidate_mode in _fallback_candidates():
        if (candidate_seed, candidate_mode) not in recent:
            plan = dict(plan)
            plan["seed_label"] = candidate_seed
            plan["mode"] = candidate_mode
            plan["rationale"] = str(plan.get("rationale", "")).strip() + " | switched to a novel seed/mode combo"
            return plan
    return plan


def _experiment_fingerprint(plan: dict, build_manifest: dict) -> str:
    payload = {
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
        "base_zip": manifest.get("seed_zip") or manifest.get("base_zip"),
        "mode": manifest.get("mode"),
        "preserved_task000": manifest.get("preserved_task000"),
        "replaced_count": manifest.get("replaced_count"),
        "tasks": sorted(set(tasks)),
        "zip_size_bytes": manifest.get("zip_size_bytes"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


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


def plan_next_action(summary: dict) -> dict:
    operator_note = _load_operator_note()
    team = _team_strategy_briefs(summary)
    _daemon_log("planner:start orchestrator merge")
    _write_status("planning", "Merging strategist outputs", progress=0.6)
    raw = ask(
        MODEL_REGISTRY["orchestrator"],
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
- action: "seed_preserving_build" or "sync_only"
- seed_label: one of ["latest_artem_part4", "latest_rocker_5353", "latest_konbu_5344", "latest_afr1ste_5177"]
- mode: one of ["processable_best", "fill_invalid_priority", "hybrid_priority"]
- should_submit: true/false
- rationale: short string
- submission_message: short string

Default behavior:
- prefer latest_artem_part4 + processable_best
- keep task000
- prefer safer processable-best swaps over aggressive fills
- use imported Kaggle assets when they materially improve the experiment quality
""",
        system=NEUROGOLF_AUTONOMY_SYSTEM_PROMPT,
        timeout=240,
        options=MODEL_OPTIONS["orchestrator"],
    )
    _daemon_log("planner:done orchestrator merge")
    _write_status("planning", "Planner selected next experiment", progress=0.72)
    parsed = _extract_json_object(raw)
    if not parsed:
        return {
            "action": "seed_preserving_build",
            "seed_label": "latest_artem_part4",
            "mode": "processable_best",
            "should_submit": False,
            "rationale": "Fallback plan",
            "submission_message": "autonomy fallback seed-preserving build",
            "raw_plan": raw,
            "team_strategy": team,
        }
    parsed["raw_plan"] = raw
    parsed["team_strategy"] = team
    return parsed


def run_seed_preserving_build(zip_name: str, seed_label: str, mode: str) -> dict:
    cached = _find_cached_seed_manifest(seed_label, mode)
    output_zip = NEUROGOLF_PROJECT_ROOT / "outputs" / zip_name
    manifest_path = NEUROGOLF_PROJECT_ROOT / "outputs" / f"{Path(zip_name).stem}_manifest.json"
    if cached is not None:
        cached_zip = Path(str(cached["zip_path"]))
        if cached_zip.resolve() != output_zip.resolve():
            shutil.copyfile(cached_zip, output_zip)
        manifest = dict(cached)
        manifest["zip_path"] = str(output_zip)
        manifest["zip_size_bytes"] = output_zip.stat().st_size
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _daemon_log(
            "build:cache-hit "
            f"seed={seed_label} mode={mode} source={cached.get('_manifest_path')} zip={output_zip.name}"
        )
        _write_status("building", f"Using cached build for {seed_label} / {mode}", progress=0.84, extra={"seed_label": seed_label, "mode": mode})
        return manifest

    _daemon_log(f"build:start seed={seed_label} mode={mode} zip={zip_name}")
    _write_status("building", f"Building {seed_label} / {mode}", progress=0.8, extra={"seed_label": seed_label, "mode": mode})
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
    subprocess.run(command, cwd=NEUROGOLF_PROJECT_ROOT, check=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    _daemon_log(
        "build:done "
        f"seed={seed_label} mode={mode} replaced={manifest.get('replaced_count')} "
        f"delta={manifest.get('delta_sum_local')} files={manifest.get('files')}"
    )
    _write_status("building", f"Finished build for {seed_label} / {mode}", progress=0.88, extra={"seed_label": seed_label, "mode": mode})
    return manifest


def maybe_submit(zip_name: str, message: str, manifest: dict, min_local_delta: float) -> dict:
    if float(manifest.get("delta_sum_local", 0.0)) < min_local_delta:
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


def run_cycle(args: argparse.Namespace) -> dict:
    summary = sync_neurogolf_state(limit_history=args.history)
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
        },
    )
    if args.sync_only:
        _write_status("idle", "Sync complete", progress=1.0)
        return {"synced": True, "best_completed_public_score": summary.get("best_completed_public_score")}

    reports = _recent_reports()
    plan = plan_next_action(summary)
    plan = _normalize_plan(plan, reports)
    action = str(plan.get("action", "seed_preserving_build")).strip() or "seed_preserving_build"
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "plan": plan,
        "best_completed_public_score": summary.get("best_completed_public_score"),
        "pending_submission_count": summary.get("pending_submission_count"),
    }

    if action == "sync_only":
        report["result"] = {"synced": True}
        report["submit_result"] = {"submitted": False, "reason": "sync-only cycle"}
        _daemon_log("cycle:planner chose sync-only")
        _write_status("idle", "Planner chose sync-only", progress=1.0)
    else:
        seed_label = str(plan.get("seed_label", "latest_artem_part4")).strip() or "latest_artem_part4"
        mode = str(plan.get("mode", "processable_best")).strip() or "processable_best"
        submit_message = str(plan.get("submission_message", "")).strip() or f"autonomy {seed_label} {mode}"
        _daemon_log(f"cycle:plan seed={seed_label} mode={mode} submit={bool(plan.get('should_submit', False))}")
        build_manifest = run_seed_preserving_build(args.zip_name, seed_label, mode)
        report["build_manifest"] = build_manifest
        report["experiment_fingerprint"] = _experiment_fingerprint(plan, build_manifest)
        report["candidate_signature"] = _candidate_signature(build_manifest)
        already_submitted = report["experiment_fingerprint"] in _submitted_fingerprints(reports)
        current_manifest = summary.get("current_manifest") if isinstance(summary.get("current_manifest"), dict) else {}
        matches_current = bool(current_manifest) and _candidate_signature(build_manifest) == _candidate_signature(current_manifest)
        recent_descriptions = {
            str(item.get("description", "")).strip()
            for item in summary.get("recent_submissions", [])
            if str(item.get("description", "")).strip()
        }
        if args.allow_submit and bool(plan.get("should_submit", False)):
            if int(summary.get("pending_submission_count", 0)) >= args.max_pending_submissions:
                report["submit_result"] = {"submitted": False, "reason": "pending submission cap reached"}
                _write_status("idle", "Waiting: pending submission cap reached", progress=1.0)
            elif matches_current:
                report["submit_result"] = {"submitted": False, "reason": "candidate matches current active pack"}
                _write_status("idle", "Skipped: candidate matches active pack", progress=1.0)
            elif submit_message in recent_descriptions:
                report["submit_result"] = {"submitted": False, "reason": "submission message already exists in recent history"}
                _write_status("idle", "Skipped: recent submission already used this message", progress=1.0)
            elif already_submitted:
                report["submit_result"] = {"submitted": False, "reason": "experiment already submitted"}
                _write_status("idle", "Skipped: experiment already submitted", progress=1.0)
            else:
                report["submit_result"] = maybe_submit(
                    args.zip_name,
                    submit_message,
                    build_manifest,
                    args.min_local_delta,
                )
        elif args.allow_submit:
            report["submit_result"] = {"submitted": False, "reason": "planner held submission"}
            _write_status("idle", "Planner held submission", progress=1.0)
        else:
            report["submit_result"] = {"submitted": False, "reason": "submission disabled"}
            _write_status("idle", "Submission disabled for this run", progress=1.0)
        _daemon_log(
            f"cycle:submit submitted={report['submit_result'].get('submitted')} "
            f"reason={report['submit_result'].get('reason', '')}"
        )

    NEUROGOLF_AUTONOMY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _daemon_log(f"cycle:report {report_path}")
    _write_status("idle", "Cycle complete", progress=1.0, extra={"latest_report": str(report_path)})
    return {"report_path": str(report_path), "action": action, "report": report}


def main() -> int:
    args = build_parser().parse_args()
    if args.loop:
        cycle = 0
        while True:
            cycle += 1
            try:
                result = run_cycle(args)
                print(json.dumps(result, indent=2))
            except Exception as exc:
                NEUROGOLF_AUTONOMY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                error_report = {
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                error_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / f"cycle_error_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                error_path.write_text(json.dumps(error_report, indent=2), encoding="utf-8")
                _daemon_log(f"cycle:error {error_report['error']} path={error_path}")
                _write_status("error", error_report["error"], progress=1.0, extra={"error_report": str(error_path)})
                print(json.dumps({"error_report": str(error_path), "error": error_report["error"]}, indent=2))
            if args.max_cycles > 0 and cycle >= args.max_cycles:
                break
            time.sleep(max(30, args.sleep_seconds))
        return 0

    result = run_cycle(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
