from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import (
    KAGGLE_COMPETITION,
    MEMORY_PATH,
    NEUROGOLF_AUTONOMY_MAX_HISTORY,
    NEUROGOLF_CURRENT_MANIFEST,
    NEUROGOLF_IMPORTED_SOURCES_DIR,
    NEUROGOLF_OUTPUTS_DIR,
    NEUROGOLF_PROJECT_ROOT,
    NEUROGOLF_SEED_CONTROLS_PATH,
    NEUROGOLF_SYNC_STATE_PATH,
)
from core.env_settings import load_env_settings
from core.memory import load_memory, replace_source_patterns, save_memory
from core.kaggle_competition_data import (
    is_competition_data_ready,
    load_competition_metadata,
    list_competition_files,
    download_competition_data,
)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:  # pragma: no cover
    KaggleApi = None


PUBLIC_SEED_HINTS = {
    "latest_known_leaderboard_top": 9554.56,
    "kaggle_agent_9538_audit": 9538.00,
    "afr1ste_5653_open_solution": 5653.86,
    "magmacot_5550_logic_ensemble": 5550.00,
    "latest_artem_part4": 5383.96,
    "latest_rocker_5353": 5353.80,
    "latest_konbu_5344": 5344.29,
    "latest_afr1ste_5177": 5177.86,
    "latest_artem_part2": 5367.13,
}

BUILDABLE_SEED_HINTS_BASE = {
    "latest_artem_part4": 5383.96,
    "latest_artem_part2": 5367.13,
    "latest_rocker_5353": 5353.80,
    "latest_konbu_5344": 5344.29,
    "latest_afr1ste_5177": 5177.86,
}

METRIC_V3_RULES = {
    "effective_date": "2026-04-28",
    "constant_params_counted": True,
    "static_shapes_required": True,
    "dynamic_shapes_yield_zero": True,
    "memory_calculation": "sum_of_static_shape_bytes_excluding_io",
    "parameter_contributions": "includes_constant_values",
    "shape_inference_must_succeed": True,
    "symbolic_dimensions_invalid": True,
}

EXPLOIT_AUDIT = {
    "note": "April 28 2026 metric update closed dynamic-shape and constant-node exploits",
    "public_score": 9538.0,
    "omitted_tasks_due_to_grid_exclusion": 8,
    "dynamic_shape_flagged_tasks": 373,
    "runtime_output_shape_mismatch_tasks": 24,
    "zero_tensor_runtime_error_tasks": 20,
    "clean_static_shape_tasks": 19,
    "historical_constant_elements_total": 1368988,
    "top_hidden_constant_tasks": ["285", "118", "233", "158", "129", "357", "319", "076", "363", "392"],
    "metric_v3_compliant": False,
}
PAIR_SCORE_RE = re.compile(r"(?<!\d)(\d{4,5})[-_](\d{2})(?!\d)")
INT_SCORE_RE = re.compile(r"(?<!\d)(\d{4,5})(?!\d)")


METRIC_V3_RULES = {
    "effective_date": "2026-04-28",
    "constant_params_counted": True,
    "static_shapes_required": True,
    "dynamic_shapes_yield_zero": True,
    "memory_calculation": "sum_of_static_shape_bytes_excluding_io",
    "parameter_contributions": "includes_constant_values",
    "shape_inference_must_succeed": True,
    "symbolic_dimensions_invalid": True,
}

EXPLOIT_AUDIT = {
    "note": "April 28 2026 metric update closed dynamic-shape and constant-node exploits",
    "public_score": 9538.0,
    "omitted_tasks_due_to_grid_exclusion": 8,
    "dynamic_shape_flagged_tasks": 373,
    "runtime_output_shape_mismatch_tasks": 24,
    "zero_tensor_runtime_error_tasks": 20,
    "clean_static_shape_tasks": 19,
    "historical_constant_elements_total": 1368988,
    "top_hidden_constant_tasks": ["285", "118", "233", "158", "129", "357", "319", "076", "363", "392"],
    "metric_v3_compliant": False,
}


LESSONS = [
    "CRITICAL: APRIL 28 2026 METRIC V3 IS NOW ACTIVE. V2-compatible files and approaches are OBSOLETE and will score ZERO.",
    "CRITICAL: Dynamic shapes, symbolic dimensions, and non-static networks now YIELD ZERO POINTS. Build only V3-compliant static-shape networks.",
    "CRITICAL: Constant values now correctly count toward parameter contributions. The old hidden-constant exploit is CLOSED.",
    "CRITICAL: Memory footprint = sum of bytes for all static shapes (excluding input/output). No profiling quirks.",
    "CRITICAL: Submissions with V2-era invalid files are now BLOCKED by validation. The system will refuse to submit them.",
    "NOTE: Imported Kaggle notebooks are now sorted by RECENT submission time, not by score. Prioritize fresh submissions.",
    "REJECT all V2-era submissions that relied on dynamic shapes or constant undercounting. They are worthless now.",
    "BUILD new V3-compliant files for any task that doesn't have a valid static-shape network. Do not reuse invalid V2 files.",
    "User-imported Kaggle ensembles should be treated as reference material until each ONNX file passes local validation.",
    "Keep candidate zips, validation reports, and manifests between cycles so the system can build on earlier small gains instead of recreating them.",
    "Preserve useful validated task files, but do not resubmit whole seeds without local proof.",
    "Keep task000 when building seed-based submission packs.",
    "Processable-best swaps are safer than fill-invalid replacements.",
    "Do not trust full local scorer-wide blends over accepted public seed packs.",
    "Use Kaggle results as final truth; local score is only a weak hint.",
    "Stop repeating the same seed and mode combinations once the system has already marked them duplicate or already-submitted.",
    "Prefer fresh locally validated static graph builds over stale V2 exploit packs.",
    "Prioritize seeds and models with clean static shapes. Historically high-scoring V2 dynamic-shape submissions score ZERO.",
    "Validate all ONNX models for static shape inference before submission; invalid models waste submission slots.",
    "Progress is measured by more valid tasks, smaller graph costs, and only then higher Kaggle score.",
    # Metric V3 Optimal Strategy - Minimal Computational Graphs
    "METRIC V3 SCORING: points = 25.0 - log(MACs + memory + params). Minimize the sum for maximum score.",
    "Use big models as teachers/searchers, then compile or distill their rule into the smallest static ONNX graph.",
    "The 9538 exploit used 1.37M hidden constants + 373 dynamic-shape tasks. Only 19 clean tasks. Now INVALID.",
    "BUILD MINIMAL NETWORKS: Small + correct = high score. Large + complex = low score (or zero if invalid).",
    "Prefer simple numpy operations over deep neural networks. Geometric transforms > learned weights.",
    "Use smallest possible weight matrices. Every parameter counts against your score.",
    "Minimize intermediate tensor sizes. Memory footprint is sum of bytes for all internal tensors.",
    "Use fewest operations possible. Every MAC (multiply-accumulate) reduces your score.",
    "Prefer element-wise ops (Add, Mul) over reduction ops (Sum, Mean) - fewer MACs.",
    "Avoid Conv and MatMul when possible. Use Reshape, Transpose, indexing for geometric transforms.",
    "Use smallest data types: int8/int16 preferred over float32 when precision allows.",
    "Eliminate all dead code and unused variables. They add parameters/MACs without value.",
    "Fuse operations when possible: Conv+ReLU together vs separate nodes.",
    "Vectorize numpy operations. Loops add overhead and may increase graph size.",
    "Use lookup tables instead of learned mappings when the mapping is fixed.",
    "Hardcode transformations that work for all examples. Don't learn what's constant.",
    "Look for mathematical shortcuts: symmetry, invariants, constraints reduce computation.",
    "The best solution is often counter-intuitively simple. Avoid 'impressive' complexity.",
    "If two solutions both pass all test cases, the one with fewer ops scores higher.",
    "Build networks that 'just barely' solve the task. Any extra capacity hurts your score.",
    "Task-specific minimal solutions outperform generic architectures. Optimize per-task.",
    "ALL OLD V2 SEEDS DELETED: artem_part4, konbu_5344, rocker_5353, etc. are no longer available.",
    "FRESH BUILD MODE: When no valid seeds exist, build 400 new minimal networks from scratch.",
    "Build -> Test -> Validate -> Keep only what passes V3 -> Submit complete 400/400 pack.",
    "Only keep submissions where ALL 400 tasks pass V3 validation. Reject any with invalid tasks.",
    "CRITICAL: Check Kaggle submission error details when submissions fail. Error messages often indicate which specific tasks failed.",
    "When a submission fails, fetch error details from Kaggle to identify failed tasks and fix or replace them.",
    "Failed tasks from Kaggle errors should be added to the invalid_tasks list and replaced in subsequent builds.",
]


def _campaign_progress(submissions: list[dict]) -> dict:
    minimum_public_gain = 1.0
    completed = [item for item in submissions if item.get("public_score") is not None]
    if not completed:
        return {
            "submission_mode": "submit_any_public_gain",
            "minimum_public_gain_to_submit": minimum_public_gain,
            "best_completed_public_score": None,
            "latest_completed_public_score": None,
            "next_submit_score": None,
            "goal_reached": True,
        }

    best_completed = max(float(item["public_score"]) for item in completed)
    latest_completed = float(completed[0]["public_score"])
    return {
        "submission_mode": "submit_any_public_gain",
        "minimum_public_gain_to_submit": minimum_public_gain,
        "best_completed_public_score": best_completed,
        "latest_completed_public_score": latest_completed,
        "next_submit_score": best_completed + minimum_public_gain,
        "goal_reached": True,
    }


def _dynamic_lessons(submissions: list[dict]) -> list[str]:
    lessons: list[str] = []
    completed = [item for item in submissions if item.get("public_score") is not None]
    if not completed:
        return lessons
    best_score = max(float(item["public_score"]) for item in completed)
    latest = completed[0]
    latest_score = float(latest["public_score"])
    latest_desc = str(latest.get("description", "")).strip()
    if latest_score + 50.0 < best_score:
        lessons.append(
            f"The latest completed submission scored {latest_score:.2f}, well below the best known {best_score:.2f}; avoid repeating that exact experiment family."
        )
    if "processable-best" in latest_desc.lower() and "6285" in latest_desc:
        lessons.append(
            "The 6285 imported seed with processable-best swaps underperformed; prefer the direct seed or a different splice family over that exact swap pattern."
        )
    campaign = _campaign_progress(submissions)
    if campaign["best_completed_public_score"] is not None:
        lessons.append(
            f"Submission policy is immediate now: any valid pack that credibly beats {campaign['best_completed_public_score']:.2f} by +{campaign['minimum_public_gain_to_submit']:.2f} should go out."
        )
    return lessons


def _parse_claimed_public_score(text: str) -> float | None:
    scores: list[float] = []
    for match in PAIR_SCORE_RE.finditer(text):
        major = int(match.group(1))
        minor = int(match.group(2))
        if 3000 <= major <= 9999:
            scores.append(float(f"{major}.{minor:02d}"))
    for match in INT_SCORE_RE.finditer(text):
        major = int(match.group(1))
        if 3000 <= major <= 9999:
            scores.append(float(major))
    if not scores:
        return None
    return max(scores)


def _sanitize_seed_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _count_task_files_in_dir(dir_path: Path) -> int:
    return sum(1 for item in dir_path.glob("task*.onnx"))


def _count_task_files_in_zip(zip_path: Path) -> int:
    try:
        import zipfile

        with zipfile.ZipFile(zip_path) as archive:
            return sum(1 for name in archive.namelist() if re.fullmatch(r"task\d{3}\.onnx", Path(name).name))
    except Exception:
        return 0


def _detect_imported_seed_artifact(root: Path) -> dict[str, object] | None:
    best_zip: tuple[int, int, Path] | None = None
    for zip_path in sorted(root.rglob("*.zip")):
        task_count = _count_task_files_in_zip(zip_path)
        if task_count <= 0:
            continue
        priority = 1 if "submission" in zip_path.name.lower() else 0
        if best_zip is None or (task_count, priority) > (best_zip[0], best_zip[1]):
            best_zip = (task_count, priority, zip_path)

    best_dir: tuple[int, int, Path] | None = None
    for dir_path in [root] + sorted(path for path in root.rglob("*") if path.is_dir()):
        task_count = _count_task_files_in_dir(dir_path)
        if task_count <= 0:
            continue
        priority = 1 if "submission" in dir_path.name.lower() or "onnx" in dir_path.name.lower() else 0
        if best_dir is None or (task_count, priority) > (best_dir[0], best_dir[1]):
            best_dir = (task_count, priority, dir_path)

    if best_zip is None and best_dir is None:
        return None
    if best_zip is None:
        return {"artifact_path": str(best_dir[2]), "seed_kind": "dir", "task_file_count": best_dir[0]}
    if best_dir is None:
        return {"artifact_path": str(best_zip[2]), "seed_kind": "zip", "task_file_count": best_zip[0]}
    if best_zip[0] >= best_dir[0]:
        return {"artifact_path": str(best_zip[2]), "seed_kind": "zip", "task_file_count": best_zip[0]}
    return {"artifact_path": str(best_dir[2]), "seed_kind": "dir", "task_file_count": best_dir[0]}


def _json_or_none(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_blend_mode(payload: dict) -> str | None:
    mode = str(payload.get("mode", "")).strip()
    if mode:
        return mode
    score_scope = str(payload.get("score_scope", "")).strip().lower()
    if score_scope == "known_examples_only":
        return "strict_known"
    source_counts = payload.get("source_counts")
    if isinstance(source_counts, dict) and any(str(name).startswith("sanitized_") for name in source_counts):
        return "skip_known_dynamic_sanitized"
    if score_scope == "profile_only":
        return "skip_known_dynamic"
    return None


def _normalized_output_manifest(path: Path, payload: dict) -> dict:
    row = dict(payload)
    build_strategy = str(row.get("build_strategy", "")).strip()
    if not build_strategy:
        if path.name == "submission_manifest.json":
            build_strategy = "scorer_blend"
        elif path.name.endswith("_validated_manifest.json"):
            build_strategy = "validated_repair"
        elif row.get("seed_label") and row.get("artifact_path"):
            build_strategy = "direct_seed_pack"
        elif row.get("seed_label") and row.get("mode"):
            build_strategy = "seed_preserving"
        elif row.get("source_zip"):
            build_strategy = "cached_candidate_submit"
    row["build_strategy"] = build_strategy or None

    if build_strategy == "scorer_blend" and not row.get("mode"):
        inferred_mode = _infer_blend_mode(row)
        if inferred_mode:
            row["mode"] = inferred_mode

    if build_strategy == "validated_repair":
        row["validation_status"] = "valid"
        source_zip = Path(str(row.get("source_zip", "")).strip()) if str(row.get("source_zip", "")).strip() else None
        if source_zip is not None:
            candidate_manifests = [NEUROGOLF_OUTPUTS_DIR / f"{source_zip.stem}_manifest.json", NEUROGOLF_OUTPUTS_DIR / "submission_manifest.json"]
            best_source_row: dict | None = None
            best_source_manifest: Path | None = None
            for source_manifest in candidate_manifests:
                try:
                    if source_manifest.resolve() == path.resolve():
                        continue
                except Exception:
                    pass
                source_payload = _json_or_none(source_manifest)
                if not isinstance(source_payload, dict):
                    continue
                source_row = _normalized_output_manifest(source_manifest, source_payload)
                source_zip_path = Path(str(source_row.get("zip_path", "")).strip()) if str(source_row.get("zip_path", "")).strip() else None
                if source_zip_path is None or source_zip_path != source_zip:
                    continue
                if best_source_row is None:
                    best_source_row = source_row
                    best_source_manifest = source_manifest
                if source_row.get("known_score") not in (None, "") or source_row.get("estimated_live_score") not in (None, ""):
                    best_source_row = source_row
                    best_source_manifest = source_manifest
                    break
            if best_source_row is not None and best_source_manifest is not None:
                for key in (
                    "seed_label",
                    "mode",
                    "files",
                    "replaced_count",
                    "delta_sum_local",
                    "known_score",
                    "estimated_live_score",
                    "score_scope",
                ):
                    if row.get(key) in (None, "") and best_source_row.get(key) not in (None, ""):
                        row[key] = best_source_row.get(key)
                row["source_manifest_path"] = str(best_source_manifest)
    else:
        row["validation_status"] = str(row.get("validation_status", "")).strip() or "unknown"

    return row


def _recent_output_manifests(limit: int = 12) -> list[dict]:
    manifests: list[dict] = []
    if not NEUROGOLF_OUTPUTS_DIR.exists():
        return manifests
    for path in sorted(
        NEUROGOLF_OUTPUTS_DIR.glob("*manifest*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:limit]:
        payload = _json_or_none(path)
        if not isinstance(payload, dict):
            continue
        payload = _normalized_output_manifest(path, payload)
        manifests.append(
            {
                "path": str(path),
                "name": path.name,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "build_strategy": payload.get("build_strategy"),
                "validation_status": payload.get("validation_status"),
                "seed_label": payload.get("seed_label"),
                "mode": payload.get("mode"),
                "files": payload.get("files"),
                "replaced_count": payload.get("replaced_count"),
                "delta_sum_local": payload.get("delta_sum_local"),
                "zip_path": payload.get("zip_path"),
                "zip_size_bytes": payload.get("zip_size_bytes"),
                "score_scope": payload.get("score_scope"),
                "known_score": payload.get("known_score"),
                "estimated_live_score": payload.get("estimated_live_score"),
                "source_zip": payload.get("source_zip"),
            }
        )
    return manifests


def _recent_output_files(limit: int = 25) -> list[dict]:
    files: list[dict] = []
    if not NEUROGOLF_OUTPUTS_DIR.exists():
        return files
    for path in sorted(
        (item for item in NEUROGOLF_OUTPUTS_DIR.iterdir() if item.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:limit]:
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return files


def _imported_sources(limit: int = 20) -> list[dict]:
    rows: list[dict] = []
    if not NEUROGOLF_IMPORTED_SOURCES_DIR.exists():
        return rows
    controls = _seed_controls()
    for path in sorted(NEUROGOLF_IMPORTED_SOURCES_DIR.glob("*/*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        claimed_public_score = _parse_claimed_public_score(path.name)
        seed_label = f"imported_{_sanitize_seed_label(path.name)}"
        artifact = _detect_imported_seed_artifact(path)
        control = controls.get(seed_label, {})
        manual_score = control.get("manual_public_score")
        score_source = "inferred" if claimed_public_score is not None else "none"
        if manual_score is not None:
            try:
                parsed_manual_score = float(manual_score)
                if math.isfinite(parsed_manual_score):
                    claimed_public_score = parsed_manual_score
                    score_source = "manual"
            except (TypeError, ValueError):
                manual_score = None
        disabled = bool(control.get("disabled", False))
        rows.append(
            {
                "kind": path.parent.name,
                "name": path.name,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "claimed_public_score": claimed_public_score,
                "score_source": score_source,
                "seed_label": seed_label,
                "disabled": disabled,
                "control_note": str(control.get("note", "")),
                "usable_as_seed": bool(artifact) and not disabled,
                "seed_kind": artifact.get("seed_kind") if artifact else None,
                "artifact_path": artifact.get("artifact_path") if artifact else None,
                "task_file_count": artifact.get("task_file_count") if artifact else 0,
            }
        )
    # Sort by updated_at (newest first), then by score (highest first)
    # This prioritizes recent submissions over historical high scores
    rows.sort(key=lambda item: (item.get("updated_at", ""), (item.get("claimed_public_score") or 0.0)), reverse=True)
    return rows[:limit]


def _seed_controls() -> dict:
    if not NEUROGOLF_SEED_CONTROLS_PATH.exists():
        return {}
    try:
        payload = json.loads(NEUROGOLF_SEED_CONTROLS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    controls = payload.get("seeds", payload) if isinstance(payload, dict) else {}
    return controls if isinstance(controls, dict) else {}


def _recent_cycle_patterns(limit: int = 12) -> dict:
    reports_dir = NEUROGOLF_SYNC_STATE_PATH.parent.parent / "outputs" / "neurogolf"
    plan_counts: dict[str, int] = {}
    submit_reasons: dict[str, int] = {}
    seen = 0
    if not reports_dir.exists():
        return {"plan_counts": {}, "submit_reasons": {}}
    for path in sorted(reports_dir.glob("cycle_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        plan = payload.get("plan") or {}
        action = str(plan.get("action", "")).strip() or "seed_preserving_build"
        target = str(plan.get("target", "")).strip() or str(plan.get("seed_label", "")).strip() or "-"
        variant = str(plan.get("mode", "")).strip() or str(plan.get("variant_hint", "")).strip() or "-"
        if action == "seed_preserving_build":
            value = variant.lower().replace("-", "_")
            if value in {"repair", "repair_heavy", "fill", "fill_invalid", "invalid", "recovery"}:
                variant = "fill_invalid_priority"
            elif value in {"aggressive", "hybrid", "mixed", "balanced", "open"}:
                variant = "hybrid_priority"
            elif value not in {"processable_best", "fill_invalid_priority", "hybrid_priority"}:
                variant = "processable_best"
        elif action == "scorer_blend_build":
            value = variant.lower()
            if value in {"strict", "safe", "known"}:
                variant = "strict_known"
            elif value in {"sanitized", "sanitize", "patched"}:
                variant = "skip_known_dynamic_sanitized"
            elif value not in {"skip_known_dynamic", "strict_known", "skip_known_dynamic_sanitized"}:
                variant = "strict_known"
        plan_key = f"{action}|{target}|{variant}"
        plan_counts[plan_key] = plan_counts.get(plan_key, 0) + 1
        submit_result = payload.get("submit_result") or {}
        reason = str(submit_result.get("reason", "")).strip()
        if reason:
            submit_reasons[reason] = submit_reasons.get(reason, 0) + 1
        seen += 1
        if seen >= limit:
            break
    return {"plan_counts": plan_counts, "submit_reasons": submit_reasons}


def _submission_history(limit: int = NEUROGOLF_AUTONOMY_MAX_HISTORY) -> list[dict]:
    api_rows: list[dict] = []
    env_settings = load_env_settings()
    configured_dir = Path(str(env_settings.get("KAGGLE_CONFIG_DIR") or "")).expanduser()
    if configured_dir and not configured_dir.is_absolute():
        configured_dir = (Path(__file__).resolve().parents[1] / configured_dir).resolve()
    kaggle_json = configured_dir / "kaggle.json" if configured_dir else Path()
    legacy_json = NEUROGOLF_PROJECT_ROOT / "kaggle.json"
    if kaggle_json.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(configured_dir)
    elif legacy_json.exists() and "KAGGLE_CONFIG_DIR" not in os.environ:
        os.environ["KAGGLE_CONFIG_DIR"] = str(NEUROGOLF_PROJECT_ROOT)

    if KaggleApi is not None:
        try:
            api = KaggleApi()
            api.authenticate()
            submissions = list(api.competition_submissions(KAGGLE_COMPETITION))[:limit]
            for submission in submissions:
                api_rows.append(
                    {
                        "description": getattr(submission, "description", None),
                        "status": str(getattr(submission, "status", None)),
                        "public_score": getattr(submission, "public_score", None),
                        "private_score": getattr(submission, "private_score", None),
                        "file_name": getattr(submission, "file_name", None),
                        "date": getattr(getattr(submission, "date", None), "isoformat", lambda: None)(),
                        "team_name": getattr(submission, "team_name", None),
                        "submitted_by": getattr(submission, "submitted_by", None),
                    }
                )
        except Exception:
            api_rows = []

    kaggle_exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle.exe"))
    try:
        completed = subprocess.run(
            [kaggle_exe, "competitions", "submissions", KAGGLE_COMPETITION, "-v", "--page-size", str(limit)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    lines = [line for line in completed.stdout.splitlines() if line.strip() and not line.startswith("Warning:")]
    if not lines:
        return []
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return api_rows
    rows: list[dict] = []
    for parts in reader:
        if len(parts) < 6:
            continue
        rows.append(
            {
                "file_name": parts[0],
                "date": parts[1],
                "description": parts[2],
                "status": parts[3],
                "public_score": float(parts[4]) if parts[4] else None,
                "private_score": float(parts[5]) if parts[5] else None,
            }
        )
    if rows and any(item.get("public_score") is not None for item in rows):
        for index, row in enumerate(rows):
            if index >= len(api_rows):
                continue
            row["team_name"] = api_rows[index].get("team_name")
            row["submitted_by"] = api_rows[index].get("submitted_by")
    
    # Fetch error details for failed/error submissions
    for row in rows:
        status = str(row.get("status", "")).lower()
        if "error" in status or "fail" in status:
            # Try to get more details about which tasks failed
            error_details = _fetch_submission_error_details(row.get("file_name"))
            row["error_details"] = error_details
            # Add failed tasks to a separate field for easier access
            if error_details.get("failed_tasks"):
                row["failed_tasks"] = error_details["failed_tasks"]
    
    return rows if rows else api_rows


def _fetch_submission_error_details(submission_id: str | None = None) -> dict:
    """Fetch error details from Kaggle for failed submissions.
    
    Kaggle often includes which specific tasks failed in error messages.
    This helps identify which tasks need to be fixed or replaced.
    """
    error_details = {"error_message": None, "failed_tasks": [], "raw_output": None}
    
    kaggle_exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle.exe"))
    
    # Try to get detailed submission info including error messages
    try:
        # First try: Get submissions with verbose output
        completed = subprocess.run(
            [kaggle_exe, "competitions", "submissions", KAGGLE_COMPETITION, "-v"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        error_details["raw_output"] = completed.stdout + "\n" + completed.stderr
        
        # Parse output to find error information
        output = completed.stdout + completed.stderr
        
        # Look for common error patterns that indicate which tasks failed
        # Pattern 1: "taskXXX failed" or "taskXXX error"
        task_failures = re.findall(r'task(\d{3}).*?(?:failed|error|invalid|could not)', output, re.IGNORECASE)
        if task_failures:
            error_details["failed_tasks"] = [f"task{t}" for t in task_failures]
        
        # Pattern 2: Look for error messages after "Error:" or "error:"
        error_lines = [line for line in output.splitlines() if "error" in line.lower()]
        if error_lines:
            error_details["error_message"] = error_lines[0][:500]  # First error line, truncated
            
    except Exception as exc:
        error_details["error_message"] = f"Could not fetch error details: {type(exc).__name__}"
    
    # Also try API method if available
    if KaggleApi is not None and submission_id:
        try:
            api = KaggleApi()
            api.authenticate()
            # Note: Kaggle API doesn't directly expose error details,
            # but we can try to get more info from the submission object
            submissions = list(api.competition_submissions(KAGGLE_COMPETITION))
            for sub in submissions:
                if getattr(sub, "ref", None) == submission_id or getattr(sub, "file_name", None) == submission_id:
                    status = str(getattr(sub, "status", "")).lower()
                    if "error" in status or "fail" in status:
                        error_details["error_message"] = f"Submission status: {status}"
                    break
        except Exception:
            pass  # API method failed, CLI output already captured
    
    return error_details


def summarize_neurogolf_workspace(limit_history: int = NEUROGOLF_AUTONOMY_MAX_HISTORY) -> dict:
    current_manifest = _json_or_none(NEUROGOLF_CURRENT_MANIFEST)
    submissions = _submission_history(limit=limit_history)
    imported_sources = _imported_sources()
    public_seed_hints = dict(PUBLIC_SEED_HINTS)
    buildable_seed_hints = dict(BUILDABLE_SEED_HINTS_BASE)
    for row in imported_sources:
        seed_label = str(row.get("seed_label", "")).strip()
        claimed_public_score = row.get("claimed_public_score")
        if seed_label and claimed_public_score is not None:
            public_seed_hints[seed_label] = float(claimed_public_score)
            if row.get("usable_as_seed"):
                buildable_seed_hints[seed_label] = float(claimed_public_score)
    available_seed_labels = [
        label
        for label, _score in sorted(buildable_seed_hints.items(), key=lambda item: item[1], reverse=True)
    ]
    completed_scores = [float(item["public_score"]) for item in submissions if item.get("public_score") is not None]
    dynamic_lessons = _dynamic_lessons(submissions)
    campaign_progress = _campaign_progress(submissions)
    pending_count = sum(
        1
        for item in submissions
        if any(token in str(item.get("status", "")).strip().lower() for token in ("pending", "running", "processing", "queued"))
    )
    summary = {
        "project_root": str(NEUROGOLF_PROJECT_ROOT),
        "outputs_dir": str(NEUROGOLF_OUTPUTS_DIR),
        "current_manifest_path": str(NEUROGOLF_CURRENT_MANIFEST),
        "current_manifest": current_manifest if isinstance(current_manifest, dict) else {},
        "recent_output_manifests": _recent_output_manifests(),
        "recent_output_files": _recent_output_files(),
        "imported_sources": imported_sources,
        "recent_submissions": submissions,
        "best_completed_public_score": max(completed_scores) if completed_scores else None,
        "pending_submission_count": pending_count,
        "current_team_name": next((str(item.get("team_name", "")).strip() for item in submissions if str(item.get("team_name", "")).strip()), ""),
        "campaign_progress": campaign_progress,
        "submission_policy": campaign_progress,
        "public_seed_hints": public_seed_hints,
        "buildable_seed_hints": buildable_seed_hints,
        "available_seed_labels": available_seed_labels,
        "highest_claimed_imported_public_score": max(
            (float(row["claimed_public_score"]) for row in imported_sources if row.get("claimed_public_score") is not None),
            default=None,
        ),
        "metric_v3_rules": METRIC_V3_RULES,
        "exploit_audit": EXPLOIT_AUDIT,
        "recent_cycle_patterns": _recent_cycle_patterns(),
        "lessons": LESSONS + dynamic_lessons,
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "competition_data": {
            "ready": is_competition_data_ready(),
            "metadata": load_competition_metadata(),
            "files": list_competition_files()[:20],  # Limit to first 20 files
            "competition": "neurogolf-2026",
            "metric_version": "v3",
        },
    }
    return summary


def sync_neurogolf_state(limit_history: int = NEUROGOLF_AUTONOMY_MAX_HISTORY) -> dict:
    summary = summarize_neurogolf_workspace(limit_history=limit_history)
    NEUROGOLF_SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUROGOLF_SYNC_STATE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    patterns = [
        {
            "task": "neurogolf_competition_state",
            "reasoning": lesson,
            "keywords": ["neurogolf", "seed", "kaggle", "submission"],
            "success": True,
            "source": "neurogolf_sync",
            "stored_at": summary["synced_at"],
        }
        for lesson in LESSONS
    ]
    replace_source_patterns("neurogolf_sync", patterns)
    return summary
