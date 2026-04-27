from __future__ import annotations

import csv
import json
import os
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
    NEUROGOLF_SYNC_STATE_PATH,
)
from core.memory import load_memory, replace_source_patterns, save_memory

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:  # pragma: no cover
    KaggleApi = None


PUBLIC_SEED_HINTS = {
    "latest_artem_part4": 5383.96,
    "latest_rocker_5353": 5353.80,
    "latest_konbu_5344": 5344.29,
    "latest_afr1ste_5177": 5177.86,
}


LESSONS = [
    "Preserve the strongest accepted seed and swap only on safer subsets.",
    "Keep task000 when building seed-based submission packs.",
    "Processable-best swaps are safer than fill-invalid replacements.",
    "Do not trust full local scorer-wide blends over accepted public seed packs.",
    "Use Kaggle leaderboard results as final truth; local score is only a weak hint.",
]


def _json_or_none(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
        manifests.append(
            {
                "path": str(path),
                "name": path.name,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "seed_label": payload.get("seed_label"),
                "mode": payload.get("mode"),
                "files": payload.get("files"),
                "replaced_count": payload.get("replaced_count"),
                "delta_sum_local": payload.get("delta_sum_local"),
                "zip_path": payload.get("zip_path"),
                "zip_size_bytes": payload.get("zip_size_bytes"),
                "known_score": payload.get("known_score"),
                "estimated_live_score": payload.get("estimated_live_score"),
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
    for path in sorted(NEUROGOLF_IMPORTED_SOURCES_DIR.glob("*/*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        rows.append(
            {
                "kind": path.parent.name,
                "name": path.name,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _submission_history(limit: int = NEUROGOLF_AUTONOMY_MAX_HISTORY) -> list[dict]:
    api_rows: list[dict] = []
    kaggle_json = NEUROGOLF_PROJECT_ROOT / "kaggle.json"
    if kaggle_json.exists() and "KAGGLE_CONFIG_DIR" not in os.environ:
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
        return rows
    return api_rows


def summarize_neurogolf_workspace(limit_history: int = NEUROGOLF_AUTONOMY_MAX_HISTORY) -> dict:
    current_manifest = _json_or_none(NEUROGOLF_CURRENT_MANIFEST)
    submissions = _submission_history(limit=limit_history)
    completed_scores = [float(item["public_score"]) for item in submissions if item.get("public_score") is not None]
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
        "imported_sources": _imported_sources(),
        "recent_submissions": submissions,
        "best_completed_public_score": max(completed_scores) if completed_scores else None,
        "pending_submission_count": pending_count,
        "current_team_name": next((str(item.get("team_name", "")).strip() for item in submissions if str(item.get("team_name", "")).strip()), ""),
        "public_seed_hints": PUBLIC_SEED_HINTS,
        "lessons": LESSONS,
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
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
