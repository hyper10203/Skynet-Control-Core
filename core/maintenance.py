"""Non-destructive maintenance helpers for AxiomGraph Operations Core."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, SKYNET_ARCHIVE_DIR, NEUROGOLF_AUTONOMY_REPORTS_DIR


KEEP_RECENT_CYCLE_REPORTS = 60


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _item_record(path: Path, kind: str, reason: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(PROJECT_ROOT)) if _inside_project(path) else str(path),
        "kind": kind,
        "reason": reason,
        "size_bytes": _path_size(path),
    }


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    return total


def _old_cycle_reports() -> list[Path]:
    if not NEUROGOLF_AUTONOMY_REPORTS_DIR.exists():
        return []
    reports = sorted(
        NEUROGOLF_AUTONOMY_REPORTS_DIR.glob("cycle*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return reports[KEEP_RECENT_CYCLE_REPORTS:]


def find_clutter() -> list[dict[str, Any]]:
    """Return generated clutter that can be archived without losing source state."""
    items: list[dict[str, Any]] = []

    for path in PROJECT_ROOT.glob("__pycache__"):
        items.append(_item_record(path, "python_cache", "root Python bytecode cache"))
    for folder in ("agents", "core", "utils"):
        base = PROJECT_ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("__pycache__"):
            items.append(_item_record(path, "python_cache", f"{folder} Python bytecode cache"))
        for path in base.rglob("*.backup.*"):
            items.append(_item_record(path, "source_backup", "timestamped source backup"))

    for path in _old_cycle_reports():
        items.append(_item_record(path, "old_cycle_report", f"older than latest {KEEP_RECENT_CYCLE_REPORTS} cycle reports"))

    for path in NEUROGOLF_AUTONOMY_REPORTS_DIR.glob("*.backup_*"):
        items.append(_item_record(path, "state_backup", "timestamped state backup"))

    return items


def archive_clutter(*, dry_run: bool = False) -> dict[str, Any]:
    """Archive generated clutter under the configured maintenance archive folder.

    Files are moved, not deleted. Source files, memory controls, current status,
    current learning state, and recent cycle reports are left in place.
    """
    items = find_clutter()
    stamp = _utc_stamp()
    archive_root = SKYNET_ARCHIVE_DIR / stamp
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if not dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)

    for item in items:
        src = Path(str(item["path"]))
        if not src.exists():
            skipped.append({**item, "skip_reason": "missing"})
            continue
        if not _inside_project(src):
            skipped.append({**item, "skip_reason": "outside project root"})
            continue
        relative = src.relative_to(PROJECT_ROOT)
        dest = archive_root / relative
        record = {**item, "archive_path": str(dest)}
        if dry_run:
            moved.append(record)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(dest.name + f".{stamp}")
            record["archive_path"] = str(dest)
        shutil.move(str(src), str(dest))
        moved.append(record)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dry_run": dry_run,
        "archive_root": str(archive_root),
        "moved_count": len(moved),
        "skipped_count": len(skipped),
        "moved_size_bytes": sum(int(item.get("size_bytes") or 0) for item in moved),
        "moved": moved,
        "skipped": skipped,
    }
    if not dry_run:
        (archive_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
