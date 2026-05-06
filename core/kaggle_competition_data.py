"""Kaggle Competition Data Manager for NeuroGolf 2026.

Handles downloading, caching, and accessing competition data and utilities.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
)
from core.runtime_control import _kaggle_auth_env

COMPETITION_NAME = "neurogolf-2026"
COMPETITION_DATA_DIR = DEFAULT_DATA_DIR / COMPETITION_NAME
COMPETITION_METADATA_PATH = COMPETITION_DATA_DIR / "metadata.json"


def _kaggle_cli_prefix() -> list[str]:
    """Build Kaggle CLI prefix with proper Python executable."""
    return [str(Path(sys.executable).parent / "kaggle")]


def ensure_competition_data_dir() -> Path:
    """Ensure the competition data directory exists."""
    COMPETITION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return COMPETITION_DATA_DIR


def download_competition_data(force: bool = False) -> dict:
    """Download the competition data from Kaggle.
    
    Args:
        force: If True, re-download even if data already exists.
        
    Returns:
        dict with status, paths, and any errors.
    """
    ensure_competition_data_dir()
    
    # Check if already downloaded
    if not force and COMPETITION_METADATA_PATH.exists():
        metadata = json.loads(COMPETITION_METADATA_PATH.read_text())
        if metadata.get("downloaded", False):
            return {
                "status": "already_exists",
                "path": str(COMPETITION_DATA_DIR),
                "metadata": metadata,
            }
    
    # Download using Kaggle CLI
    command = _kaggle_cli_prefix() + [
        "competitions", "download",
        "-c", COMPETITION_NAME,
        "-p", str(COMPETITION_DATA_DIR),
        "--force" if force else ""
    ]
    command = [c for c in command if c]  # Remove empty strings
    
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            env=_kaggle_auth_env(),
            timeout=300,
        )
        
        if completed.returncode != 0:
            return {
                "status": "error",
                "error": completed.stderr.strip(),
                "command": " ".join(command),
            }
        
        # Unzip the downloaded file
        zip_files = list(COMPETITION_DATA_DIR.glob("*.zip"))
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(COMPETITION_DATA_DIR)
                # Keep the zip file but note it was extracted
            except zipfile.BadZipFile:
                pass
        
        # Save metadata
        metadata = {
            "downloaded": True,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "competition": COMPETITION_NAME,
            "zip_files": [str(z.name) for z in zip_files],
        }
        COMPETITION_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        
        return {
            "status": "success",
            "path": str(COMPETITION_DATA_DIR),
            "metadata": metadata,
            "files": [str(f.relative_to(COMPETITION_DATA_DIR)) for f in COMPETITION_DATA_DIR.rglob("*") if f.is_file()],
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Download timed out after 300 seconds",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def list_competition_files() -> list[dict]:
    """List all files in the competition data directory."""
    if not COMPETITION_DATA_DIR.exists():
        return []
    
    files = []
    for path in COMPETITION_DATA_DIR.rglob("*"):
        if path.is_file() and path.name != "metadata.json":
            files.append({
                "path": str(path.relative_to(COMPETITION_DATA_DIR)),
                "size": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    return files


def get_competition_file(path: str | Path) -> Path | None:
    """Get the full path to a competition file.
    
    Args:
        path: Relative path within the competition data directory.
        
    Returns:
        Full Path if file exists, None otherwise.
    """
    full_path = COMPETITION_DATA_DIR / path
    if full_path.exists() and full_path.is_file():
        return full_path
    return None


def load_competition_metadata() -> dict | None:
    """Load the competition metadata if available."""
    if COMPETITION_METADATA_PATH.exists():
        try:
            return json.loads(COMPETITION_METADATA_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return None


def is_competition_data_ready() -> bool:
    """Check if competition data has been downloaded."""
    metadata = load_competition_metadata()
    return metadata is not None and metadata.get("downloaded", False)


def get_task_example_data(task_id: str) -> dict | None:
    """Load example data for a specific task if available.
    
    Args:
        task_id: The task identifier (e.g., "task001")
        
    Returns:
        Task data dict or None if not found.
    """
    # Look for task files in various formats
    possible_paths = [
        f"tasks/{task_id}.json",
        f"{task_id}.json",
        f"task/{task_id}.json",
        f"examples/{task_id}.json",
    ]
    
    for rel_path in possible_paths:
        full_path = get_competition_file(rel_path)
        if full_path:
            try:
                return json.loads(full_path.read_text())
            except (json.JSONDecodeError, IOError):
                pass
    
    return None


def download_public_dataset(owner: str, slug: str) -> dict:
    """Download a public Kaggle dataset into the competition data directory.
    
    Args:
        owner: Dataset owner username
        slug: Dataset slug/name
        
    Returns:
        dict with status and paths.
    """
    import sys
    
    target_dir = COMPETITION_DATA_DIR / "datasets" / f"{owner}__{slug}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        str(Path(sys.executable).parent / "kaggle"),
        "datasets", "download",
        f"{owner}/{slug}",
        "-p", str(target_dir),
        "--unzip"
    ]
    
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=PROJECT_ROOT,
            env=_kaggle_auth_env(),
            timeout=300,
        )
        
        return {
            "status": "success" if completed.returncode == 0 else "error",
            "path": str(target_dir),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip() if completed.stderr else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


import sys
