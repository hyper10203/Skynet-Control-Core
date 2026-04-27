from __future__ import annotations

import csv
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import psutil
import requests

from core.config import (
    MODEL_OPTIONS,
    MODEL_REGISTRY,
    NEUROGOLF_AUTONOMY_LOG_PATH,
    NEUROGOLF_AUTONOMY_PID_PATH,
    NEUROGOLF_AUTONOMY_REPORTS_DIR,
    NEUROGOLF_AUTONOMY_STATUS_PATH,
    NEUROGOLF_IMPORTED_SOURCES_DIR,
    NEUROGOLF_OPERATOR_NOTE_PATH,
    NEUROGOLF_PROJECT_ROOT,
    NEUROGOLF_SYNC_STATE_PATH,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
)
from core.env_settings import load_env_settings, update_env_settings
from core.model_profiles import default_context_for_model


def _normalize_csv_row(row: dict) -> dict:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        clean_key = str(key or "").replace("\ufeff", "").strip()
        normalized[clean_key] = value
    return normalized


def _python_exe() -> str:
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else "python"


def _kaggle_cli_prefix() -> list[str]:
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "kaggle.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "kaggle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    for name in ("kaggle.exe", "kaggle"):
        resolved = shutil.which(name)
        if resolved:
            return [resolved]
    return [_python_exe(), "-c", "from kaggle.cli import main; main()"]


def _kaggle_auth_env() -> dict[str, str]:
    env = os.environ.copy()
    kaggle_json = NEUROGOLF_PROJECT_ROOT / "kaggle.json"
    if kaggle_json.exists():
        env["KAGGLE_CONFIG_DIR"] = str(NEUROGOLF_PROJECT_ROOT)
    return env


def daemon_status() -> dict:
    pid = None
    if NEUROGOLF_AUTONOMY_PID_PATH.exists():
        raw = NEUROGOLF_AUTONOMY_PID_PATH.read_text(encoding="utf-8").strip()
        pid = int(raw) if raw.isdigit() else None
    if pid and psutil.pid_exists(pid):
        proc = psutil.Process(pid)
        return {
            "running": True,
            "pid": pid,
            "created_at": datetime.fromtimestamp(proc.create_time()).isoformat(timespec="seconds"),
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "status": proc.status(),
        }
    return {"running": False, "pid": pid}


def read_status() -> dict:
    if not NEUROGOLF_AUTONOMY_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(NEUROGOLF_AUTONOMY_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_state() -> dict:
    if not NEUROGOLF_SYNC_STATE_PATH.exists():
        return {}
    try:
        return json.loads(NEUROGOLF_SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def recent_reports(limit: int = 10) -> list[dict]:
    reports: list[dict] = []
    if not NEUROGOLF_AUTONOMY_REPORTS_DIR.exists():
        return reports
    for path in sorted(NEUROGOLF_AUTONOMY_REPORTS_DIR.glob("cycle*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_path"] = str(path)
        reports.append(payload)
    return reports


def tail_log(lines: int = 120) -> str:
    if not NEUROGOLF_AUTONOMY_LOG_PATH.exists():
        return ""
    content = NEUROGOLF_AUTONOMY_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def persist_runtime_preferences(*, allow_submit: bool, history: int, min_local_delta: float, sleep_seconds: int, max_pending_submissions: int) -> None:
    update_env_settings(
        {
            "NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT": "1" if allow_submit else "0",
            "NEUROGOLF_AUTONOMY_MAX_HISTORY": str(history),
            "NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD": str(min_local_delta),
            "NEUROGOLF_AUTONOMY_LOOP_SECONDS": str(sleep_seconds),
            "NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS": str(max_pending_submissions),
        }
    )


def start_daemon(*, allow_submit: bool, history: int, min_local_delta: float, sleep_seconds: int, max_pending_submissions: int) -> dict:
    persist_runtime_preferences(
        allow_submit=allow_submit,
        history=history,
        min_local_delta=min_local_delta,
        sleep_seconds=sleep_seconds,
        max_pending_submissions=max_pending_submissions,
    )
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "start_autonomous_neurogolf.ps1"),
        "-History",
        str(history),
        "-MinLocalDelta",
        str(min_local_delta),
        "-SleepSeconds",
        str(sleep_seconds),
        "-MaxPendingSubmissions",
        str(max_pending_submissions),
    ]
    if allow_submit:
        command.append("-AllowSubmit")
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def stop_daemon() -> dict:
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "stop_autonomous_neurogolf.ps1"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def sync_state(history: int = 10) -> dict:
    command = [_python_exe(), str(PROJECT_ROOT / "sync_neurogolf_artifacts.py"), "--history", str(history)]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT, env=_kaggle_auth_env())
    return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def healthcheck() -> dict:
    command = [_python_exe(), str(PROJECT_ROOT / "healthcheck.py")]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def run_single_cycle(*, allow_submit: bool, history: int, min_local_delta: float) -> dict:
    env_settings = load_env_settings()
    sleep_seconds = int(env_settings.get("NEUROGOLF_AUTONOMY_LOOP_SECONDS", "1800"))
    max_pending_submissions = int(env_settings.get("NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS", "1"))
    persist_runtime_preferences(
        allow_submit=allow_submit,
        history=history,
        min_local_delta=min_local_delta,
        sleep_seconds=sleep_seconds,
        max_pending_submissions=max_pending_submissions,
    )
    command = [
        _python_exe(),
        str(PROJECT_ROOT / "autonomous_neurogolf.py"),
        "--history",
        str(history),
        "--min-local-delta",
        str(min_local_delta),
    ]
    if allow_submit:
        command.append("--allow-submit")
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT, env=_kaggle_auth_env())
    return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def ollama_models() -> list[dict]:
    response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=30)
    response.raise_for_status()
    rows = response.json().get("models", [])
    return [
        {
            "name": row.get("name"),
            "size_gb": round(float(row.get("size", 0)) / (1024 ** 3), 2) if row.get("size") is not None else None,
            "modified_at": row.get("modified_at"),
        }
        for row in rows
    ]


def current_role_map() -> list[dict]:
    env_settings = load_env_settings()
    return [
        {
            "role": role,
            "model": env_settings.get(f"ARC_MODEL_{role.upper()}", model),
            "context": int(env_settings.get(f"ARC_CTX_{role.upper()}", MODEL_OPTIONS.get(role, {}).get("num_ctx", default_context_for_model(model, role)))),
            "temperature": float(env_settings.get(f"ARC_TEMP_{role.upper()}", MODEL_OPTIONS.get(role, {}).get("temperature", 0.2))),
            "predict": int(env_settings.get(f"ARC_PREDICT_{role.upper()}", MODEL_OPTIONS.get(role, {}).get("num_predict", 256))),
        }
        for role, model in MODEL_REGISTRY.items()
    ]


def _parse_kaggle_reference(text: str) -> dict | None:
    value = str(text or "").strip()
    if not value:
        return None

    if value.lower().startswith("kaggle "):
        try:
            parts = shlex.split(value)
        except ValueError:
            return None
        if len(parts) >= 4 and parts[0] == "kaggle" and parts[1] in {"kernels", "datasets"}:
            subject = parts[1]
            action = parts[2]
            ref = parts[3].strip()
            if action not in {"pull", "download"} or "/" not in ref:
                return None
            owner, slug = ref.split("/", 1)
            return {"kind": "kernel" if subject == "kernels" else "dataset", "owner": owner, "slug": slug, "source": value}

    match = re.search(r"kaggle\.com/(code|datasets)/([^/\s]+)/([^/?#\s]+)", value, flags=re.IGNORECASE)
    if match:
        return {
            "kind": "kernel" if match.group(1).lower() == "code" else "dataset",
            "owner": match.group(2),
            "slug": match.group(3),
            "source": value,
        }
    return None


def list_imported_kaggle_sources(limit: int = 20) -> list[dict]:
    base_dir = NEUROGOLF_IMPORTED_SOURCES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in sorted(base_dir.glob("*/*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        rows.append(
            {
                "kind": path.parent.name,
                "name": path.name,
                "path": str(path),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def fetch_kaggle_targets(raw_text: str) -> dict:
    base_dir = NEUROGOLF_IMPORTED_SOURCES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    parsed = []
    for line in str(raw_text or "").splitlines():
        candidate = _parse_kaggle_reference(line)
        if candidate:
            parsed.append(candidate)

    if not parsed:
        return {"ok": False, "stdout": "", "stderr": "No valid Kaggle kernel or dataset references found."}

    results: list[dict] = []
    failures: list[str] = []
    for item in parsed:
        target_dir = base_dir / f"{item['kind']}s" / f"{item['owner']}__{item['slug']}"
        target_dir.mkdir(parents=True, exist_ok=True)
        if item["kind"] == "kernel":
            command = _kaggle_cli_prefix() + ["kernels", "pull", f"{item['owner']}/{item['slug']}", "-p", str(target_dir)]
        else:
            command = _kaggle_cli_prefix() + ["datasets", "download", f"{item['owner']}/{item['slug']}", "-p", str(target_dir), "--unzip"]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=_kaggle_auth_env(),
            timeout=900,
        )
        result = {
            "kind": item["kind"],
            "ref": f"{item['owner']}/{item['slug']}",
            "path": str(target_dir),
            "ok": completed.returncode == 0,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        results.append(result)
        if not result["ok"]:
            failures.append(f"{item['kind']} {item['owner']}/{item['slug']}: {result['stderr'] or result['stdout'] or 'unknown error'}")

    stdout = json.dumps(results, indent=2)
    stderr = "\n".join(failures)
    return {"ok": not failures, "stdout": stdout, "stderr": stderr}


def leaderboard_snapshot(limit: int = 100) -> list[dict]:
    output_dir = NEUROGOLF_AUTONOMY_REPORTS_DIR / "leaderboard_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "neurogolf-2026.zip"
    csv_rows: list[dict] = []
    command = [
        _python_exe(),
        "-c",
        (
            "import os; "
            f"os.environ['KAGGLE_CONFIG_DIR']=r'{NEUROGOLF_PROJECT_ROOT}'; "
            "from kaggle.api.kaggle_api_extended import KaggleApi; "
            "api=KaggleApi(); api.authenticate(); "
            f"api.competition_leaderboard_download('neurogolf-2026', r'{output_dir}', quiet=True)"
        ),
    ]
    subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT, env=_kaggle_auth_env(), timeout=240)
    if not zip_path.exists():
        return csv_rows
    with zipfile.ZipFile(zip_path) as archive:
        csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), None)
        if not csv_name:
            return csv_rows
        with archive.open(csv_name) as handle:
            decoded = io.TextIOWrapper(handle, encoding="utf-8-sig")
            reader = csv.DictReader(decoded)
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                csv_rows.append(_normalize_csv_row(row))
    return csv_rows


def leaderboard_rank_for_team(team_name: str, *, limit: int = 2000) -> dict:
    rows = leaderboard_snapshot(limit=limit)
    normalized_target = team_name.strip().lower()
    for row in rows:
        if str(row.get("TeamName", "")).strip().lower() == normalized_target:
            rank = row.get("Rank")
            score = row.get("Score")
            submission_count = row.get("SubmissionCount")
            return {
                "rank": int(rank) if rank not in (None, "") else None,
                "score": float(score) if score not in (None, "") else None,
                "team_name": row["TeamName"],
                "submission_count": int(submission_count) if submission_count not in (None, "") else None,
            }
    return {}


def latest_team_name() -> str:
    state = read_state()
    submissions = state.get("recent_submissions", [])
    for item in submissions:
        score = item.get("public_score")
        if score is not None:
            return str(item.get("team_name") or "").strip()
    return ""


def operator_note() -> str:
    if not NEUROGOLF_OPERATOR_NOTE_PATH.exists():
        return ""
    return NEUROGOLF_OPERATOR_NOTE_PATH.read_text(encoding="utf-8")


def save_operator_note(text: str) -> Path:
    NEUROGOLF_OPERATOR_NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUROGOLF_OPERATOR_NOTE_PATH.write_text(text, encoding="utf-8")
    return NEUROGOLF_OPERATOR_NOTE_PATH
