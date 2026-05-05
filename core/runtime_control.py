from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import zipfile
from datetime import datetime, timezone
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
    NEUROGOLF_SEED_CONTROLS_PATH,
    NEUROGOLF_SYNC_STATE_PATH,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
)
from core.env_settings import load_env_settings, update_env_settings
from core.distillation import (
    distillation_status,
    initialize_distillation_plan,
    set_fine_tune_policy,
)
from core.maintenance import archive_clutter, find_clutter
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
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def daemon_status() -> dict:
    pid = None
    if NEUROGOLF_AUTONOMY_PID_PATH.exists():
        raw = NEUROGOLF_AUTONOMY_PID_PATH.read_text(encoding="utf-8").strip()
        pid = int(raw) if raw.isdigit() else None
    status_payload = read_status()
    status_updated_at = str(status_payload.get("updated_at", "")).strip()
    stale_seconds = None
    stale = False
    if status_updated_at:
        try:
            stamp = datetime.fromisoformat(status_updated_at.replace("Z", "+00:00"))
            stale_seconds = max(0.0, (datetime.now(stamp.tzinfo) - stamp).total_seconds())
            stale = stale_seconds > 1800
        except Exception:
            stale_seconds = None
    if pid and psutil.pid_exists(pid):
        proc = psutil.Process(pid)
        return {
            "running": True,
            "pid": pid,
            "created_at": datetime.fromtimestamp(proc.create_time()).isoformat(timespec="seconds"),
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "status": proc.status(),
            "status_updated_at": status_updated_at or None,
            "status_stale": stale,
            "status_age_seconds": stale_seconds,
        }
    return {
        "running": False,
        "pid": pid,
        "status_updated_at": status_updated_at or None,
        "status_stale": stale,
        "status_age_seconds": stale_seconds,
    }


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


def read_seed_controls() -> dict:
    if not NEUROGOLF_SEED_CONTROLS_PATH.exists():
        return {"seeds": {}}
    try:
        payload = json.loads(NEUROGOLF_SEED_CONTROLS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"seeds": {}}
    if not isinstance(payload, dict):
        return {"seeds": {}}
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict):
        seeds = {key: value for key, value in payload.items() if isinstance(value, dict)}
    return {"seeds": seeds}


def skynet_clutter_summary() -> dict:
    items = find_clutter()
    total_bytes = sum(int(item.get("size_bytes") or 0) for item in items)
    by_kind: dict[str, int] = {}
    for item in items:
        kind = str(item.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "count": len(items),
        "total_bytes": total_bytes,
        "by_kind": by_kind,
        "items": items[:200],
    }


def archive_skynet_clutter(*, dry_run: bool = False) -> dict:
    manifest = archive_clutter(dry_run=dry_run)
    return {
        "ok": True,
        "stdout": (
            f"{'Would archive' if dry_run else 'Archived'} {manifest['moved_count']} items "
            f"({manifest['moved_size_bytes']} bytes) under {manifest['archive_root']}"
        ),
        "stderr": "" if not manifest.get("skipped") else f"Skipped {manifest['skipped_count']} items",
        "manifest": manifest,
    }


def read_distillation_status() -> dict:
    return distillation_status()


def reset_distillation_plan(*, teacher_model: str = "", dataset_path: str = "") -> dict:
    plan = initialize_distillation_plan(teacher_model=teacher_model, dataset_path=dataset_path)
    return {"ok": True, "stdout": f"Distillation plan initialized at {distillation_status()['plan_path']}", "stderr": "", "plan": plan}


def save_distillation_settings(*, enabled: bool, dataset_path: str, output_adapter_path: str, teacher_model: str) -> dict:
    plan = set_fine_tune_policy(
        enabled=enabled,
        dataset_path=dataset_path,
        output_adapter_path=output_adapter_path,
        teacher_model=teacher_model,
    )
    return {"ok": True, "stdout": "Distillation settings saved.", "stderr": "", "plan": plan}


def save_imported_seed_controls(rows: list[dict]) -> dict:
    payload = read_seed_controls()
    seeds = dict(payload.get("seeds", {}))
    changed = 0
    for row in rows:
        seed_label = str(row.get("seed_label", "")).strip()
        if not seed_label:
            continue
        control = dict(seeds.get(seed_label, {}))
        score_raw = row.get("manual_public_score")
        manual_score = None
        if score_raw not in (None, ""):
            try:
                parsed_score = float(score_raw)
            except (TypeError, ValueError):
                return {"ok": False, "stdout": "", "stderr": f"Invalid manual score for {seed_label}: {score_raw}"}
            if math.isfinite(parsed_score):
                manual_score = parsed_score
        if manual_score is None:
            control.pop("manual_public_score", None)
        else:
            control["manual_public_score"] = manual_score
        control["disabled"] = bool(row.get("disabled", False))
        note = str(row.get("note", "")).strip()
        if note:
            control["note"] = note
        else:
            control.pop("note", None)
        seeds[seed_label] = control
        changed += 1
    NEUROGOLF_SEED_CONTROLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUROGOLF_SEED_CONTROLS_PATH.write_text(json.dumps({"seeds": seeds}, indent=2), encoding="utf-8")
    sync_result = sync_state(history=int(load_env_settings().get("NEUROGOLF_AUTONOMY_MAX_HISTORY", "10")))
    return {
        "ok": bool(sync_result.get("ok", True)),
        "stdout": f"Saved controls for {changed} imported seeds.\n{sync_result.get('stdout', '')}".strip(),
        "stderr": str(sync_result.get("stderr", "")).strip(),
    }


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
    current = daemon_status()
    if current.get("running"):
        return {
            "ok": True,
            "stdout": f"Autonomy daemon already running with PID {current.get('pid')}",
            "stderr": "",
        }
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
    reports_dir = NEUROGOLF_AUTONOMY_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    launcher_out = reports_dir / "launcher.out.log"
    launcher_err = reports_dir / "launcher.err.log"
    stdout_handle = launcher_out.open("a", encoding="utf-8")
    stderr_handle = launcher_err.open("a", encoding="utf-8")
    subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout_handle.close()
    stderr_handle.close()
    return {
        "ok": True,
        "stdout": "Autonomy daemon start requested in background. Check live status or launcher logs if boot takes a moment.",
        "stderr": "",
    }


def stop_daemon(*, reset_state: bool = True) -> dict:
    """Stop the autonomy daemon and optionally reset state for fresh start.
    
    When reset_state=True (default):
    - Clears learning_state.json (forgets blocked branches, streaks, etc.)
    - Resets status to idle
    - Next start will begin from scratch
    """
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "stop_autonomous_neurogolf.ps1"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    result = {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}
    
    if reset_state:
        # Clear learning state so next start is fresh
        learning_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / "learning_state.json"
        if learning_path.exists():
            try:
                backup_name = f"learning_state.json.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                backup_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / backup_name
                learning_path.rename(backup_path)
                result["stdout"] += f"\nLearning state backed up to {backup_name}"
            except Exception as e:
                result["stderr"] += f"\nCould not backup learning state: {e}"
        
        # Reset status to idle
        try:
            status_path = NEUROGOLF_AUTONOMY_STATUS_PATH
            if status_path.exists():
                reset_status = {
                    "phase": "idle",
                    "message": "Daemon stopped - fresh start on next launch",
                    "progress": 0.0,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "reset": True,
                }
                status_path.write_text(json.dumps(reset_status, indent=2), encoding="utf-8")
                result["stdout"] += "\nStatus reset to idle (fresh start on next launch)"
        except Exception as e:
            result["stderr"] += f"\nCould not reset status: {e}"
        
        # Clear cycle_error files to reset error counters
        try:
            error_files = list(NEUROGOLF_AUTONOMY_REPORTS_DIR.glob("cycle_error_*.json"))
            for ef in error_files:
                ef.unlink()
            if error_files:
                result["stdout"] += f"\nCleared {len(error_files)} cycle error files"
        except Exception:
            pass
    
    return result


def restart_daemon_async(*, allow_submit: bool, history: int, min_local_delta: float, sleep_seconds: int, max_pending_submissions: int) -> dict:
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
        str(PROJECT_ROOT / "restart_autonomous_neurogolf.ps1"),
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
    subprocess.Popen(command, cwd=PROJECT_ROOT)
    return {
        "ok": True,
        "stdout": "Restart requested in background.",
        "stderr": "",
    }


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
    state = read_state()
    imported_from_state = state.get("imported_sources")
    if isinstance(imported_from_state, list) and imported_from_state:
        return imported_from_state[:limit]
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
            encoding="utf-8",
            errors="replace",
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
        if item["kind"] == "kernel":
            output_dir = target_dir / "kernel_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_command = _kaggle_cli_prefix() + [
                "kernels",
                "output",
                f"{item['owner']}/{item['slug']}",
                "-p",
                str(output_dir),
                "-o",
            ]
            output_completed = subprocess.run(
                output_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PROJECT_ROOT,
                env=_kaggle_auth_env(),
                timeout=1800,
            )
            result["output_ok"] = output_completed.returncode == 0
            result["output_path"] = str(output_dir)
            result["output_stdout"] = output_completed.stdout.strip()
            result["output_stderr"] = output_completed.stderr.strip()
            if output_completed.returncode != 0:
                result["ok"] = False
        results.append(result)
        if not result["ok"]:
            failures.append(f"{item['kind']} {item['owner']}/{item['slug']}: {result['stderr'] or result['stdout'] or 'unknown error'}")
        elif item["kind"] == "kernel" and not result.get("output_ok", False):
            failures.append(
                f"{item['kind']} {item['owner']}/{item['slug']} output: "
                f"{result.get('output_stderr') or result.get('output_stdout') or 'unknown error'}"
            )

    try:
        sync_result = sync_state(history=int(load_env_settings().get("NEUROGOLF_AUTONOMY_MAX_HISTORY", "10")))
    except Exception as exc:
        sync_result = {"ok": False, "stdout": "", "stderr": f"sync failed: {type(exc).__name__}: {exc}"}

    stdout = json.dumps(results, indent=2)
    stderr_lines = list(failures)
    if not sync_result.get("ok"):
        stderr_lines.append(sync_result.get("stderr", "state sync failed"))
    return {"ok": not failures and bool(sync_result.get("ok")), "stdout": stdout, "stderr": "\n".join(line for line in stderr_lines if line)}


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


def remote_access_endpoints(port: int = 8501) -> dict:
    lan_urls: list[str] = []
    try:
        for entries in psutil.net_if_addrs().values():
            for entry in entries:
                if getattr(entry, "family", None) != socket.AF_INET:
                    continue
                address = str(getattr(entry, "address", "")).strip()
                if not address or address.startswith("127.") or address == "0.0.0.0":
                    continue
                lan_urls.append(f"http://{address}:{port}")
    except Exception:
        pass

    tailscale_urls: list[str] = []
    tailscale_binary = shutil.which("tailscale")
    if tailscale_binary:
        try:
            completed = subprocess.run(
                [tailscale_binary, "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    address = line.strip()
                    if address:
                        tailscale_urls.append(f"http://{address}:{port}")
        except Exception:
            pass

    return {
        "hostname": socket.gethostname(),
        "localhost_url": f"http://localhost:{port}",
        "lan_urls": sorted(set(lan_urls)),
        "tailscale_urls": sorted(set(tailscale_urls)),
        "recommended_remote": (sorted(set(tailscale_urls)) or sorted(set(lan_urls)) or [f"http://localhost:{port}"])[0],
    }
