from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from core.config import (
    AXIOMGRAPH_BRIDGE_BRANCH,
    AXIOMGRAPH_BRIDGE_GITHUB_TOKEN,
    AXIOMGRAPH_BRIDGE_REPO,
    AXIOMGRAPH_BRIDGE_ROOT,
)
from core.env_settings import load_env_settings
from core.runtime_node import load_runtime_node_state, load_runtime_node_summary, save_runtime_node_state


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bridge_settings() -> dict[str, str]:
    env_settings = load_env_settings()
    return {
        "repo": str(env_settings.get("AXIOMGRAPH_BRIDGE_REPO") or AXIOMGRAPH_BRIDGE_REPO).strip(),
        "branch": str(env_settings.get("AXIOMGRAPH_BRIDGE_BRANCH") or AXIOMGRAPH_BRIDGE_BRANCH).strip(),
        "root": str(env_settings.get("AXIOMGRAPH_BRIDGE_ROOT") or AXIOMGRAPH_BRIDGE_ROOT).strip() or "bridge",
    }


def _bridge_path(*parts: str) -> str:
    root = bridge_settings()["root"].strip("/").replace("\\", "/")
    return "/".join([root, *[str(part).strip("/").replace("\\", "/") for part in parts if str(part).strip("/")]])


def _resolve_token(explicit_token: str | None = None, *, allow_cli: bool = False) -> str:
    if explicit_token:
        return explicit_token.strip()
    env_token = os.getenv("AXIOMGRAPH_BRIDGE_GITHUB_TOKEN", "").strip()
    if env_token:
        return env_token
    if AXIOMGRAPH_BRIDGE_GITHUB_TOKEN:
        return AXIOMGRAPH_BRIDGE_GITHUB_TOKEN.strip()
    if allow_cli:
        try:
            completed = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
        except Exception:
            pass
    return ""


def bridge_token_available(explicit_token: str | None = None, *, allow_cli: bool = False) -> bool:
    return bool(_resolve_token(explicit_token, allow_cli=allow_cli))


def _github_headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AxiomGraphRuntimeNode/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _contents_url(repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def _raw_url(repo: str, branch: str, path: str) -> str:
    owner, name = repo.split("/", 1)
    return f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}"


def _get_contents_metadata(path: str, *, repo: str, branch: str, token: str | None = None) -> requests.Response:
    return requests.get(
        _contents_url(repo, path),
        headers=_github_headers(token),
        params={"ref": branch},
        timeout=30,
    )


def read_repo_json(path: str, *, repo: str | None = None, branch: str | None = None, token: str | None = None) -> dict | list | None:
    settings = bridge_settings()
    repo = repo or settings["repo"]
    branch = branch or settings["branch"]
    response = requests.get(
        _raw_url(repo, branch, path),
        headers=_github_headers(token),
        timeout=30,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def put_repo_json(
    path: str,
    payload: dict | list,
    *,
    message: str,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
) -> dict:
    settings = bridge_settings()
    repo = repo or settings["repo"]
    branch = branch or settings["branch"]
    token = _resolve_token(token, allow_cli=True)
    if not token:
        raise RuntimeError("No GitHub token available for bridge write.")

    sha = None
    meta_response = _get_contents_metadata(path, repo=repo, branch=branch, token=token)
    if meta_response.status_code == 200:
        try:
            sha = meta_response.json().get("sha")
        except Exception:
            sha = None
    elif meta_response.status_code != 404:
        meta_response.raise_for_status()

    body = {
        "message": message,
        "content": base64.b64encode(json.dumps(payload, indent=2).encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha

    response = requests.put(
        _contents_url(repo, path),
        headers=_github_headers(token),
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def list_bridge_nodes(*, repo: str | None = None, branch: str | None = None, token: str | None = None) -> list[dict]:
    settings = bridge_settings()
    repo = repo or settings["repo"]
    branch = branch or settings["branch"]
    path = _bridge_path("nodes")
    response = _get_contents_metadata(path, repo=repo, branch=branch, token=token)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    entries = response.json()
    if not isinstance(entries, list):
        return []

    nodes: list[dict] = []
    for entry in entries:
        download_url = entry.get("download_url")
        if not download_url:
            continue
        try:
            payload = requests.get(download_url, timeout=30).json()
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["_bridge_path"] = entry.get("path")
            nodes.append(payload)
    return sorted(nodes, key=lambda item: str(item.get("last_seen_at") or ""), reverse=True)


def _build_runtime_node_payload() -> dict[str, Any]:
    from core.runtime_control import daemon_status, read_state, read_status, recent_reports

    summary = load_runtime_node_summary()
    state = read_state() or {}
    status = read_status() or {}
    daemon = daemon_status() or {}
    reports = recent_reports(limit=3)
    best_completed = state.get("best_completed_public_score")
    latest_submissions = state.get("recent_submissions", [])
    latest_completed = next(
        (
            row
            for row in latest_submissions
            if isinstance(row, dict) and str(row.get("status", "")).lower() == "complete"
        ),
        {},
    )
    payload: dict[str, Any] = {
        "node_id": summary.get("node_id"),
        "device_label": summary.get("device_label"),
        "remote_control_url": summary.get("remote_control_url"),
        "workspace_root": summary.get("workspace_root"),
        "kaggle_username": summary.get("kaggle_username"),
        "daemon": daemon,
        "phase": status.get("phase", "idle"),
        "message": status.get("message", ""),
        "progress": status.get("progress", 0.0),
        "best_completed_public_score": best_completed,
        "latest_completed_public_score": latest_completed.get("public_score"),
        "pending_submission_count": state.get("pending_submission_count", 0),
        "team_name": state.get("current_team_name"),
        "current_manifest": state.get("current_manifest", {}),
        "submission_policy": state.get("submission_policy", state.get("campaign_progress", {})),
        "latest_report": reports[0] if reports else {},
        "last_seen_at": _utc_now(),
        "pairing_code": summary.get("pairing_code"),
        "bridge_branch": bridge_settings()["branch"],
    }
    return payload


def publish_runtime_node_heartbeat(*, token: str | None = None) -> dict:
    payload = _build_runtime_node_payload()
    node_id = str(payload.get("node_id") or "unknown-node")
    path = _bridge_path("nodes", f"{node_id}.json")
    put_repo_json(
        path,
        payload,
        message=f"heartbeat: {node_id} @ {_utc_now()}",
        token=token,
    )
    return {
        "ok": True,
        "stdout": f"Published runtime node heartbeat to {path}",
        "stderr": "",
        "payload": payload,
        "path": path,
    }


def queue_runtime_node_command(
    node_id: str,
    action: str,
    *,
    args: dict[str, Any] | None = None,
    requested_by: str = "remote-control-center",
    token: str | None = None,
) -> dict:
    clean_node_id = str(node_id or "").strip()
    if not clean_node_id:
        return {"ok": False, "stdout": "", "stderr": "Node ID is required."}
    payload = {
        "id": f"cmd_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}",
        "node_id": clean_node_id,
        "action": str(action or "").strip(),
        "args": args or {},
        "status": "pending",
        "requested_at": _utc_now(),
        "requested_by": requested_by,
    }
    path = _bridge_path("commands", f"{clean_node_id}.json")
    put_repo_json(
        path,
        payload,
        message=f"command: {clean_node_id} -> {payload['action']}",
        token=token,
    )
    return {
        "ok": True,
        "stdout": f"Queued command `{payload['action']}` for node {clean_node_id}.",
        "stderr": "",
        "command": payload,
        "path": path,
    }


def fetch_runtime_node_command(node_id: str, *, repo: str | None = None, branch: str | None = None, token: str | None = None) -> dict | None:
    clean_node_id = str(node_id or "").strip()
    if not clean_node_id:
        return None
    path = _bridge_path("commands", f"{clean_node_id}.json")
    payload = read_repo_json(path, repo=repo, branch=branch, token=token)
    return payload if isinstance(payload, dict) else None


def _acknowledge_runtime_node_command(command: dict, result: dict, *, token: str | None = None) -> dict:
    node_id = str(command.get("node_id") or "").strip()
    path = _bridge_path("commands", f"{node_id}.json")
    payload = dict(command)
    payload["status"] = "executed" if result.get("ok", False) else "failed"
    payload["executed_at"] = _utc_now()
    payload["result"] = {
        "ok": bool(result.get("ok", False)),
        "stdout": str(result.get("stdout", ""))[:4000],
        "stderr": str(result.get("stderr", ""))[:4000],
    }
    put_repo_json(
        path,
        payload,
        message=f"command result: {node_id} -> {payload.get('action')}",
        token=token,
    )
    return payload


def execute_runtime_node_command(action: str, args: dict[str, Any] | None = None) -> dict:
    args = args or {}
    from core.runtime_control import (
        restart_daemon_async,
        run_single_cycle,
        runtime_node_healthcheck,
        save_operator_note,
        start_daemon,
        stop_daemon,
        sync_state,
    )

    action = str(action or "").strip()
    if action == "run_healthcheck":
        return runtime_node_healthcheck()
    if action == "sync_state":
        return sync_state(history=int(args.get("history", 10)))
    if action == "run_single_cycle":
        return run_single_cycle(
            allow_submit=bool(args.get("allow_submit", False)),
            history=int(args.get("history", 10)),
            min_local_delta=float(args.get("min_local_delta", 0.0)),
        )
    if action == "restart_daemon":
        return restart_daemon_async(
            allow_submit=bool(args.get("allow_submit", False)),
            history=int(args.get("history", 10)),
            min_local_delta=float(args.get("min_local_delta", 0.0)),
            sleep_seconds=int(args.get("sleep_seconds", 1800)),
            max_pending_submissions=int(args.get("max_pending_submissions", 1)),
        )
    if action == "start_daemon":
        return start_daemon(
            allow_submit=bool(args.get("allow_submit", False)),
            history=int(args.get("history", 10)),
            min_local_delta=float(args.get("min_local_delta", 0.0)),
            sleep_seconds=int(args.get("sleep_seconds", 1800)),
            max_pending_submissions=int(args.get("max_pending_submissions", 1)),
        )
    if action == "stop_daemon":
        return stop_daemon(reset_state=bool(args.get("reset_state", False)))
    if action == "save_operator_note":
        save_operator_note(str(args.get("note", "")))
        return {"ok": True, "stdout": "Operator note saved from remote command.", "stderr": ""}
    if action == "publish_heartbeat":
        return publish_runtime_node_heartbeat()
    return {"ok": False, "stdout": "", "stderr": f"Unsupported runtime-node action: {action}"}


def process_runtime_node_command(*, token: str | None = None) -> dict:
    summary = load_runtime_node_summary()
    node_id = str(summary.get("node_id") or "").strip()
    if not node_id:
        return {"ok": False, "stdout": "", "stderr": "Runtime node has no node_id."}

    command = fetch_runtime_node_command(node_id)
    if not command:
        return {"ok": True, "stdout": "No remote command queued.", "stderr": "", "command": None}

    command_id = str(command.get("id") or "").strip()
    if not command_id:
        return {"ok": False, "stdout": "", "stderr": "Queued command has no ID.", "command": command}

    state = load_runtime_node_state()
    if state.get("last_processed_command_id") == command_id:
        return {"ok": True, "stdout": f"Command {command_id} already processed.", "stderr": "", "command": command}

    result = execute_runtime_node_command(str(command.get("action") or ""), dict(command.get("args") or {}))
    state["last_processed_command_id"] = command_id
    state["last_processed_command_at"] = _utc_now()
    save_runtime_node_state(state)
    ack_payload = _acknowledge_runtime_node_command(command, result, token=token)
    try:
        publish_runtime_node_heartbeat(token=token)
    except Exception:
        pass
    return {
        "ok": bool(result.get("ok", False)),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "command": ack_payload,
    }
