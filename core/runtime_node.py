from __future__ import annotations

import json
import os
import secrets
import socket
from datetime import datetime, timezone
from pathlib import Path

from core.config import (
    AXIOMGRAPH_NODE_LABEL,
    AXIOMGRAPH_REMOTE_CONTROL_URL,
    AXIOMGRAPH_RUNTIME_NODE_STATE_PATH,
    KAGGLE_CONFIG_DIR,
    NEUROGOLF_PROJECT_ROOT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_DIR,
    PROJECT_ROOT,
)
from core.env_settings import load_env_settings, update_env_settings


def _default_label() -> str:
    return (
        os.getenv("COMPUTERNAME", "").strip()
        or socket.gethostname().strip()
        or AXIOMGRAPH_NODE_LABEL
        or "AxiomGraph Runtime Node"
    )


def _default_state() -> dict[str, str]:
    env_settings = load_env_settings()
    return {
        "node_id": secrets.token_hex(8),
        "pair_token": secrets.token_urlsafe(24),
        "device_label": str(env_settings.get("AXIOMGRAPH_NODE_LABEL") or _default_label()).strip(),
        "remote_control_url": str(
            env_settings.get("AXIOMGRAPH_REMOTE_CONTROL_URL") or AXIOMGRAPH_REMOTE_CONTROL_URL
        ).strip(),
        "registered_operator_email": "",
        "paired_at": "",
    }


def _pairing_code(node_id: str, pair_token: str) -> str:
    left = str(node_id or "").replace("-", "").upper()[:6]
    right = str(pair_token or "").replace("-", "").replace("_", "").upper()[:6]
    return f"{left}-{right}"


def _resolve_local_path(raw: str | Path, fallback: Path) -> Path:
    value = Path(str(raw).strip()) if str(raw).strip() else fallback
    if not value.is_absolute():
        value = (PROJECT_ROOT / value).resolve()
    return value


def load_runtime_node_state() -> dict[str, str]:
    state = _default_state()
    path = AXIOMGRAPH_RUNTIME_NODE_STATE_PATH
    dirty = False
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                state.update({key: str(value) for key, value in payload.items() if value is not None})
        except Exception:
            dirty = True
    else:
        dirty = True
    if not state.get("node_id"):
        state["node_id"] = secrets.token_hex(8)
        dirty = True
    if not state.get("pair_token"):
        state["pair_token"] = secrets.token_urlsafe(24)
        dirty = True
    if not state.get("device_label"):
        state["device_label"] = _default_label()
        dirty = True
    if not state.get("remote_control_url"):
        state["remote_control_url"] = AXIOMGRAPH_REMOTE_CONTROL_URL
        dirty = True
    if dirty:
        save_runtime_node_state(state)
    return state


def save_runtime_node_state(state: dict[str, str]) -> Path:
    payload = _default_state()
    if AXIOMGRAPH_RUNTIME_NODE_STATE_PATH.exists():
        try:
            current = json.loads(AXIOMGRAPH_RUNTIME_NODE_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(current, dict):
                payload.update({key: str(value) for key, value in current.items() if value is not None})
        except Exception:
            pass
    payload.update({key: str(value) for key, value in state.items() if value is not None})
    AXIOMGRAPH_RUNTIME_NODE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AXIOMGRAPH_RUNTIME_NODE_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return AXIOMGRAPH_RUNTIME_NODE_STATE_PATH


def rotate_pair_token() -> dict[str, str]:
    state = load_runtime_node_state()
    state["pair_token"] = secrets.token_urlsafe(24)
    save_runtime_node_state(state)
    return state


def save_runtime_node_settings(
    *,
    device_label: str,
    remote_control_url: str,
    workspace_root: str,
    ollama_base_url: str,
    ollama_model_dir: str,
    kaggle_config_dir: str,
) -> dict[str, str]:
    update_env_settings(
        {
            "AXIOMGRAPH_NODE_LABEL": str(device_label).strip(),
            "AXIOMGRAPH_REMOTE_CONTROL_URL": str(remote_control_url).strip(),
            "NEUROGOLF_PROJECT_ROOT": str(workspace_root).strip(),
            "OLLAMA_BASE_URL": str(ollama_base_url).strip(),
            "OLLAMA_MODEL_DIR": str(ollama_model_dir).strip(),
            "KAGGLE_CONFIG_DIR": str(kaggle_config_dir).strip(),
        }
    )
    save_runtime_node_state(
        {
            "device_label": str(device_label).strip(),
            "remote_control_url": str(remote_control_url).strip(),
        }
    )
    return load_runtime_node_summary()


def save_kaggle_credentials(*, username: str, key: str, config_dir: str) -> dict[str, str]:
    target_dir = _resolve_local_path(config_dir, KAGGLE_CONFIG_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = target_dir / "kaggle.json"
    payload = {"username": str(username).strip(), "key": str(key).strip()}
    kaggle_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_env_settings({"KAGGLE_CONFIG_DIR": str(target_dir)})
    return {
        "config_dir": str(target_dir),
        "kaggle_json_path": str(kaggle_json_path),
        "username": payload["username"],
        "saved": "1",
    }


def register_paired_operator(email: str) -> dict[str, str]:
    state = load_runtime_node_state()
    state["registered_operator_email"] = str(email).strip()
    if state["registered_operator_email"]:
        state["paired_at"] = state.get("paired_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    else:
        state["paired_at"] = ""
    save_runtime_node_state(state)
    return state


def load_runtime_node_summary() -> dict:
    env_settings = load_env_settings()
    state = load_runtime_node_state()
    kaggle_dir = _resolve_local_path(env_settings.get("KAGGLE_CONFIG_DIR", ""), KAGGLE_CONFIG_DIR)
    kaggle_json_path = kaggle_dir / "kaggle.json"
    kaggle_username = ""
    if kaggle_json_path.exists():
        try:
            kaggle_username = json.loads(kaggle_json_path.read_text(encoding="utf-8")).get("username", "")
        except Exception:
            kaggle_username = ""

    workspace_root_path = _resolve_local_path(
        env_settings.get("NEUROGOLF_PROJECT_ROOT", ""),
        NEUROGOLF_PROJECT_ROOT,
    )
    workspace_root = str(workspace_root_path)
    ollama_base_url = str(env_settings.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).strip()
    raw_ollama_model_dir = str(env_settings.get("OLLAMA_MODEL_DIR") or OLLAMA_MODEL_DIR).strip()
    ollama_model_dir_path = _resolve_local_path(raw_ollama_model_dir, PROJECT_ROOT / ".local" / "models") if raw_ollama_model_dir else None
    ollama_model_dir = str(ollama_model_dir_path) if ollama_model_dir_path else ""
    remote_control_url = str(
        env_settings.get("AXIOMGRAPH_REMOTE_CONTROL_URL")
        or state.get("remote_control_url")
        or AXIOMGRAPH_REMOTE_CONTROL_URL
    ).strip()
    device_label = str(
        env_settings.get("AXIOMGRAPH_NODE_LABEL")
        or state.get("device_label")
        or _default_label()
    ).strip()

    return {
        "node_id": state.get("node_id", ""),
        "device_label": device_label,
        "remote_control_url": remote_control_url,
        "pair_token": state.get("pair_token", ""),
        "pairing_code": _pairing_code(state.get("node_id", ""), state.get("pair_token", "")),
        "registered_operator_email": state.get("registered_operator_email", ""),
        "paired_at": state.get("paired_at", ""),
        "workspace_root": workspace_root,
        "workspace_exists": workspace_root_path.exists(),
        "ollama_base_url": ollama_base_url,
        "ollama_model_dir": ollama_model_dir,
        "ollama_model_dir_exists": bool(ollama_model_dir_path) and bool(ollama_model_dir_path.exists()),
        "kaggle_config_dir": str(kaggle_dir),
        "kaggle_json_path": str(kaggle_json_path),
        "kaggle_json_exists": kaggle_json_path.exists(),
        "kaggle_username": kaggle_username,
        "state_path": str(AXIOMGRAPH_RUNTIME_NODE_STATE_PATH),
    }
