from __future__ import annotations

from pathlib import Path

from core.config import (
    AXIOMGRAPH_BRIDGE_BRANCH,
    AXIOMGRAPH_BRIDGE_LOOP_SECONDS,
    AXIOMGRAPH_BRIDGE_REPO,
    AXIOMGRAPH_BRIDGE_ROOT,
    AXIOMGRAPH_NODE_LABEL,
    AXIOMGRAPH_REMOTE_CONTROL_URL,
    KAGGLE_COMPETITION,
    KAGGLE_CONFIG_DIR,
    LLM_BACKEND,
    KAGGLE_POLL_INITIAL_SECONDS,
    KAGGLE_POLL_ATTEMPTS,
    KAGGLE_POLL_SECONDS,
    MAX_ARC_GEN_EXAMPLES,
    MAX_TEST_EXAMPLES,
    MAX_TRAIN_EXAMPLES,
    MODEL_OPTIONS,
    MODEL_REGISTRY,
    NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT,
    NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS,
    NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD,
    NEUROGOLF_AUTONOMY_LOOP_SECONDS,
    NEUROGOLF_AUTONOMY_MAX_HISTORY,
    NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS,
    NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS,
    NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE,
    NEUROGOLF_CURRENT_MANIFEST,
    NEUROGOLF_IMPORTED_SOURCES_DIR,
    NEUROGOLF_OPERATOR_NOTE_PATH,
    NEUROGOLF_OUTPUTS_DIR,
    NEUROGOLF_PROJECT_ROOT,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL_DIR,
    OPENCLAUDE_BIN,
    OPENCLAUDE_DEFAULT_MODEL,
    OPENCLAUDE_EFFORT,
    OPENCLAUDE_PROVIDER,
    REQUEST_TIMEOUT,
    SUBMISSION_MIN_DELTA,
    PROJECT_ROOT,
)

ENV_PATH = PROJECT_ROOT / ".env"


def _parse_env_lines(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def runtime_defaults() -> dict[str, str]:
    return {
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
        "OLLAMA_MODEL_DIR": OLLAMA_MODEL_DIR,
        "OLLAMA_TIMEOUT": str(REQUEST_TIMEOUT),
        "LLM_BACKEND": LLM_BACKEND,
        "OPENCLAUDE_BIN": OPENCLAUDE_BIN,
        "OPENCLAUDE_PROVIDER": OPENCLAUDE_PROVIDER,
        "OPENCLAUDE_DEFAULT_MODEL": OPENCLAUDE_DEFAULT_MODEL,
        "OPENCLAUDE_EFFORT": OPENCLAUDE_EFFORT,
        "KAGGLE_CONFIG_DIR": str(KAGGLE_CONFIG_DIR),
        "AXIOMGRAPH_REMOTE_CONTROL_URL": AXIOMGRAPH_REMOTE_CONTROL_URL,
        "AXIOMGRAPH_NODE_LABEL": AXIOMGRAPH_NODE_LABEL,
        "AXIOMGRAPH_BRIDGE_REPO": AXIOMGRAPH_BRIDGE_REPO,
        "AXIOMGRAPH_BRIDGE_BRANCH": AXIOMGRAPH_BRIDGE_BRANCH,
        "AXIOMGRAPH_BRIDGE_ROOT": AXIOMGRAPH_BRIDGE_ROOT,
        "AXIOMGRAPH_BRIDGE_LOOP_SECONDS": str(AXIOMGRAPH_BRIDGE_LOOP_SECONDS),
        "ARC_MODEL_ORCHESTRATOR": MODEL_REGISTRY["orchestrator"],
        "ARC_MODEL_OPERATOR_FAST": MODEL_REGISTRY["operator_fast"],
        "ARC_MODEL_REASONING_PRIMARY": MODEL_REGISTRY["reasoning_primary"],
        "ARC_MODEL_REASONING_SECONDARY": MODEL_REGISTRY["reasoning_secondary"],
        "ARC_MODEL_REASONING_TERTIARY": MODEL_REGISTRY["reasoning_tertiary"],
        "ARC_MODEL_CODER": MODEL_REGISTRY["coder"],
        "ARC_MODEL_CRITIC": MODEL_REGISTRY["critic"],
        "ARC_MAX_TRAIN_EXAMPLES": str(MAX_TRAIN_EXAMPLES),
        "ARC_MAX_TEST_EXAMPLES": str(MAX_TEST_EXAMPLES),
        "ARC_MAX_ARC_GEN_EXAMPLES": str(MAX_ARC_GEN_EXAMPLES),
        "ARC_CTX_ORCHESTRATOR": str(MODEL_OPTIONS["orchestrator"]["num_ctx"]),
        "ARC_CTX_OPERATOR_FAST": str(MODEL_OPTIONS["operator_fast"]["num_ctx"]),
        "ARC_CTX_REASONING_PRIMARY": str(MODEL_OPTIONS["reasoning_primary"]["num_ctx"]),
        "ARC_CTX_REASONING_SECONDARY": str(MODEL_OPTIONS["reasoning_secondary"]["num_ctx"]),
        "ARC_CTX_REASONING_TERTIARY": str(MODEL_OPTIONS["reasoning_tertiary"]["num_ctx"]),
        "ARC_CTX_CODER": str(MODEL_OPTIONS["coder"]["num_ctx"]),
        "ARC_CTX_CRITIC": str(MODEL_OPTIONS["critic"]["num_ctx"]),
        "ARC_PREDICT_ORCHESTRATOR": str(MODEL_OPTIONS["orchestrator"]["num_predict"]),
        "ARC_PREDICT_OPERATOR_FAST": str(MODEL_OPTIONS["operator_fast"]["num_predict"]),
        "ARC_PREDICT_REASONING_PRIMARY": str(MODEL_OPTIONS["reasoning_primary"]["num_predict"]),
        "ARC_PREDICT_REASONING_SECONDARY": str(MODEL_OPTIONS["reasoning_secondary"]["num_predict"]),
        "ARC_PREDICT_REASONING_TERTIARY": str(MODEL_OPTIONS["reasoning_tertiary"]["num_predict"]),
        "ARC_PREDICT_CODER": str(MODEL_OPTIONS["coder"]["num_predict"]),
        "ARC_PREDICT_CRITIC": str(MODEL_OPTIONS["critic"]["num_predict"]),
        "ARC_TEMP_ORCHESTRATOR": str(MODEL_OPTIONS["orchestrator"]["temperature"]),
        "ARC_TEMP_OPERATOR_FAST": str(MODEL_OPTIONS["operator_fast"]["temperature"]),
        "ARC_TEMP_REASONING_PRIMARY": str(MODEL_OPTIONS["reasoning_primary"]["temperature"]),
        "ARC_TEMP_REASONING_SECONDARY": str(MODEL_OPTIONS["reasoning_secondary"]["temperature"]),
        "ARC_TEMP_REASONING_TERTIARY": str(MODEL_OPTIONS["reasoning_tertiary"]["temperature"]),
        "ARC_TEMP_CODER": str(MODEL_OPTIONS["coder"]["temperature"]),
        "ARC_TEMP_CRITIC": str(MODEL_OPTIONS["critic"]["temperature"]),
        "ARC_KAGGLE_COMPETITION": KAGGLE_COMPETITION,
        "ARC_SUBMISSION_MIN_DELTA": str(SUBMISSION_MIN_DELTA),
        "ARC_KAGGLE_POLL_INITIAL_SECONDS": str(KAGGLE_POLL_INITIAL_SECONDS),
        "ARC_KAGGLE_POLL_SECONDS": str(KAGGLE_POLL_SECONDS),
        "ARC_KAGGLE_POLL_ATTEMPTS": str(KAGGLE_POLL_ATTEMPTS),
        "NEUROGOLF_PROJECT_ROOT": str(NEUROGOLF_PROJECT_ROOT),
        "NEUROGOLF_OUTPUTS_DIR": str(NEUROGOLF_OUTPUTS_DIR),
        "NEUROGOLF_IMPORTED_SOURCES_DIR": str(NEUROGOLF_IMPORTED_SOURCES_DIR),
        "NEUROGOLF_CURRENT_MANIFEST": str(NEUROGOLF_CURRENT_MANIFEST),
        "NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD": str(NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD),
        "NEUROGOLF_AUTONOMY_MAX_HISTORY": str(NEUROGOLF_AUTONOMY_MAX_HISTORY),
        "NEUROGOLF_AUTONOMY_LOOP_SECONDS": str(NEUROGOLF_AUTONOMY_LOOP_SECONDS),
        "NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS": str(NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS),
        "NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT": "1" if NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT else "0",
        "NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS": str(NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS),
        "NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE": str(NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE),
        "NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS": str(NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS),
        "NEUROGOLF_OPERATOR_NOTE_PATH": str(NEUROGOLF_OPERATOR_NOTE_PATH),
    }


def load_env_settings() -> dict[str, str]:
    defaults = runtime_defaults()
    if ENV_PATH.exists():
        defaults.update(_parse_env_lines(ENV_PATH.read_text(encoding="utf-8")))
    return defaults


def write_env_settings(values: dict[str, str]) -> Path:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(values.items())]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ENV_PATH


def update_env_settings(updates: dict[str, str]) -> Path:
    current = load_env_settings()
    current.update({key: str(value) for key, value in updates.items()})
    return write_env_settings(current)
