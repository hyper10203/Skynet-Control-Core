from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Always let the saved local control-center settings win over any stale shell
# variables from old sessions or background launchers.
load_dotenv(PROJECT_ROOT / ".env", override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


MODEL_REGISTRY = {
    "orchestrator": os.getenv("ARC_MODEL_ORCHESTRATOR", "qwen2.5"),
    "operator_fast": os.getenv("ARC_MODEL_OPERATOR_FAST", "qwen2.5"),
    "reasoning_primary": os.getenv("ARC_MODEL_REASONING_PRIMARY", "deepseek-r1"),
    "reasoning_secondary": os.getenv("ARC_MODEL_REASONING_SECONDARY", "mistral"),
    "reasoning_tertiary": os.getenv("ARC_MODEL_REASONING_TERTIARY", "llama3"),
    "coder": os.getenv("ARC_MODEL_CODER", "qwen2.5-coder"),
    "critic": os.getenv("ARC_MODEL_CRITIC", "llama3"),
}

PREFER_DIRECT_ORCHESTRATION = os.getenv("ARC_PREFER_DIRECT_ORCHESTRATION", "1").strip().lower() not in {"0", "false", "no"}

MODEL_OPTIONS = {
    "orchestrator": {
        "num_ctx": _env_int("ARC_CTX_ORCHESTRATOR", 8192),
        "temperature": _env_float("ARC_TEMP_ORCHESTRATOR", 0.2),
        "num_predict": _env_int("ARC_PREDICT_ORCHESTRATOR", 220),
    },
    "operator_fast": {
        "num_ctx": _env_int("ARC_CTX_OPERATOR_FAST", 6144),
        "temperature": _env_float("ARC_TEMP_OPERATOR_FAST", 0.15),
        "num_predict": _env_int("ARC_PREDICT_OPERATOR_FAST", 180),
    },
    "reasoning_primary": {
        "num_ctx": _env_int("ARC_CTX_REASONING_PRIMARY", 12288),
        "temperature": _env_float("ARC_TEMP_REASONING_PRIMARY", 0.3),
        "num_predict": _env_int("ARC_PREDICT_REASONING_PRIMARY", 350),
    },
    "reasoning_secondary": {
        "num_ctx": _env_int("ARC_CTX_REASONING_SECONDARY", 8192),
        "temperature": _env_float("ARC_TEMP_REASONING_SECONDARY", 0.5),
        "num_predict": _env_int("ARC_PREDICT_REASONING_SECONDARY", 260),
    },
    "reasoning_tertiary": {
        "num_ctx": _env_int("ARC_CTX_REASONING_TERTIARY", 8192),
        "temperature": _env_float("ARC_TEMP_REASONING_TERTIARY", 0.35),
        "num_predict": _env_int("ARC_PREDICT_REASONING_TERTIARY", 260),
    },
    "coder": {
        "num_ctx": _env_int("ARC_CTX_CODER", 8192),
        "temperature": _env_float("ARC_TEMP_CODER", 0.2),
        "num_predict": _env_int("ARC_PREDICT_CODER", 700),
    },
    "critic": {
        "num_ctx": _env_int("ARC_CTX_CRITIC", 6144),
        "temperature": _env_float("ARC_TEMP_CRITIC", 0.2),
        "num_predict": _env_int("ARC_PREDICT_CRITIC", 220),
    },
}

MAX_TRAIN_EXAMPLES = int(os.getenv("ARC_MAX_TRAIN_EXAMPLES", "2"))
MAX_TEST_EXAMPLES = int(os.getenv("ARC_MAX_TEST_EXAMPLES", "0"))
MAX_ARC_GEN_EXAMPLES = int(os.getenv("ARC_MAX_ARC_GEN_EXAMPLES", "0"))

MEMORY_PATH = PROJECT_ROOT / "memory" / "patterns.json"

DEFAULT_DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "neurogolf-2026",
    PROJECT_ROOT.parent / "data" / "neurogolf-2026",
    PROJECT_ROOT / "data" / "training",
]
DEFAULT_DATA_DIR = next((path for path in DEFAULT_DATA_DIR_CANDIDATES if path.exists()), DEFAULT_DATA_DIR_CANDIDATES[0])

KAGGLE_COMPETITION = os.getenv("ARC_KAGGLE_COMPETITION", "neurogolf-2026")
KAGGLE_SUBMISSION_FILE = Path(os.getenv("ARC_KAGGLE_SUBMISSION_FILE", str(PROJECT_ROOT / "outputs" / "submission.zip")))
SUBMISSION_STATE_PATH = Path(os.getenv("ARC_SUBMISSION_STATE_PATH", str(PROJECT_ROOT / "outputs" / "submission_state.json")))
SUBMISSION_MIN_DELTA = _env_float("ARC_SUBMISSION_MIN_DELTA", 250.0)
KAGGLE_POLL_INITIAL_SECONDS = _env_int("ARC_KAGGLE_POLL_INITIAL_SECONDS", 240)
KAGGLE_POLL_SECONDS = _env_int("ARC_KAGGLE_POLL_SECONDS", 60)
KAGGLE_POLL_ATTEMPTS = _env_int("ARC_KAGGLE_POLL_ATTEMPTS", 45)

NEUROGOLF_PROJECT_ROOT = _env_path(
    "NEUROGOLF_PROJECT_ROOT",
    Path(r"D:\E into D(12-15-2025_15-33)\coding 25-26\neurogolf_arc_system"),
)
NEUROGOLF_OUTPUTS_DIR = _env_path("NEUROGOLF_OUTPUTS_DIR", NEUROGOLF_PROJECT_ROOT / "outputs")
NEUROGOLF_IMPORTED_SOURCES_DIR = _env_path(
    "NEUROGOLF_IMPORTED_SOURCES_DIR",
    NEUROGOLF_PROJECT_ROOT / "scratch" / "gui_imports",
)
NEUROGOLF_CURRENT_MANIFEST = _env_path(
    "NEUROGOLF_CURRENT_MANIFEST",
    NEUROGOLF_OUTPUTS_DIR / "submission_manifest_current.json",
)
NEUROGOLF_PACKAGE_SCRIPT = _env_path(
    "NEUROGOLF_PACKAGE_SCRIPT",
    NEUROGOLF_PROJECT_ROOT / "package_submission.ps1",
)
NEUROGOLF_SYNC_STATE_PATH = _env_path(
    "NEUROGOLF_SYNC_STATE_PATH",
    PROJECT_ROOT / "memory" / "neurogolf_state.json",
)
NEUROGOLF_AUTONOMY_REPORTS_DIR = _env_path(
    "NEUROGOLF_AUTONOMY_REPORTS_DIR",
    PROJECT_ROOT / "outputs" / "neurogolf",
)
NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD = _env_float("NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD", 40.0)
NEUROGOLF_AUTONOMY_MAX_HISTORY = _env_int("NEUROGOLF_AUTONOMY_MAX_HISTORY", 10)
NEUROGOLF_AUTONOMY_LOOP_SECONDS = _env_int("NEUROGOLF_AUTONOMY_LOOP_SECONDS", 1800)
NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS = _env_int("NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS", 1)
NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT = os.getenv("NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT", "1").strip().lower() not in {"0", "false", "no"}
NEUROGOLF_AUTONOMY_RECENT_REPORT_LIMIT = _env_int("NEUROGOLF_AUTONOMY_RECENT_REPORT_LIMIT", 50)
NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS = _env_int("NEUROGOLF_AUTONOMY_BUILD_TIMEOUT_SECONDS", 1800)
# DISABLED: Target score and streak requirements removed to allow submissions
# Previously: NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE = 10000.0
# Previously: NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS = 2
# Now: Always allow submissions regardless of score or streak
NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE = _env_float("NEUROGOLF_AUTONOMY_TARGET_PUBLIC_SCORE", 0.0)  # 0 = disabled
NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS = _env_int("NEUROGOLF_AUTONOMY_TARGET_CONSECUTIVE_BESTS", 0)  # 0 = disabled
NEUROGOLF_AUTONOMY_LOG_PATH = _env_path(
    "NEUROGOLF_AUTONOMY_LOG_PATH",
    NEUROGOLF_AUTONOMY_REPORTS_DIR / "daemon.log",
)
NEUROGOLF_AUTONOMY_STATUS_PATH = _env_path(
    "NEUROGOLF_AUTONOMY_STATUS_PATH",
    NEUROGOLF_AUTONOMY_REPORTS_DIR / "status.json",
)
NEUROGOLF_AUTONOMY_PID_PATH = _env_path(
    "NEUROGOLF_AUTONOMY_PID_PATH",
    NEUROGOLF_AUTONOMY_REPORTS_DIR / "autonomous.pid",
)
NEUROGOLF_OPERATOR_NOTE_PATH = _env_path(
    "NEUROGOLF_OPERATOR_NOTE_PATH",
    PROJECT_ROOT / "memory" / "operator_note.txt",
)
NEUROGOLF_SEED_CONTROLS_PATH = _env_path(
    "NEUROGOLF_SEED_CONTROLS_PATH",
    PROJECT_ROOT / "memory" / "seed_controls.json",
)
SKYNET_ARCHIVE_DIR = _env_path(
    "SKYNET_ARCHIVE_DIR",
    PROJECT_ROOT / "archive" / "axiomgraph_maintenance",
)
SKYNET_DISTILLATION_PLAN_PATH = _env_path(
    "SKYNET_DISTILLATION_PLAN_PATH",
    PROJECT_ROOT / "memory" / "distillation_plan.json",
)
