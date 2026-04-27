from __future__ import annotations

import hmac
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

from core.arc_loader import load_tasks
from core.config import DEFAULT_DATA_DIR
from core.env_settings import load_env_settings, update_env_settings
from core.executor import ask
from core.model_profiles import (
    MODEL_PROFILES,
    MODEL_SOURCES,
    ROLE_LABELS,
    default_context_for_model,
    env_updates_for_profile,
)
from core.prompts import FAST_OPERATOR_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from core.runtime_control import (
    current_role_map,
    daemon_status,
    fetch_kaggle_targets,
    healthcheck,
    latest_team_name,
    leaderboard_rank_for_team,
    list_imported_kaggle_sources,
    ollama_models,
    operator_note,
    persist_runtime_preferences,
    read_state,
    read_status,
    recent_reports,
    restart_daemon_async,
    remote_access_endpoints,
    run_single_cycle,
    save_operator_note,
    start_daemon,
    stop_daemon,
    sync_state,
    tail_log,
)


st.set_page_config(page_title="Skynet Control Core", page_icon="S", layout="wide")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Share+Tech+Mono&display=swap');

:root {
    --bg-0: #02060d;
    --bg-1: #07121b;
    --bg-2: #091d28;
    --line: rgba(127, 255, 238, 0.18);
    --line-hot: rgba(255, 73, 132, 0.28);
    --ink: #e9fdff;
    --sub: #8fb8c3;
    --aqua: #7fffee;
    --cyan: #20beff;
    --pink: #ff4984;
}

.stApp {
    background:
        radial-gradient(circle at 10% 18%, rgba(0, 255, 209, 0.12), transparent 23%),
        radial-gradient(circle at 85% 15%, rgba(255, 73, 132, 0.10), transparent 24%),
        linear-gradient(150deg, var(--bg-0) 0%, var(--bg-1) 38%, var(--bg-2) 64%, var(--bg-0) 100%);
    color: var(--ink);
}

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
}

h1, h2, h3, .stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.04em;
}

.top-shell {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(127, 255, 238, 0.22);
    border-radius: 28px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    background:
        linear-gradient(135deg, rgba(127,255,238,0.07), rgba(32,190,255,0.06) 55%, rgba(255,73,132,0.08)),
        rgba(4, 10, 17, 0.88);
    box-shadow: 0 0 34px rgba(0, 255, 209, 0.12);
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 1rem;
    align-items: center;
}

.hero-kicker {
    color: var(--aqua);
    font-size: 0.82rem;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.hero-title {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--ink);
}

.hero-copy {
    margin-top: 0.55rem;
    color: #c0d8df;
    line-height: 1.55;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 0.95rem;
}

.badge-chip {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.36rem 0.78rem;
    background: rgba(6, 14, 22, 0.82);
    color: var(--aqua);
    font-size: 0.8rem;
}

.asset-rail {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 0.9rem;
}

.asset-rail img {
    width: 58px;
    height: 58px;
    opacity: 0.95;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem 1rem 0.95rem 1rem;
    background: rgba(5, 11, 18, 0.76);
    box-shadow: 0 0 18px rgba(0, 255, 209, 0.07);
}

.panel h4 {
    margin: 0 0 0.55rem 0;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.04em;
    color: var(--ink);
}

.panel-copy {
    color: var(--sub);
    line-height: 1.55;
    margin-bottom: 0.35rem;
}

.status-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.85rem 0 1rem 0;
}

.status-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.85rem 0.9rem;
    background: linear-gradient(180deg, rgba(10,20,30,0.95), rgba(4,10,17,0.95));
}

.status-label {
    color: var(--sub);
    font-size: 0.78rem;
    text-transform: uppercase;
}

.status-value {
    margin-top: 0.3rem;
    color: var(--ink);
    font-size: 1.35rem;
    font-family: 'Orbitron', sans-serif;
}

.status-sub {
    margin-top: 0.2rem;
    color: var(--aqua);
    font-size: 0.78rem;
}

[data-testid="stMetric"] {
    background: rgba(5, 12, 19, 0.74);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.72rem;
}

div.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(127, 255, 238, 0.22);
    background: linear-gradient(135deg, rgba(127,255,238,0.16), rgba(32,190,255,0.18));
    color: var(--ink);
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--aqua), var(--cyan) 50%, var(--pink) 100%);
}

div[data-baseweb="select"] > div,
.stTextArea textarea,
.stTextInput input,
.stNumberInput input {
    background: rgba(4, 11, 18, 0.82) !important;
    color: var(--ink) !important;
    border-radius: 14px !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(5, 12, 19, 0.72);
    border: 1px solid var(--line);
    border-radius: 14px;
    color: #d8f7ff;
    padding: 0.5rem 0.9rem;
}

.auth-shell {
    max-width: 760px;
    margin: 4vh auto 0 auto;
    border: 1px solid rgba(127, 255, 238, 0.22);
    border-radius: 28px;
    padding: 1.6rem;
    background:
        linear-gradient(135deg, rgba(127,255,238,0.08), rgba(32,190,255,0.05) 55%, rgba(255,73,132,0.08)),
        rgba(4, 10, 17, 0.9);
    box-shadow: 0 0 34px rgba(0, 255, 209, 0.12);
}

.auth-title {
    margin: 0.3rem 0 0.9rem 0;
    font-size: 2rem;
    font-weight: 800;
    color: var(--ink);
}

.auth-copy {
    color: #c0d8df;
    line-height: 1.6;
    margin-bottom: 1rem;
}

@media (max-width: 1100px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .asset-rail {
        justify-content: flex-start;
    }
    .status-card-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .top-shell {
        padding: 1rem;
        border-radius: 22px;
    }
    .hero-title {
        font-size: 1.6rem;
    }
    .hero-copy {
        font-size: 0.95rem;
    }
    .asset-rail img {
        width: 44px;
        height: 44px;
    }
    .status-card-grid {
        grid-template-columns: 1fr;
    }
    .badge-chip {
        font-size: 0.72rem;
        padding: 0.3rem 0.6rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        padding: 0.45rem 0.6rem;
    }
    .auth-shell {
        margin-top: 2vh;
        padding: 1rem;
    }
    .auth-title {
        font-size: 1.45rem;
    }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
REFRESH_EVERY_SECONDS = 300
ROLE_ENV_KEYS = {role: f"ARC_MODEL_{role.upper()}" for role in ROLE_LABELS}
CTX_ENV_KEYS = {role: f"ARC_CTX_{role.upper()}" for role in ROLE_LABELS}
AUTH_STATE_KEY = "skynet_ui_authenticated"
AUTH_INPUT_KEY = "skynet_ui_password_input"


def fmt_value(value, *, digits: int = 2, fallback: str = "n/a") -> str:
    if value in (None, ""):
        return fallback
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def current_completed_submission(submissions: list[dict]) -> dict:
    for item in submissions:
        if item.get("public_score") is not None:
            return item
    return {}


def completed_scores(submissions: list[dict]) -> list[float]:
    scores: list[float] = []
    for item in reversed(submissions):
        score = item.get("public_score")
        if score is None:
            continue
        try:
            scores.append(float(score))
        except Exception:
            continue
    return scores


def configured_ui_password(env_settings: dict[str, str]) -> str:
    return str(env_settings.get("SKYNET_UI_PASSWORD", "")).strip()


def ui_requires_password(env_settings: dict[str, str]) -> bool:
    return bool(configured_ui_password(env_settings))


def is_authenticated(env_settings: dict[str, str]) -> bool:
    if not ui_requires_password(env_settings):
        return True
    return bool(st.session_state.get(AUTH_STATE_KEY, False))


def render_auth_gate(env_settings: dict[str, str]) -> None:
    if not ui_requires_password(env_settings):
        st.session_state[AUTH_STATE_KEY] = True
        return
    if st.session_state.get(AUTH_STATE_KEY):
        return

    st.markdown(
        """
        <div class="auth-shell">
          <div class="hero-kicker">Secure Access Layer</div>
          <h1 class="auth-title">Skynet Control Core</h1>
          <div class="auth-copy">
            This deployment is locked. Enter the control password to access the dashboard, daemon controls,
            ARC visualizer, and NeuroGolf automation surfaces.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_mid, col_right = st.columns([1, 1.3, 1])
    with col_mid:
        password = st.text_input("Control password", type="password", key=AUTH_INPUT_KEY)
        if st.button("Unlock Time Core", width="stretch"):
            expected = configured_ui_password(env_settings)
            if hmac.compare_digest(password, expected):
                st.session_state[AUTH_STATE_KEY] = True
                st.session_state.pop(AUTH_INPUT_KEY, None)
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")


def clamp_progress(value) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))


@st.cache_data(ttl=300)
def cached_ollama_models() -> list[dict]:
    return ollama_models()


@st.cache_data(ttl=300)
def cached_rank(team_name: str) -> dict:
    if not team_name:
        return {}
    return leaderboard_rank_for_team(team_name)


@st.cache_data(ttl=300)
def cached_tasks(data_dir: str) -> list[dict]:
    return load_tasks(data_dir)


def set_action_result(label: str, result: dict) -> None:
    st.session_state["last_action"] = {"label": label, "result": result}


def show_action_result() -> None:
    action = st.session_state.get("last_action")
    if not action:
        return
    result = action["result"]
    body = "\n".join(
        part for part in [result.get("stdout", "").strip(), result.get("stderr", "").strip()] if part
    )
    if result.get("ok", True):
        st.success(action["label"])
    else:
        st.error(action["label"])
    if body:
        st.code(body, language="text")


def role_widget_key(role: str) -> str:
    return f"role_model_{role}"


def role_context_key(role: str) -> str:
    return f"role_ctx_{role}"


def control_widget_key(name: str) -> str:
    return f"control_{name}"


def seed_role_editor(env_settings: dict[str, str]) -> None:
    for role in ROLE_LABELS:
        model_key = role_widget_key(role)
        ctx_key = role_context_key(role)
        default_model = env_settings.get(ROLE_ENV_KEYS[role], "")
        if model_key not in st.session_state:
            st.session_state[model_key] = default_model
        if ctx_key not in st.session_state:
            raw_ctx = env_settings.get(CTX_ENV_KEYS[role], "")
            st.session_state[ctx_key] = int(raw_ctx) if str(raw_ctx).strip() else default_context_for_model(st.session_state[model_key], role)


def seed_control_editor(env_settings: dict[str, str]) -> None:
    defaults = {
        "allow_submit": str(env_settings.get("NEUROGOLF_AUTONOMY_ALLOW_SUBMIT_DEFAULT", "1")).strip().lower() not in {"0", "false", "no"},
        "history": int(env_settings.get("NEUROGOLF_AUTONOMY_MAX_HISTORY", "10")),
        "min_local_delta": float(env_settings.get("NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD", "40")),
        "sleep_seconds": int(env_settings.get("NEUROGOLF_AUTONOMY_LOOP_SECONDS", "1800")),
        "max_pending_submissions": int(env_settings.get("NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS", "1")),
    }
    for name, value in defaults.items():
        key = control_widget_key(name)
        if key not in st.session_state:
            st.session_state[key] = value


def current_control_settings() -> dict[str, int | float | bool]:
    return {
        "allow_submit": bool(st.session_state.get(control_widget_key("allow_submit"), True)),
        "history": int(st.session_state.get(control_widget_key("history"), 10)),
        "min_local_delta": float(st.session_state.get(control_widget_key("min_local_delta"), 40.0)),
        "sleep_seconds": int(st.session_state.get(control_widget_key("sleep_seconds"), 1800)),
        "max_pending_submissions": int(st.session_state.get(control_widget_key("max_pending_submissions"), 1)),
    }


def apply_pending_context_resets() -> None:
    pending_roles = st.session_state.pop("pending_context_resets", [])
    if not pending_roles:
        return
    for role in pending_roles:
        model_name = st.session_state.get(role_widget_key(role), "")
        st.session_state[role_context_key(role)] = default_context_for_model(model_name, role)


def sync_context_from_role(role: str) -> None:
    model_name = st.session_state.get(role_widget_key(role), "")
    st.session_state[role_context_key(role)] = default_context_for_model(model_name, role)


def load_profile_into_editor(profile_key: str) -> None:
    profile = MODEL_PROFILES[profile_key]
    for role, model_name in profile["roles"].items():
        st.session_state[role_widget_key(role)] = model_name
        st.session_state[role_context_key(role)] = int(profile["contexts"][role])


def detect_pattern(input_grid: list[list[int]], output_grid: list[list[int]]) -> str:
    input_array = np.array(input_grid)
    output_array = np.array(output_grid)

    if input_array.shape == output_array.shape:
        if np.array_equal(np.rot90(input_array), output_array):
            return "Rotation 90"
        if np.array_equal(np.rot90(input_array, 2), output_array):
            return "Rotation 180"
        if np.array_equal(np.fliplr(input_array), output_array):
            return "Horizontal flip"
        if np.array_equal(np.flipud(input_array), output_array):
            return "Vertical flip"
        if np.array_equal(input_array.T, output_array):
            return "Transpose"

    if input_array.shape != output_array.shape:
        return "Resize or crop"

    if set(np.unique(input_array)) != set(np.unique(output_array)):
        return "Color remap"

    return "Unknown"


def plot_grid(grid: list[list[int]], title: str):
    figure, axis = plt.subplots(figsize=(3.1, 3.1))
    axis.imshow(np.array(grid), cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    return figure


def render_arc_visualizer() -> None:
    st.markdown('<div class="panel"><h4>ARC Recon Chamber</h4><div class="panel-copy">Inspect train and test grids with actual ARC colors and lightweight pattern hints.</div></div>', unsafe_allow_html=True)
    default_path = str(DEFAULT_DATA_DIR)
    dataset_path = st.text_input("Dataset path", value=default_path)

    try:
        tasks = cached_tasks(dataset_path)
    except Exception as exc:
        st.error(f"Unable to load ARC tasks: {exc}")
        return

    if not tasks:
        st.warning("No ARC tasks found in that folder.")
        return

    task_ids = [task["id"] for task in tasks]
    selected_task_id = st.selectbox("Select task", task_ids)
    task = next(task for task in tasks if task["id"] == selected_task_id)
    payload = task["payload"]

    st.caption(f"Loaded from `{task['path']}`")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Train pairs", len(payload.get("train", [])))
    overview_cols[1].metric("Test pairs", len(payload.get("test", [])))
    overview_cols[2].metric("Arc-gen pairs", len(payload.get("arc-gen", [])))
    overview_cols[3].metric("Task id", task["id"].replace("task", ""))

    st.subheader("Training examples")
    for index, example in enumerate(payload.get("train", [])):
        info_col, input_col, output_col = st.columns([0.9, 1, 1])
        with info_col:
            input_shape = np.array(example["input"]).shape
            output_shape = np.array(example["output"]).shape
            colors = sorted(set(np.array(example["input"]).flatten()).union(set(np.array(example["output"]).flatten())))
            st.markdown(f"**Example {index + 1}**")
            st.caption(f"Input {input_shape} -> Output {output_shape}")
            st.caption(f"Colors: {colors}")
            st.info(detect_pattern(example["input"], example["output"]))
        with input_col:
            st.pyplot(plot_grid(example["input"], f"Input {index + 1}"))
        with output_col:
            st.pyplot(plot_grid(example["output"], f"Output {index + 1}"))

    st.subheader("Test inputs")
    for index, example in enumerate(payload.get("test", [])):
        test_col, shape_col = st.columns([1, 0.8])
        with test_col:
            st.pyplot(plot_grid(example["input"], f"Test input {index + 1}"))
        with shape_col:
            st.caption(f"Shape: {np.array(example['input']).shape}")
            st.caption(f"Colors: {sorted(set(np.array(example['input']).flatten()))}")


def build_orchestrator_chat_context(
    state: dict,
    status: dict,
    reports: list[dict],
    imported_sources: list[dict],
    role_map: list[dict],
) -> str:
    payload = {
        "team_name": state.get("current_team_name"),
        "best_completed_public_score": state.get("best_completed_public_score"),
        "pending_submission_count": state.get("pending_submission_count"),
        "latest_status": {
            "phase": status.get("phase"),
            "message": status.get("message"),
            "progress": status.get("progress"),
        },
        "recent_submissions": state.get("recent_submissions", [])[:4],
        "recent_output_manifests": state.get("recent_output_manifests", [])[:4],
        "imported_sources": imported_sources[:8],
        "roles": role_map,
        "recent_reports": [
            {
                "started_at": item.get("started_at"),
                "plan": item.get("plan"),
                "submit_result": item.get("submit_result"),
            }
            for item in reports[:3]
        ],
        "operator_note": operator_note(),
    }
    return str(payload)


def ensure_chat_state() -> None:
    if "orchestrator_chat_messages" not in st.session_state:
        st.session_state["orchestrator_chat_messages"] = [
            {
                "role": "assistant",
                "content": "I am the control center chat. Use Fast Operator Link for quick help, or escalate to the Orchestrator for deeper strategy.",
            }
        ]


def query_control_chat(target_role: str, prompt: str, context: str, env_settings: dict[str, str]) -> str:
    use_fast_lane = target_role == "operator_fast"
    model_name = env_settings.get(
        "ARC_MODEL_OPERATOR_FAST" if use_fast_lane else "ARC_MODEL_ORCHESTRATOR",
        "qwen2.5",
    )
    system_prompt = FAST_OPERATOR_SYSTEM_PROMPT if use_fast_lane else ORCHESTRATOR_SYSTEM_PROMPT
    context_key = "ARC_CTX_OPERATOR_FAST" if use_fast_lane else "ARC_CTX_ORCHESTRATOR"
    temp_key = "ARC_TEMP_OPERATOR_FAST" if use_fast_lane else "ARC_TEMP_ORCHESTRATOR"
    predict_key = "ARC_PREDICT_OPERATOR_FAST" if use_fast_lane else "ARC_PREDICT_ORCHESTRATOR"
    response = ask(
        model_name,
        f"""You are the {'fast operator-side helper' if use_fast_lane else 'direct front-door orchestrator chat'} for this machine.

Live workspace context:
{context}

User message:
{prompt}

Respond like an operational lead. Be concrete, aware of the current machine state, and optimize for better ARC and NeuroGolf outcomes.
""",
        system=system_prompt,
        options={
            "num_ctx": int(env_settings.get(context_key, "8192")),
            "temperature": float(env_settings.get(temp_key, "0.2")),
            "num_predict": int(env_settings.get(predict_key, "220")),
        },
        timeout=180,
    )
    return response


def model_options_from_installed(installed_models: list[dict], env_settings: dict[str, str]) -> list[str]:
    options = {row["name"] for row in installed_models if row.get("name")}
    for role in ROLE_LABELS:
        saved = env_settings.get(ROLE_ENV_KEYS[role], "").strip()
        if saved:
            options.add(saved)
    for profile in MODEL_PROFILES.values():
        for model_name in profile["roles"].values():
            options.add(model_name)
    return sorted(options)


def installed_model_name_set(installed_models: list[dict]) -> set[str]:
    return {str(row.get("name") or "").strip().lower() for row in installed_models if row.get("name")}


def has_installed_model(installed_names: set[str], desired_name: str) -> bool:
    desired = str(desired_name or "").strip().lower()
    if not desired:
        return False
    if desired in installed_names:
        return True

    desired_base = desired.split(":", 1)[0]
    for candidate in installed_names:
        candidate_base = candidate.split(":", 1)[0]
        if candidate == desired or candidate_base == desired_base or desired_base == candidate:
            return True
    return False


def best_profile_for_installed_models(installed_models: list[dict]) -> tuple[str, dict]:
    installed_names = installed_model_name_set(installed_models)
    if not installed_names:
        return "current-safe", {"matched": [], "missing": ["No Ollama models detected."]}

    candidates = [
        "your-local-max",
        "agentic-max",
        "coding-lab",
        "balanced-2026",
        "current-safe",
    ]

    best_key = "current-safe"
    best_score = -1
    best_meta = {"matched": [], "missing": []}

    for profile_key in candidates:
        profile = MODEL_PROFILES[profile_key]
        role_models = profile["roles"]
        matched = []
        missing = []
        for role, model_name in role_models.items():
            if has_installed_model(installed_names, model_name):
                matched.append({"role": role, "model": model_name})
            else:
                missing.append({"role": role, "model": model_name})

        score = len(matched) * 100 - len(missing)
        if not missing:
            score += 1000
        if score > best_score:
            best_key = profile_key
            best_score = score
            best_meta = {"matched": matched, "missing": missing}

    return best_key, best_meta


@st.fragment(run_every=REFRESH_EVERY_SECONDS)
def render_sidebar_fragment() -> None:
    env_settings = load_env_settings()
    state = read_state()
    daemon = daemon_status()
    endpoints = remote_access_endpoints()
    submissions = state.get("recent_submissions", [])
    team_name = str(state.get("current_team_name") or latest_team_name()).strip()
    rank_info = cached_rank(team_name)
    latest_completed = current_completed_submission(submissions)

    st.title("Resistance Console")
    st.caption("Auto refresh cadence: every 5 minutes")
    if st.button("Refresh Intel", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.metric("Daemon", "Running" if daemon.get("running") else "Stopped")
    st.metric("Rank", fmt_value(rank_info.get("rank"), digits=0))
    st.metric("Latest Score", fmt_value(latest_completed.get("public_score")))
    st.metric("Best Public", fmt_value(state.get("best_completed_public_score")))
    st.metric("Pending Subs", fmt_value(state.get("pending_submission_count"), digits=0))
    if team_name:
        st.caption(f"Team: {team_name}")
    st.caption(f"Remote: {endpoints.get('recommended_remote', 'n/a')}")
    if ui_requires_password(env_settings) and st.button("Lock Time Core", width="stretch"):
        st.session_state[AUTH_STATE_KEY] = False
        st.rerun()


@st.fragment(run_every=REFRESH_EVERY_SECONDS)
def render_dashboard() -> None:
    state = read_state()
    status = read_status()
    daemon = daemon_status()
    reports = recent_reports(limit=12)
    env_settings = load_env_settings()
    seed_role_editor(env_settings)
    seed_control_editor(env_settings)
    apply_pending_context_resets()
    models = cached_ollama_models()
    model_options = model_options_from_installed(models, env_settings)
    submissions = state.get("recent_submissions", [])
    team_name = str(state.get("current_team_name") or latest_team_name()).strip()
    rank_info = cached_rank(team_name)
    latest_completed = current_completed_submission(submissions)
    latest_submission = submissions[0] if submissions else {}
    progress = clamp_progress(status.get("progress", 0.0))
    scores = completed_scores(submissions)
    latest_report = reports[0] if reports else {}
    latest_plan = latest_report.get("plan") or {}
    latest_submit = latest_report.get("submit_result") or {}
    imported_sources = list_imported_kaggle_sources(limit=15)
    endpoints = remote_access_endpoints()

    st.markdown(
        f"""
        <div class="top-shell">
          <div class="hero-grid">
            <div>
              <div class="hero-kicker">Cyberdyne Operations Deck</div>
              <h1 class="hero-title">Skynet Control Core</h1>
              <div class="hero-copy">
                A live control surface for your ARC and NeuroGolf machine with daemon control, Kaggle harvesting,
                model-role management, submission telemetry, and an ARC task recon chamber.
              </div>
              <div class="badge-row">
                <div class="badge-chip">Daemon: {"ONLINE" if daemon.get("running") else "OFFLINE"}</div>
                <div class="badge-chip">Rank: {fmt_value(rank_info.get("rank"), digits=0)}</div>
                <div class="badge-chip">Latest: {fmt_value(latest_completed.get("public_score"))}</div>
                <div class="badge-chip">Best: {fmt_value(state.get("best_completed_public_score"))}</div>
                <div class="badge-chip">Working On: {status.get("phase", "idle")}</div>
              </div>
            </div>
            <div class="asset-rail">
              <img src="https://cdn.simpleicons.org/ollama/7FFFEF" alt="Ollama" />
              <img src="https://cdn.simpleicons.org/kaggle/20BEFF" alt="Kaggle" />
              <img src="https://cdn.simpleicons.org/python/FFD166" alt="Python" />
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_action_result()

    st.markdown(
        f"""
        <div class="status-card-grid">
          <div class="status-card">
            <div class="status-label">Current Phase</div>
            <div class="status-value">{status.get("phase", "idle")}</div>
            <div class="status-sub">{status.get("message", "No live status yet.")}</div>
          </div>
          <div class="status-card">
            <div class="status-label">Current Rank</div>
            <div class="status-value">{fmt_value(rank_info.get("rank"), digits=0)}</div>
            <div class="status-sub">Team: {team_name or "n/a"}</div>
          </div>
          <div class="status-card">
            <div class="status-label">Latest Completed</div>
            <div class="status-value">{fmt_value(latest_completed.get("public_score"))}</div>
            <div class="status-sub">{latest_completed.get("message", "Awaiting completed score")}</div>
          </div>
          <div class="status-card">
            <div class="status-label">Daemon Telemetry</div>
            <div class="status-value">{fmt_value(daemon.get("pid"), digits=0)}</div>
            <div class="status-sub">CPU {fmt_value(daemon.get("cpu_percent"))}% | RAM {fmt_value(daemon.get("memory_mb"))} MB</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(progress)
    st.caption(f"Mission completion signal: {progress * 100:.0f}%")

    tab_overview, tab_control, tab_models, tab_neurogolf, tab_reports, tab_arc, tab_chat = st.tabs(
        ["Overview", "Control", "Models", "NeuroGolf", "Reports", "ARC Visualizer", "Orchestrator Chat"]
    )

    with tab_overview:
        left_col, right_col = st.columns([1.02, 0.98])
        with left_col:
            st.markdown('<div class="panel"><h4>Mission Telemetry</h4><div class="panel-copy">Live state from the daemon, planner, and Kaggle memory.</div></div>', unsafe_allow_html=True)
            st.json(
                {
                    "daemon": daemon,
                    "status": status,
                    "rank": rank_info,
                    "latest_submission": latest_submission,
                    "latest_completed": latest_completed,
                }
            )
        with right_col:
            st.markdown('<div class="panel"><h4>What It Is Working On</h4><div class="panel-copy">Current mission lane, chosen seed family, and latest autonomous decision point.</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Phase:** `{status.get('phase', 'idle')}`")
            st.markdown(f"**Message:** {status.get('message', 'No active mission text yet.')}")
            st.markdown(f"**Latest planned seed:** `{latest_plan.get('seed_label', 'n/a')}`")
            st.markdown(f"**Latest plan mode:** `{latest_plan.get('mode', 'n/a')}`")
            st.markdown(f"**Latest submit decision:** `{latest_submit.get('reason', 'n/a')}`")
            st.subheader("Signal Feed")
            st.code(tail_log(80) or "No daemon log yet.", language="text")

        trend_col, note_col = st.columns([1.1, 0.9])
        with trend_col:
            st.markdown('<div class="panel"><h4>Score Trajectory</h4><div class="panel-copy">Completed public scores seen by the autonomous system so far.</div></div>', unsafe_allow_html=True)
            if scores:
                st.line_chart(scores, height=240)
            else:
                st.info("No completed public scores have been synced yet.")
        with note_col:
            st.markdown('<div class="panel"><h4>Mission Flags</h4><div class="panel-copy">Persistent steering and imported source count.</div></div>', unsafe_allow_html=True)
            st.metric("Imported Kaggle sources", len(imported_sources))
            st.markdown("**Operator note preview**")
            st.code((operator_note().strip() or "No persistent steering note saved.")[:900], language="text")

        access_left, access_right = st.columns([1.0, 1.0])
        with access_left:
            st.markdown('<div class="panel"><h4>Remote Access</h4><div class="panel-copy">Use these URLs from other devices. Tailscale is the safest internet-facing route. The dashboard password gate is enforced before the UI loads.</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Local:** `{endpoints.get('localhost_url', 'n/a')}`")
            lan_urls = endpoints.get("lan_urls", [])
            tailscale_urls = endpoints.get("tailscale_urls", [])
            if lan_urls:
                st.markdown("**LAN URLs**")
                for url in lan_urls[:6]:
                    st.code(url, language="text")
            if tailscale_urls:
                st.markdown("**Tailscale URLs**")
                for url in tailscale_urls[:6]:
                    st.code(url, language="text")
        with access_right:
            st.markdown('<div class="panel"><h4>Always-On Checklist</h4><div class="panel-copy">Keep the laptop reachable while the control center and daemon run in the background.</div></div>', unsafe_allow_html=True)
            st.markdown("- Keep the laptop on AC power.")
            st.markdown("- Disable sleep and hibernate for plugged-in mode.")
            st.markdown("- Let the dashboard bind on all interfaces.")
            st.markdown("- Use Tailscale for remote access outside your home network.")
            st.markdown("- Keep the UI password enabled for deployed access.")

    with tab_control:
        st.markdown('<div class="panel"><h4>Autonomy Flight Deck</h4><div class="panel-copy">Start, stop, restart, run a single cycle, inject steering, and harvest Kaggle sources.</div></div>', unsafe_allow_html=True)
        allow_submit = st.checkbox(
            "Allow Kaggle submissions",
            key=control_widget_key("allow_submit"),
        )
        history = st.number_input(
            "Intel history depth",
            min_value=3,
            max_value=100,
            key=control_widget_key("history"),
        )
        delta = st.number_input(
            "Minimum local delta to care about",
            min_value=0.0,
            max_value=1000.0,
            step=1.0,
            key=control_widget_key("min_local_delta"),
        )
        sleep_seconds = st.number_input(
            "Sleep between loops",
            min_value=60,
            max_value=86400,
            step=60,
            key=control_widget_key("sleep_seconds"),
        )
        max_pending = st.number_input(
            "Pending submission cap",
            min_value=0,
            max_value=10,
            step=1,
            key=control_widget_key("max_pending_submissions"),
        )

        top_actions = st.columns(3)
        if top_actions[0].button("Bring Skynet Online", width="stretch"):
            set_action_result(
                "Autonomy daemon start requested.",
                start_daemon(
                    allow_submit=allow_submit,
                    history=int(history),
                    min_local_delta=float(delta),
                    sleep_seconds=int(sleep_seconds),
                    max_pending_submissions=int(max_pending),
                ),
            )
            st.rerun()
        if top_actions[1].button("Destroy Skynet", width="stretch"):
            set_action_result("Autonomy daemon stop requested.", stop_daemon())
            st.rerun()
        if top_actions[2].button("Send a T-800", width="stretch"):
            set_action_result(
                "One-cycle run finished.",
                run_single_cycle(allow_submit=allow_submit, history=int(history), min_local_delta=float(delta)),
            )
            st.rerun()

        lower_actions = st.columns(3)
        if lower_actions[0].button("Scan the Battlefield", width="stretch"):
            persist_runtime_preferences(
                allow_submit=bool(allow_submit),
                history=int(history),
                min_local_delta=float(delta),
                sleep_seconds=int(sleep_seconds),
                max_pending_submissions=int(max_pending),
            )
            set_action_result("State sync finished.", sync_state(history=int(history)))
            st.rerun()
        if lower_actions[1].button("Cyberdyne Diagnostics", width="stretch"):
            set_action_result("Healthcheck finished.", healthcheck())
            st.rerun()
        if lower_actions[2].button("Reboot the Time Core", width="stretch"):
            save_operator_note(operator_note())
            set_action_result(
                "Daemon restart requested.",
                restart_daemon_async(
                    allow_submit=allow_submit,
                    history=int(history),
                    min_local_delta=float(delta),
                    sleep_seconds=int(sleep_seconds),
                    max_pending_submissions=int(max_pending),
                ),
            )
            st.rerun()

        st.subheader("Operator Prompt")
        note_value = st.text_area(
            "Persistent steering note for the autonomous planner",
            value=operator_note(),
            height=170,
            placeholder="Keep seed-preserving swaps near the strongest accepted pack, avoid repeating the last failed family, and only care about large public jumps.",
        )
        note_buttons = st.columns(3)
        if note_buttons[0].button("Rewrite the Future", width="stretch"):
            save_operator_note(note_value)
            st.success("Operator note saved.")
        if note_buttons[1].button("Erase the Timeline", width="stretch"):
            save_operator_note("")
            st.success("Operator note cleared.")
        if note_buttons[2].button("Arm Judgment Day", width="stretch"):
            save_operator_note(note_value)
            st.success("Operator note armed for future cycles.")

        st.subheader("Kaggle Intel Intake")
        kaggle_seed_text = st.text_area(
            "Paste Kaggle URLs or CLI lines",
            value=st.session_state.get("kaggle_intake_text", ""),
            height=150,
            placeholder="https://www.kaggle.com/code/jiweiliu/kaggle-agent-lb794-inference\nkaggle kernels pull jonathanchan/ngc36-constraint-smart-logic-mix-blending\nhttps://www.kaggle.com/datasets/thisray/neurogolf-4743-93-submission-task-table",
            key="kaggle_intake_text",
        )
        intake_buttons = st.columns(2)
        if intake_buttons[0].button("Harvest Future Files", width="stretch"):
            set_action_result("Kaggle source fetch finished.", fetch_kaggle_targets(kaggle_seed_text))
            st.cache_data.clear()
            st.rerun()
        if intake_buttons[1].button("Review Time Displacement", width="stretch"):
            st.info("Imported sources appear below and in the NeuroGolf scratch gui_imports folder.")

        if imported_sources:
            st.dataframe(imported_sources, width="stretch", hide_index=True)

    with tab_models:
        st.markdown('<div class="panel"><h4>Neural Foundry</h4><div class="panel-copy">Role assignments, installed models, profile loading, and auto context defaults per selected model.</div></div>', unsafe_allow_html=True)
        detected_profile_key, detected_meta = best_profile_for_installed_models(models)
        detected_profile = MODEL_PROFILES[detected_profile_key]

        summary_left, summary_right = st.columns([1.05, 0.95])
        with summary_left:
            st.subheader("Current Saved Roles")
            st.dataframe(current_role_map(), width="stretch", hide_index=True)
        with summary_right:
            st.subheader("Installed Ollama Models")
            st.dataframe(models, width="stretch", hide_index=True)

        st.subheader("Recommended Model Profiles")
        profile_key = st.selectbox(
            "Pick a profile",
            options=list(MODEL_PROFILES.keys()),
            format_func=lambda key: MODEL_PROFILES[key]["label"],
        )
        profile = MODEL_PROFILES[profile_key]
        profile_cols = st.columns([1.0, 1.0])
        with profile_cols[0]:
            st.markdown(f"**Tier:** {profile['tier']}")
            st.markdown(profile["summary"])
            for note in profile["notes"]:
                st.markdown(f"- {note}")
        with profile_cols[1]:
            st.json({"roles": profile["roles"], "contexts": profile["contexts"], "recommended_pulls": profile["recommended_pulls"]})

        detected_match_count = len(detected_meta.get("matched", []))
        detected_missing = detected_meta.get("missing", [])
        st.info(
            f"Auto-detected best installed stack: **{detected_profile['label']}** "
            f"({detected_match_count}/{len(ROLE_LABELS)} role models available locally)."
        )
        if detected_missing:
            st.caption(
                "Missing pieces for that profile: "
                + ", ".join(f"{item['role']} -> {item['model']}" for item in detected_missing[:6])
                + (" ..." if len(detected_missing) > 6 else "")
            )

        source_cols = st.columns(3)
        shown = 0
        for model_name in sorted({value.split(":", 1)[0] for value in profile["recommended_pulls"]}):
            if model_name in MODEL_SOURCES:
                source_cols[shown % 3].markdown(f"[{model_name}]({MODEL_SOURCES[model_name]})")
                shown += 1

        profile_buttons = st.columns(4)
        if profile_buttons[0].button("Load Into Roles", width="stretch"):
            load_profile_into_editor(profile_key)
            st.rerun()
        if profile_buttons[1].button("Use My Installed Best Models", width="stretch"):
            load_profile_into_editor(detected_profile_key)
            st.session_state["last_action"] = {
                "label": f"Loaded detected profile: {detected_profile['label']}",
                "result": {
                    "ok": True,
                    "stdout": "\n".join(
                        [
                            f"Matched roles: {detected_match_count}/{len(ROLE_LABELS)}",
                            f"Profile key: {detected_profile_key}",
                            "Role map:",
                            *[
                                f"- {ROLE_LABELS[item['role']]} -> {item['model']}"
                                for item in detected_meta.get("matched", [])
                            ],
                        ]
                    ),
                },
            }
            st.rerun()
        if profile_buttons[2].button("Install Neural Chip", width="stretch"):
            update_env_settings(env_updates_for_profile(profile_key))
            st.success("Profile written to .env.")
        if profile_buttons[3].button("Judgment Day Upgrade", width="stretch"):
            control_settings = current_control_settings()
            update_env_settings(env_updates_for_profile(profile_key))
            persist_runtime_preferences(
                allow_submit=bool(control_settings["allow_submit"]),
                history=int(control_settings["history"]),
                min_local_delta=float(control_settings["min_local_delta"]),
                sleep_seconds=int(control_settings["sleep_seconds"]),
                max_pending_submissions=int(control_settings["max_pending_submissions"]),
            )
            restart_daemon_async(
                allow_submit=bool(control_settings["allow_submit"]),
                history=int(control_settings["history"]),
                min_local_delta=float(control_settings["min_local_delta"]),
                sleep_seconds=int(control_settings["sleep_seconds"]),
                max_pending_submissions=int(control_settings["max_pending_submissions"]),
            )
            st.success("Profile applied. Background daemon restart requested.")

        st.subheader("Manual Role Override")
        for role, label in ROLE_LABELS.items():
            role_cols = st.columns([1.6, 0.8])
            select_key = role_widget_key(role)
            context_key = role_context_key(role)
            if not model_options:
                model_options = [st.session_state.get(select_key, "")]
            if st.session_state.get(select_key, "") not in model_options:
                model_options.append(st.session_state.get(select_key, ""))
            with role_cols[0]:
                st.selectbox(
                    label,
                    options=sorted(set(model_options)),
                    index=sorted(set(model_options)).index(st.session_state.get(select_key, "")) if st.session_state.get(select_key, "") in sorted(set(model_options)) else 0,
                    key=select_key,
                    on_change=sync_context_from_role,
                    args=(role,),
                )
            with role_cols[1]:
                st.number_input(
                    f"{label} context",
                    min_value=1024,
                    max_value=262144,
                    step=1024,
                    key=context_key,
                )

        manual_buttons = st.columns(2)
        if manual_buttons[0].button("Override the Neural Net", width="stretch"):
            updates = {}
            for role in ROLE_LABELS:
                updates[ROLE_ENV_KEYS[role]] = st.session_state.get(role_widget_key(role), "")
                updates[CTX_ENV_KEYS[role]] = str(st.session_state.get(role_context_key(role), default_context_for_model("", role)))
            update_env_settings(updates)
            st.success("Manual role and context override saved to .env.")
        if manual_buttons[1].button("Reset Contexts To Model Defaults", width="stretch"):
            st.session_state["pending_context_resets"] = list(ROLE_LABELS.keys())
            st.rerun()

    with tab_neurogolf:
        st.markdown('<div class="panel"><h4>Battlefield Telemetry</h4><div class="panel-copy">Competition state, leaderboard standing, recent submissions, and output manifests.</div></div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Current Rank", fmt_value(rank_info.get("rank"), digits=0))
        metric_cols[1].metric("Leaderboard Score", fmt_value(rank_info.get("score")))
        metric_cols[2].metric("Latest Completed", fmt_value(latest_completed.get("public_score")))
        metric_cols[3].metric("Pending", fmt_value(state.get("pending_submission_count"), digits=0))

        chart_col, state_col = st.columns([1.08, 0.92])
        with chart_col:
            st.subheader("Submission Score Trend")
            if scores:
                st.line_chart(scores, height=260)
            else:
                st.info("No completed scores are available yet.")
        with state_col:
            st.subheader("Competition State")
            st.json(
                {
                    "team_name": team_name,
                    "rank": rank_info,
                    "best_completed_public_score": state.get("best_completed_public_score"),
                    "pending_submission_count": state.get("pending_submission_count"),
                    "current_manifest": state.get("current_manifest", {}),
                }
            )

        st.subheader("Recent Submissions")
        st.dataframe(submissions, width="stretch", hide_index=True)
        st.subheader("Recent Output Manifests")
        st.dataframe(state.get("recent_output_manifests", []), width="stretch", hide_index=True)

    with tab_reports:
        st.markdown('<div class="panel"><h4>Hunter-Killer Reports</h4><div class="panel-copy">Recent autonomy cycles, submit decisions, and the live daemon trace.</div></div>', unsafe_allow_html=True)
        report_rows = [
            {
                "path": item.get("_path"),
                "started_at": item.get("started_at"),
                "mode": (item.get("plan") or {}).get("mode"),
                "seed_label": (item.get("plan") or {}).get("seed_label"),
                "submitted": (item.get("submit_result") or {}).get("submitted"),
                "submit_reason": (item.get("submit_result") or {}).get("reason"),
            }
            for item in reports
        ]
        st.dataframe(report_rows, width="stretch", hide_index=True)

        report_view_col, log_col = st.columns([1.0, 1.0])
        with report_view_col:
            if reports:
                selected = st.selectbox(
                    "Inspect report",
                    options=list(range(len(reports))),
                    format_func=lambda idx: f"{reports[idx].get('started_at', 'unknown')} | {(reports[idx].get('plan') or {}).get('seed_label', 'n/a')}",
                )
                st.json(reports[selected])
            else:
                st.info("No cycle reports found yet.")
        with log_col:
            st.subheader("Latest Signal Feed")
            st.code(tail_log(140) or "No daemon log yet.", language="text")

    with tab_arc:
        render_arc_visualizer()

    with tab_chat:
        st.markdown('<div class="panel"><h4>Direct Line To Skynet</h4><div class="panel-copy">Use the Fast Operator Link for quick answers, or switch to the Orchestrator when you want deeper strategy synthesis grounded in live system state.</div></div>', unsafe_allow_html=True)
        ensure_chat_state()
        chat_context = build_orchestrator_chat_context(state, status, reports, imported_sources, current_role_map())
        chat_target = st.radio(
            "Chat target",
            options=["operator_fast", "orchestrator"],
            format_func=lambda value: "Fast Operator Link" if value == "operator_fast" else "Orchestrator",
            horizontal=True,
        )

        chat_actions = st.columns(2)
        if chat_actions[0].button("Purge Chat Memory", width="stretch"):
            st.session_state.pop("orchestrator_chat_messages", None)
            ensure_chat_state()
            st.rerun()
        if chat_actions[1].button("Inject Live State", width="stretch"):
            st.info("The orchestrator already receives live state, imported Kaggle sources, operator note, and recent experiment telemetry on every message.")

        for message in st.session_state["orchestrator_chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Ask the orchestrator what to do next, what to fetch, or how to improve the submission.")
        if user_prompt:
            st.session_state["orchestrator_chat_messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            with st.chat_message("assistant"):
                spinner_text = "Fast operator thinking..." if chat_target == "operator_fast" else "Orchestrator thinking..."
                with st.spinner(spinner_text):
                    reply = query_control_chat(chat_target, user_prompt, chat_context, env_settings)
                st.markdown(reply)
            st.session_state["orchestrator_chat_messages"].append({"role": "assistant", "content": reply})

bootstrap_env = load_env_settings()
render_auth_gate(bootstrap_env)
if not is_authenticated(bootstrap_env):
    st.stop()

with st.sidebar:
    render_sidebar_fragment()

render_dashboard()
