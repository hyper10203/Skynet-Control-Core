from __future__ import annotations

import hmac
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import ListedColormap

from core.arc_loader import load_tasks
from core.config import DEFAULT_DATA_DIR
from core.env_settings import load_env_settings, update_env_settings
from core.executor import ask_with_fallback
from core.model_profiles import (
    MODEL_PROFILES,
    MODEL_SOURCES,
    ROLE_LABELS,
    default_context_for_model,
    env_updates_for_profile,
)
from core.prompts import FAST_OPERATOR_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from core.self_improvement import (
    enable_self_improvement,
    disable_self_improvement,
    is_self_improvement_enabled,
    get_improvement_status,
    load_improvement_history,
)
from core.runtime_control import (
    archive_skynet_clutter as archive_generated_clutter,
    bridge_daemon_status,
    current_role_map,
    daemon_status,
    fetch_kaggle_targets,
    healthcheck,
    launch_openclaude_session,
    latest_team_name,
    leaderboard_rank_for_team,
    list_imported_kaggle_sources,
    ollama_models,
    operator_note,
    persist_runtime_preferences,
    read_distillation_status,
    read_state,
    read_status,
    recent_reports,
    restart_daemon_async,
    remote_access_endpoints,
    remote_bridge_status,
    rotate_runtime_node_pairing_token,
    run_single_cycle,
    save_operator_note,
    save_distillation_settings,
    save_imported_seed_controls,
    list_remote_bridge_nodes,
    process_remote_bridge_command,
    publish_remote_bridge_heartbeat,
    queue_remote_bridge_command,
    restart_bridge_daemon_async,
    save_runtime_node_kaggle_credentials,
    save_runtime_node_operator_binding,
    save_runtime_node_settings,
    skynet_clutter_summary as generated_clutter_summary,
    start_bridge_daemon,
    start_daemon,
    stop_bridge_daemon,
    stop_daemon,
    sync_state,
    tail_log,
    runtime_node_healthcheck,
    runtime_node_summary,
)


APP_NAME = "AxiomGraph Operations Core"
APP_SHORT_NAME = "AxiomGraph"
WEBSITE_BASE_URL = "http://127.0.0.1:4173"
PORTAL_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "portal"
PORTAL_PREVIEW_IMAGES = {
    "landing": PORTAL_ASSET_DIR / "landing-desktop-final.png",
    "auth": PORTAL_ASSET_DIR / "auth-final.png",
    "app": PORTAL_ASSET_DIR / "app-desktop-final.png",
}
PORTAL_PHASE_IMAGES = [
    PORTAL_ASSET_DIR / "core-phase-1.png",
    PORTAL_ASSET_DIR / "core-phase-2.png",
    PORTAL_ASSET_DIR / "core-phase-3.png",
    PORTAL_ASSET_DIR / "core-phase-4.png",
]
PORTAL_PHASE_LABELS = [
    "Wake the core",
    "Expand the graph",
    "Reveal the command plane",
    "Lock operator control",
]

st.set_page_config(page_title=APP_NAME, page_icon="A", layout="wide")

if "AXIOMGRAPH_BRIDGE_GITHUB_TOKEN" in st.secrets:
    os.environ["AXIOMGRAPH_BRIDGE_GITHUB_TOKEN"] = str(st.secrets["AXIOMGRAPH_BRIDGE_GITHUB_TOKEN"])

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-0: #0f1115;
    --bg-1: #151923;
    --bg-2: #101820;
    --line: rgba(148, 163, 184, 0.24);
    --line-hot: rgba(245, 158, 11, 0.46);
    --ink: #f8fafc;
    --sub: #a8b1c2;
    --aqua: #14b8a6;
    --cyan: #38bdf8;
    --pink: #f43f5e;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

.stApp {
    background: linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 52%, var(--bg-2) 100%);
    color: var(--ink);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

pre, code, .stCode, [data-testid="stCodeBlock"] {
    font-family: 'JetBrains Mono', monospace !important;
}

h1, h2, h3, .stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0;
}

.top-shell {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    background:
        linear-gradient(135deg, rgba(20,184,166,0.10), rgba(56,189,248,0.07) 56%, rgba(245,158,11,0.08)),
        rgba(15, 17, 21, 0.92);
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.24);
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
    border-radius: 6px;
    padding: 0.36rem 0.78rem;
    background: rgba(17, 24, 39, 0.82);
    color: var(--aqua);
    font-size: 0.8rem;
    transition: all 0.2s ease;
}

.badge-chip:hover {
    border-color: var(--aqua);
    box-shadow: 0 0 12px rgba(127, 255, 238, 0.2);
}

.badge-online {
    background: linear-gradient(135deg, rgba(46, 204, 64, 0.2), rgba(46, 204, 64, 0.1));
    border-color: rgba(46, 204, 64, 0.5);
    color: #2ECC40;
}

.badge-offline {
    background: linear-gradient(135deg, rgba(255, 65, 54, 0.2), rgba(255, 65, 54, 0.1));
    border-color: rgba(255, 65, 54, 0.5);
    color: #FF4136;
}

.badge-target {
    background: linear-gradient(135deg, rgba(255, 220, 0, 0.15), rgba(255, 220, 0, 0.08));
    border-color: rgba(255, 220, 0, 0.4);
    color: #FFDC00;
}

.badge-phase {
    background: linear-gradient(135deg, rgba(32, 190, 255, 0.15), rgba(32, 190, 255, 0.08));
    border-color: rgba(32, 190, 255, 0.4);
    color: var(--cyan);
}

.hero-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 0.9rem;
}

.hero-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.42rem 0.9rem;
    border-radius: 6px;
    border: 1px solid var(--line);
    background: rgba(17, 24, 39, 0.82);
    color: #d8f7ff !important;
    text-decoration: none !important;
    font-size: 0.82rem;
    transition: all 0.2s ease;
}

.hero-link:hover {
    border-color: var(--aqua);
    box-shadow: 0 0 14px rgba(127, 255, 238, 0.18);
    transform: translateY(-1px);
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
    border-radius: 8px;
    padding: 1rem 1rem 0.95rem 1rem;
    background: rgba(17, 24, 39, 0.74);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
}

.panel h4 {
    margin: 0 0 0.55rem 0;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0;
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
    border-radius: 8px;
    padding: 0.85rem 0.9rem;
    background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,17,21,0.95));
    transition: all 0.2s ease;
}

.status-card:hover {
    border-color: var(--aqua);
    box-shadow: 0 0 20px rgba(127, 255, 238, 0.15);
}

.status-label {
    color: var(--sub);
    font-size: 0.78rem;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.status-value {
    margin-top: 0.3rem;
    color: var(--ink);
    font-size: 1.35rem;
    font-family: 'Inter', sans-serif;
}

.status-sub {
    margin-top: 0.2rem;
    color: var(--aqua);
    font-size: 0.78rem;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.3rem;
}

.status-online { background: var(--success); box-shadow: 0 0 8px var(--success); animation: pulse-green 2s infinite; }
.status-offline { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
.status-warning { background: var(--warning); box-shadow: 0 0 8px var(--warning); animation: pulse-yellow 2s infinite; }
.status-info { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 8px var(--success); opacity: 1; }
    50% { box-shadow: 0 0 16px var(--success); opacity: 0.8; }
}

@keyframes pulse-yellow {
    0%, 100% { box-shadow: 0 0 8px var(--warning); opacity: 1; }
    50% { box-shadow: 0 0 16px var(--warning); opacity: 0.8; }
}

.alert-banner {
    border: 1px solid var(--line-hot);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(244,63,94,0.07));
    margin-bottom: 1rem;
}

.alert-title {
    color: var(--pink);
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.alert-text {
    color: var(--ink);
    font-size: 0.9rem;
    line-height: 1.5;
}

.metric-highlight {
    background: linear-gradient(135deg, rgba(127,255,238,0.15), rgba(32,190,255,0.1));
    border: 1px solid rgba(127,255,238,0.25);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.5rem 0;
}

/* Improve Streamlit's default info/success/warning boxes */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
}

[data-testid="stAlert"] > div {
    background: rgba(17, 24, 39, 0.9) !important;
}

/* Success - green tint */
[data-testid="stAlert"][data-baseweb="notification"][data-kind="positive"] {
    border-color: rgba(46, 204, 64, 0.5) !important;
    box-shadow: 0 0 15px rgba(46, 204, 64, 0.15) !important;
}

/* Warning - yellow tint */
[data-testid="stAlert"][data-baseweb="notification"][data-kind="warning"] {
    border-color: rgba(255, 220, 0, 0.5) !important;
    box-shadow: 0 0 15px rgba(255, 220, 0, 0.15) !important;
}

/* Info - cyan tint */
[data-testid="stAlert"][data-baseweb="notification"][data-kind="info"] {
    border-color: rgba(32, 190, 255, 0.5) !important;
    box-shadow: 0 0 15px rgba(32, 190, 255, 0.15) !important;
}

[data-testid="stMetric"] {
    background: rgba(5, 12, 19, 0.74);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.72rem;
}

div.stButton > button {
    border-radius: 6px;
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(56,189,248,0.14));
    color: var(--ink);
    font-family: 'Inter', sans-serif;
    font-weight: 650;
    letter-spacing: 0;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: var(--aqua);
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(56,189,248,0.22));
    box-shadow: 0 12px 24px rgba(20, 184, 166, 0.16);
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(1px);
}

div.stButton > button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    filter: grayscale(0.5);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--aqua), var(--cyan) 55%, var(--warning) 100%);
    border-radius: 6px;
    box-shadow: none;
}

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
    margin: 1.5rem 0;
}

.subsection-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    color: var(--ink);
    margin: 1.2rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

div[data-baseweb="select"] > div,
.stTextArea textarea,
.stTextInput input,
.stNumberInput input {
    background: rgba(4, 11, 18, 0.82) !important;
    color: var(--ink) !important;
    border-radius: 6px !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(5, 12, 19, 0.72);
    border: 1px solid var(--line);
    border-radius: 6px;
    color: #d8f7ff;
    padding: 0.5rem 0.9rem;
}

.auth-shell {
    max-width: 760px;
    margin: 4vh auto 0 auto;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.6rem;
    background:
        linear-gradient(135deg, rgba(20,184,166,0.10), rgba(56,189,248,0.06) 55%, rgba(245,158,11,0.08)),
        rgba(15, 17, 21, 0.93);
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.24);
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
        border-radius: 8px;
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


def format_bytes(value) -> str:
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f"{size:.1f} {units[unit]}" if unit else f"{int(size)} B"


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
          <h1 class="auth-title">AxiomGraph Operations Core</h1>
          <div class="auth-copy">
            This deployment is locked. Enter the workspace password to access the dashboard, daemon controls,
            ARC visualizer, and NeuroGolf automation surfaces.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_mid, col_right = st.columns([1, 1.3, 1])
    with col_mid:
        password = st.text_input("Workspace password", type="password", key=AUTH_INPUT_KEY)
        if st.button("Unlock Console", width="stretch"):
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


@st.cache_data(ttl=60, show_spinner=False)
def website_portal_available(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/index.html", timeout=1.2) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def render_portal_fallback(
    *,
    daemon: dict,
    rank_info: dict,
    latest_completed: dict,
    latest_report: dict,
    submission_policy: dict,
) -> None:
    st.caption(
        "Local website service is not reachable, so this Streamlit deployment is carrying the portal surface directly."
    )

    hero_left, hero_right = st.columns([1.02, 0.98])
    with hero_left:
        st.markdown("### A cinematic shell for the live control core")
        st.write(
            "This deployment keeps the landing narrative, operator framing, and command visibility inside the same "
            "Streamlit runtime that controls the NeuroGolf loop. When the separate Node website is available, the "
            "Portal tab embeds it live. When it is not, the core experience still ships as one app."
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("Daemon", "Online" if daemon.get("running") else "Offline")
        metric_cols[1].metric("Rank", fmt_value(rank_info.get("rank"), digits=0))
        metric_cols[2].metric("Best Public", fmt_value(latest_completed.get("public_score")))
        metric_cols[3].metric(
            "Next Submit",
            fmt_value(submission_policy.get("next_submit_score")),
        )
        st.markdown(
            f"**Submit policy:** any verified public gain at or above "
            f"`{fmt_value(submission_policy.get('minimum_public_gain_to_submit'))}` point."
        )
        if latest_report:
            plan = latest_report.get("plan") or {}
            st.markdown(
                f"**Latest cycle:** `{plan.get('action', 'unknown')}` targeting "
                f"`{plan.get('target') or plan.get('seed_label') or 'n/a'}`"
            )
    with hero_right:
        landing_preview = PORTAL_PREVIEW_IMAGES["landing"]
        if landing_preview.exists():
            st.image(str(landing_preview), caption="Streamlit-hosted landing preview", use_container_width=True)
        else:
            st.info("Landing preview asset is not available yet in the repo.")

    st.markdown("### Scroll Sequence Frames")
    phase_cols = st.columns(len(PORTAL_PHASE_IMAGES))
    for idx, image_path in enumerate(PORTAL_PHASE_IMAGES):
        with phase_cols[idx]:
            if image_path.exists():
                st.image(str(image_path), caption=PORTAL_PHASE_LABELS[idx], use_container_width=True)
            else:
                st.warning(PORTAL_PHASE_LABELS[idx])

    st.markdown("### Command Surfaces")
    preview_cols = st.columns(2)
    preview_specs = [
        ("Secure Entry", PORTAL_PREVIEW_IMAGES["auth"]),
        ("Command Nexus", PORTAL_PREVIEW_IMAGES["app"]),
    ]
    for col, (label, image_path) in zip(preview_cols, preview_specs):
        with col:
            if image_path.exists():
                st.image(str(image_path), caption=label, use_container_width=True)
            else:
                st.info(f"{label} preview asset is not available yet in the repo.")


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


def shutdown_and_logout() -> dict:
    daemon_result = stop_daemon(reset_state=False)
    bridge_result = stop_bridge_daemon()
    st.session_state[AUTH_STATE_KEY] = False
    st.session_state.pop("orchestrator_chat_messages", None)
    st.session_state.pop("last_action", None)
    st.cache_data.clear()
    return {
        "ok": bool(daemon_result.get("ok", False)) and bool(bridge_result.get("ok", True)),
        "stdout": "\n".join(
            part for part in [
                daemon_result.get("stdout", "").strip(),
                bridge_result.get("stdout", "").strip(),
            ] if part
        ),
        "stderr": "\n".join(
            part for part in [
                daemon_result.get("stderr", "").strip(),
                bridge_result.get("stderr", "").strip(),
            ] if part
        ),
    }


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
        "min_local_delta": float(env_settings.get("NEUROGOLF_AUTONOMY_LOCAL_DELTA_THRESHOLD", "0")),
        "sleep_seconds": int(env_settings.get("NEUROGOLF_AUTONOMY_LOOP_SECONDS", "1800")),
        "max_pending_submissions": int(env_settings.get("NEUROGOLF_AUTONOMY_MAX_PENDING_SUBMISSIONS", "1")),
    }
    for name, value in defaults.items():
        key = control_widget_key(name)
        if key not in st.session_state:
            st.session_state[key] = value
    migration_key = "submission_policy_any_gain_v1"
    if migration_key not in st.session_state:
        st.session_state[control_widget_key("min_local_delta")] = defaults["min_local_delta"]
        st.session_state[migration_key] = True


def current_control_settings() -> dict[str, int | float | bool]:
    return {
        "allow_submit": bool(st.session_state.get(control_widget_key("allow_submit"), True)),
        "history": int(st.session_state.get(control_widget_key("history"), 10)),
        "min_local_delta": float(st.session_state.get(control_widget_key("min_local_delta"), 0.0)),
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
    fallback_models = [env_settings.get("ARC_MODEL_OPERATOR_FAST", "qwen2.5")] if not use_fast_lane else [env_settings.get("ARC_MODEL_ORCHESTRATOR", "qwen2.5")]
    response = ask_with_fallback(
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
        fallback_models=fallback_models,
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
        "rtx3050-offline-max",
        "current-safe",
        "giant-anchor",
        "balanced-2026",
        "coding-lab",
        "your-local-max",
        "agentic-max",
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
    try:
        env_settings = load_env_settings()
    except Exception:
        env_settings = {}
    
    try:
        state = read_state() or {}
    except Exception:
        state = {}
    
    try:
        daemon = daemon_status() or {"running": False}
    except Exception:
        daemon = {"running": False}
    
    try:
        endpoints = remote_access_endpoints()
    except Exception:
        endpoints = {}
    
    submissions = state.get("recent_submissions", [])
    team_name = str(state.get("current_team_name") or latest_team_name()).strip()
    
    try:
        rank_info = cached_rank(team_name)
    except Exception:
        rank_info = {}
    
    latest_completed = current_completed_submission(submissions)

    st.title("Operations Console")
    st.caption("Auto refresh cadence: every 5 minutes")
    if st.button("Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.metric("Daemon", "Running" if daemon.get("running") else "Stopped")
    st.metric("Rank", fmt_value(rank_info.get("rank"), digits=0))
    st.metric("Latest Score", fmt_value(latest_completed.get("public_score")))
    st.metric("Best Public", fmt_value(state.get("best_completed_public_score")))
    st.metric("Pending Submissions", fmt_value(state.get("pending_submission_count"), digits=0))
    if team_name:
        st.caption(f"Team: {team_name}")
    st.caption(f"Remote: {endpoints.get('recommended_remote', 'n/a')}")
    if ui_requires_password(env_settings) and st.button("Lock Console", width="stretch"):
        st.session_state[AUTH_STATE_KEY] = False
        st.rerun()
    if st.button("Sign Out And Stop", width="stretch"):
        shutdown_and_logout()
        st.rerun()


@st.fragment(run_every=REFRESH_EVERY_SECONDS)
def render_dashboard() -> None:
    # Wrap in try-except to prevent dashboard crashes
    try:
        state = read_state() or {}
    except Exception as e:
        st.error(f"Failed to read state: {e}")
        state = {}
    
    try:
        status = read_status() or {}
    except Exception as e:
        st.error(f"Failed to read status: {e}")
        status = {}
    
    try:
        daemon = daemon_status() or {"running": False}
    except Exception as e:
        st.error(f"Failed to read daemon status: {e}")
        daemon = {"running": False}
    
    try:
        reports = recent_reports(limit=12)
    except Exception as e:
        st.error(f"Failed to read reports: {e}")
        reports = []
    
    try:
        env_settings = load_env_settings()
    except Exception as e:
        st.error(f"Failed to load env settings: {e}")
        env_settings = {}
    
    seed_role_editor(env_settings)
    seed_control_editor(env_settings)
    apply_pending_context_resets()
    
    try:
        models = cached_ollama_models()
    except Exception as e:
        st.warning(f"Failed to load Ollama models: {e}")
        models = []
    
    model_options = model_options_from_installed(models, env_settings)
    submissions = state.get("recent_submissions", [])
    team_name = str(state.get("current_team_name") or latest_team_name()).strip()
    
    try:
        rank_info = cached_rank(team_name)
    except Exception as e:
        st.warning(f"Failed to load rank: {e}")
        rank_info = {}
    
    latest_completed = current_completed_submission(submissions)
    latest_submission = submissions[0] if submissions else {}
    submission_policy = state.get("submission_policy", state.get("campaign_progress", {}))
    submission_policy = submission_policy if isinstance(submission_policy, dict) else {}
    progress = clamp_progress(status.get("progress", 0.0))
    scores = completed_scores(submissions)
    latest_report = reports[0] if reports else {}
    latest_plan = latest_report.get("plan") or {}
    latest_submit = latest_report.get("submit_result") or {}
    
    try:
        imported_sources = list_imported_kaggle_sources(limit=15)
    except Exception as e:
        st.warning(f"Failed to load imported sources: {e}")
        imported_sources = []
    
    try:
        endpoints = remote_access_endpoints()
    except Exception as e:
        st.warning(f"Failed to load endpoints: {e}")
        endpoints = []
    
    try:
        node_summary = runtime_node_summary()
    except Exception as e:
        st.warning(f"Failed to load runtime node summary: {e}")
        node_summary = {}
    
    try:
        bridge_summary = remote_bridge_status()
    except Exception as e:
        st.warning(f"Failed to load remote bridge status: {e}")
        bridge_summary = {"bridge_daemon": bridge_daemon_status()}
    
    try:
        remote_nodes = list_remote_bridge_nodes()
    except Exception as e:
        st.warning(f"Failed to load remote bridge nodes: {e}")
        remote_nodes = []

    daemon_badge_class = "badge-online" if daemon.get("running") else "badge-offline"
    daemon_badge_text = "ONLINE" if daemon.get("running") else "OFFLINE"
    si_enabled = is_self_improvement_enabled()
    si_badge_class = "badge-online" if si_enabled else "badge-offline"
    si_badge_text = "SI ON" if si_enabled else "SI OFF"
    comp_data = state.get("competition_data", {})
    comp_ready = comp_data.get("ready", False)
    comp_badge_class = "badge-online" if comp_ready else "badge-offline"
    comp_badge_text = "DATA READY" if comp_ready else "NO DATA"

    st.markdown(
        f"""
        <div class="top-shell">
          <div class="hero-grid">
            <div>
              <div class="hero-kicker">Local ARC and NeuroGolf Operations</div>
              <h1 class="hero-title">AxiomGraph Operations Core</h1>
              <div class="hero-copy">
                A focused workbench for ARC reasoning and NeuroGolf graph optimization with daemon control,
                Kaggle source intake, model-role management, submission telemetry, and ARC task review.
              </div>
              <div class="badge-row">
                <div class="badge-chip {daemon_badge_class}">● {daemon_badge_text}</div>
                <div class="badge-chip {si_badge_class}">{si_badge_text}</div>
                <div class="badge-chip {comp_badge_class}">{comp_badge_text}</div>
                <div class="badge-chip">Rank: {fmt_value(rank_info.get("rank"), digits=0)}</div>
                <div class="badge-chip">Latest: {fmt_value(latest_completed.get("public_score"))}</div>
                <div class="badge-chip">Best: {fmt_value(state.get("best_completed_public_score"))}</div>
                <div class="badge-chip">Policy: submit any +{fmt_value(submission_policy.get("minimum_public_gain_to_submit"), digits=2, fallback="1.00")} gain</div>
                <div class="badge-chip badge-phase">{status.get("phase", "idle")}</div>
              </div>
              <div class="hero-links">
                <a class="hero-link" href="{WEBSITE_BASE_URL}/index.html" target="_blank">Landing Site</a>
                <a class="hero-link" href="{WEBSITE_BASE_URL}/auth.html" target="_blank">Secure Entry</a>
                <a class="hero-link" href="{WEBSITE_BASE_URL}/app.html" target="_blank">Web Command Deck</a>
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

    daemon_indicator = "status-online" if daemon.get("running") else "status-offline"
    daemon_status_text = "ONLINE" if daemon.get("running") else "OFFLINE"

    phase = status.get("phase", "idle")
    is_reset = status.get("reset", False)
    if phase == "improving":
        phase_indicator = "status-online"  # Green for self-improvement
    elif phase == "idle":
        phase_indicator = "status-warning"  # Yellow for idle
    else:
        phase_indicator = "status-info"  # Blue for active work

    # Show fresh-start state when daemon memory was reset.
    phase_display = f"{phase} | fresh start" if is_reset else phase

    st.markdown(
        f"""
        <div class="status-card-grid">
          <div class="status-card">
            <div class="status-label"><span class="status-indicator {phase_indicator}"></span>Current Phase</div>
            <div class="status-value">{phase_display}</div>
            <div class="status-sub">{status.get("message", "No live status yet.")}</div>
          </div>
          <div class="status-card">
            <div class="status-label"><span class="status-indicator status-info"></span>Current Rank</div>
            <div class="status-value">{fmt_value(rank_info.get("rank"), digits=0)}</div>
            <div class="status-sub">Team: {team_name or "n/a"}</div>
          </div>
          <div class="status-card">
            <div class="status-label"><span class="status-indicator status-info"></span>Latest Completed</div>
            <div class="status-value">{fmt_value(latest_completed.get("public_score"))}</div>
            <div class="status-sub">{latest_completed.get("message", "Awaiting completed score")}</div>
          </div>
          <div class="status-card">
            <div class="status-label"><span class="status-indicator {daemon_indicator}"></span>Daemon {daemon_status_text}</div>
            <div class="status-value">{fmt_value(daemon.get("pid"), digits=0)}</div>
            <div class="status-sub">CPU {fmt_value(daemon.get("cpu_percent"))}% | RAM {fmt_value(daemon.get("memory_mb"))} MB | {"STALE" if daemon.get("status_stale") else "LIVE"}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(progress)
    progress_color = "#2ECC40" if progress >= 0.8 else "#FFDC00" if progress >= 0.4 else "#FF4136"
    st.markdown(
        f"""
        <div style="text-align: center; color: {progress_color}; font-family: 'Inter', sans-serif; font-size: 0.85rem; margin-top: 0.3rem;">
            Run Progress: {progress * 100:.0f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_portal, tab_node, tab_fleet, tab_control, tab_models, tab_neurogolf, tab_reports, tab_arc, tab_chat = st.tabs(
        ["Overview", "Portal", "Runtime Node", "Fleet Bridge", "Control", "Models", "NeuroGolf", "Reports", "ARC Visualizer", "Orchestrator Chat"]
    )

    with tab_overview:
        left_col, right_col = st.columns([1.02, 0.98])
        with left_col:
            st.markdown('<div class="panel"><h4>Runtime Telemetry</h4><div class="panel-copy">Live state from the daemon, planner, and Kaggle memory.</div></div>', unsafe_allow_html=True)
            st.json(
                {
                    "daemon": daemon,
                    "status": status,
                    "rank": rank_info,
                    "latest_submission": latest_submission,
                    "latest_completed": latest_completed,
                    "submission_policy": submission_policy,
                }
            )
        with right_col:
            st.markdown('<div class="panel"><h4>Current Workstream</h4><div class="panel-copy">Active lane, chosen seed family, and latest autonomous decision point.</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Phase:** `{status.get('phase', 'idle')}`")
            st.markdown(f"**Message:** {status.get('message', 'No active workstream text yet.')}")
            st.markdown(f"**Latest plan action:** `{latest_plan.get('action', 'n/a')}`")
            st.markdown(f"**Latest plan target:** `{latest_plan.get('target') or latest_plan.get('seed_label', 'n/a')}`")
            st.markdown(f"**Latest plan variant:** `{latest_plan.get('variant_hint') or latest_plan.get('mode', 'n/a')}`")
            st.markdown(f"**Latest submit decision:** `{latest_submit.get('reason', 'n/a')}`")
            if submission_policy:
                st.markdown(
                    f"**Submission policy:** submit any valid pack that clears the best known public score by `+{fmt_value(submission_policy.get('minimum_public_gain_to_submit'), digits=2, fallback='1.00')}`."
                )
            if daemon.get("status_stale"):
                st.warning(f"Daemon status looks stale: last update was about {fmt_value(daemon.get('status_age_seconds'))} seconds ago.")
            st.subheader("Daemon Log")
            st.code(tail_log(80) or "No daemon log yet.", language="text")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        trend_col, note_col = st.columns([1.1, 0.9])
        with trend_col:
            st.markdown('<div class="panel"><h4>Score Trajectory</h4><div class="panel-copy">Completed public scores seen by the autonomous system so far.</div></div>', unsafe_allow_html=True)
            if scores:
                st.line_chart(scores, height=240)
            else:
                st.info("No completed public scores have been synced yet.")
        with note_col:
            st.markdown('<div class="panel"><h4>Runtime Flags</h4><div class="panel-copy">Persistent steering and imported source count.</div></div>', unsafe_allow_html=True)

            st.metric("Imported Sources", len(imported_sources))

            st.metric(
                "Next Submit Trigger",
                fmt_value(submission_policy.get("next_submit_score")),
                help="Any valid pack at or above this public-score line should be submitted immediately.",
            )

            st.markdown("<div style='margin-top: 1rem;'><strong>Operator Note</strong></div>", unsafe_allow_html=True)
            st.code((operator_note().strip() or "No persistent steering note saved.")[:900], language="text")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

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

    with tab_portal:
        st.markdown('<div class="panel"><h4>Unified Web Portal</h4><div class="panel-copy">The cinematic website and this local ops console now sit in one workflow. Launch the public-facing surface, secure entry, or the web command deck without leaving the dashboard.</div></div>', unsafe_allow_html=True)
        portal_view = st.radio(
            "Website surface",
            options=["Landing", "Secure Entry", "Web Command Deck"],
            horizontal=True,
            key="portal_surface_view",
        )
        portal_map = {
            "Landing": f"{WEBSITE_BASE_URL}/index.html",
            "Secure Entry": f"{WEBSITE_BASE_URL}/auth.html",
            "Web Command Deck": f"{WEBSITE_BASE_URL}/app.html",
        }
        portal_url = portal_map[portal_view]
        portal_links = st.columns(3)
        portal_links[0].markdown(f"[Open Landing]({WEBSITE_BASE_URL}/index.html)")
        portal_links[1].markdown(f"[Open Secure Entry]({WEBSITE_BASE_URL}/auth.html)")
        portal_links[2].markdown(f"[Open Web Command Deck]({WEBSITE_BASE_URL}/app.html)")
        if website_portal_available(WEBSITE_BASE_URL):
            st.caption(f"Embedded from {WEBSITE_BASE_URL} so the website and the live control core stay tied together.")
            components.iframe(portal_url, height=960, scrolling=True)
        else:
            render_portal_fallback(
                daemon=daemon,
                rank_info=rank_info,
                latest_completed=latest_completed,
                latest_report=latest_report,
                submission_policy=submission_policy,
            )

    with tab_node:
        st.markdown('<div class="panel"><h4>Runtime Node</h4><div class="panel-copy">This machine is the local engine. Keep Kaggle auth, model paths, Ollama, and filesystem access here, then let the remote control center steer it from the outside.</div></div>', unsafe_allow_html=True)

        node_cols = st.columns(4)
        node_cols[0].metric("Node Label", node_summary.get("device_label") or "unset")
        node_cols[1].metric("Node ID", str(node_summary.get("node_id") or "unset")[:12])
        node_cols[2].metric("Pairing Code", node_summary.get("pairing_code") or "unset")
        node_cols[3].metric("Kaggle Auth", "READY" if node_summary.get("kaggle_json_exists") else "MISSING")

        config_left, config_right = st.columns([1.0, 1.0])
        with config_left:
            st.subheader("Local Runtime Settings")
            node_label_value = st.text_input(
                "Node label",
                value=str(node_summary.get("device_label") or ""),
                key="runtime_node_label",
            )
            backend_options = ["openclaude", "ollama"]
            current_backend = str(node_summary.get("llm_backend") or "ollama").strip().lower()
            if current_backend not in backend_options:
                backend_options.append(current_backend)
            llm_backend_value = st.selectbox(
                "Primary LLM backend",
                options=backend_options,
                index=backend_options.index(current_backend),
                key="runtime_node_llm_backend",
            )
            remote_url_value = st.text_input(
                "Remote control center URL",
                value=str(node_summary.get("remote_control_url") or ""),
                key="runtime_node_remote_url",
            )
            workspace_root_value = st.text_input(
                "NeuroGolf workspace root",
                value=str(node_summary.get("workspace_root") or ""),
                key="runtime_node_workspace_root",
            )
            kaggle_dir_value = st.text_input(
                "Kaggle config directory",
                value=str(node_summary.get("kaggle_config_dir") or ""),
                key="runtime_node_kaggle_dir",
            )
            ollama_base_value = st.text_input(
                "Ollama base URL",
                value=str(node_summary.get("ollama_base_url") or ""),
                key="runtime_node_ollama_base",
            )
            ollama_model_dir_value = st.text_input(
                "Ollama model directory",
                value=str(node_summary.get("ollama_model_dir") or ""),
                key="runtime_node_ollama_model_dir",
            )
            openclaude_bin_value = st.text_input(
                "OpenClaude binary",
                value=str(node_summary.get("openclaude_bin") or ""),
                key="runtime_node_openclaude_bin",
            )
            openclaude_provider_value = st.text_input(
                "OpenClaude provider",
                value=str(node_summary.get("openclaude_provider") or "ollama"),
                key="runtime_node_openclaude_provider",
            )
            openclaude_default_model_value = st.text_input(
                "OpenClaude default model",
                value=str(node_summary.get("openclaude_default_model") or ""),
                key="runtime_node_openclaude_default_model",
            )
            if st.button("Save Runtime Node Settings", width="stretch"):
                set_action_result(
                    "Runtime node settings saved.",
                    save_runtime_node_settings(
                        device_label=node_label_value,
                        remote_control_url=remote_url_value,
                        workspace_root=workspace_root_value,
                        llm_backend=llm_backend_value,
                        ollama_base_url=ollama_base_value,
                        ollama_model_dir=ollama_model_dir_value,
                        openclaude_bin=openclaude_bin_value,
                        openclaude_provider=openclaude_provider_value,
                        openclaude_default_model=openclaude_default_model_value,
                        kaggle_config_dir=kaggle_dir_value,
                    ),
                )
                st.rerun()

        with config_right:
            st.subheader("Pairing Payload")
            st.json(
                {
                    "node_id": node_summary.get("node_id"),
                    "device_label": node_summary.get("device_label"),
                    "remote_control_url": node_summary.get("remote_control_url"),
                    "pairing_code": node_summary.get("pairing_code"),
                    "pair_token": node_summary.get("pair_token"),
                    "registered_operator_email": node_summary.get("registered_operator_email"),
                }
            )
            st.caption(
                "The remote control center should pair this machine using the node ID plus pair token. Keep the token local and rotate it whenever you rebind the device."
            )
            operator_email_value = st.text_input(
                "Bound operator email",
                value=str(node_summary.get("registered_operator_email") or ""),
                key="runtime_node_operator_email",
                placeholder="subham.choudhury11438@gmail.com",
            )
            if st.button("Save Operator Binding", width="stretch"):
                set_action_result(
                    "Runtime node operator binding saved.",
                    save_runtime_node_operator_binding(operator_email_value),
                )
                st.rerun()
            if node_summary.get("paired_at"):
                st.caption(f"Paired at `{node_summary.get('paired_at')}`")
            rotate_cols = st.columns([1, 1])
            if rotate_cols[0].button("Rotate Pairing Token", width="stretch"):
                set_action_result(
                    "Runtime node pairing token rotated.",
                    rotate_runtime_node_pairing_token(),
                )
                st.rerun()
            if rotate_cols[1].button("Run Runtime Node Healthcheck", width="stretch"):
                set_action_result(
                    "Runtime node healthcheck finished.",
                    runtime_node_healthcheck(),
                )
                st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("Kaggle Credentials")
        st.caption(
            f"Local Kaggle auth is stored under `{node_summary.get('kaggle_json_path', 'n/a')}` so the cloud portal never needs your competition key."
        )
        kaggle_cols = st.columns([1, 1])
        kaggle_username_value = kaggle_cols[0].text_input(
            "Kaggle username",
            value=str(node_summary.get("kaggle_username") or ""),
            key="runtime_node_kaggle_username",
        )
        kaggle_key_value = kaggle_cols[1].text_input(
            "Kaggle key",
            value="",
            type="password",
            key="runtime_node_kaggle_key",
            placeholder="Paste your Kaggle API key here",
        )
        kaggle_actions = st.columns([1, 1, 2])
        if kaggle_actions[0].button("Save Kaggle Auth", width="stretch"):
            if not kaggle_username_value.strip() or not kaggle_key_value.strip():
                st.error("Enter both a Kaggle username and key before saving.")
            else:
                set_action_result(
                    "Kaggle credentials saved.",
                    save_runtime_node_kaggle_credentials(
                        username=kaggle_username_value,
                        key=kaggle_key_value,
                        config_dir=kaggle_dir_value,
                    ),
                )
                st.rerun()
        if kaggle_actions[1].button("Re-run Generic Healthcheck", width="stretch"):
            set_action_result("Healthcheck finished.", healthcheck())
            st.rerun()
        kaggle_actions[2].caption(
            "Installed-app flow: save Kaggle auth locally, point to your Ollama models, then pair this node to the remote site for telemetry and commands."
        )

        path_cols = st.columns(3)
        path_cols[0].metric("Workspace", "FOUND" if node_summary.get("workspace_exists") else "MISSING")
        path_cols[1].metric("LLM Backend", str(node_summary.get("llm_backend") or "ollama").upper())
        path_cols[2].metric("Remote Center", "SET" if node_summary.get("remote_control_url") else "UNSET")
        path_cols[2].caption(str(node_summary.get("remote_control_url") or "No remote control URL configured."))

        llm_runtime_cols = st.columns(3)
        llm_runtime_cols[0].metric("Ollama Models Dir", "FOUND" if node_summary.get("ollama_model_dir_exists") else "UNSET")
        llm_runtime_cols[1].metric("OpenClaude Binary", "READY" if node_summary.get("openclaude_bin_exists") else "MISSING")
        llm_runtime_cols[2].metric("OpenClaude Model", str(node_summary.get("openclaude_default_model") or "unset"))
        llm_runtime_actions = st.columns(2)
        if llm_runtime_actions[0].button("Probe Local LLM Stack", width="stretch"):
            set_action_result("Runtime node healthcheck finished.", runtime_node_healthcheck())
            st.rerun()
        if llm_runtime_actions[1].button("Launch OpenClaude", width="stretch"):
            set_action_result(
                "OpenClaude launch requested.",
                launch_openclaude_session(model=str(node_summary.get("openclaude_default_model") or "")),
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("Node Bridge")
        bridge_daemon = bridge_summary.get("bridge_daemon", {}) if isinstance(bridge_summary.get("bridge_daemon"), dict) else {}
        bridge_cols = st.columns(4)
        bridge_cols[0].metric("Bridge Repo", bridge_summary.get("repo") or "unset")
        bridge_cols[1].metric("Bridge Branch", bridge_summary.get("branch") or "unset")
        bridge_cols[2].metric("Token", "READY" if bridge_summary.get("token_available") else "MISSING")
        bridge_cols[3].metric("Remote Nodes", len(remote_nodes))

        bridge_status_cols = st.columns(4)
        bridge_status_cols[0].metric("Bridge Loop", "ONLINE" if bridge_daemon.get("running") else "OFFLINE")
        bridge_status_cols[1].metric("Bridge PID", bridge_daemon.get("pid") or "n/a")
        bridge_status_cols[2].metric("Bridge Interval", f"{bridge_summary.get('loop_seconds', 90)}s")
        bridge_status_cols[3].metric("Bridge Status Age", fmt_value(bridge_daemon.get("status_age_seconds"), digits=0))
        if bridge_daemon.get("status_updated_at"):
            st.caption(
                f"Bridge loop last synced at `{bridge_daemon.get('status_updated_at')}` with stale=`{bridge_daemon.get('status_stale', False)}`."
            )

        bridge_interval = st.number_input(
            "Bridge loop interval (seconds)",
            min_value=15,
            max_value=3600,
            step=15,
            value=int(bridge_summary.get("loop_seconds", 90) or 90),
            key="runtime_bridge_loop_seconds",
        )
        bridge_service_actions = st.columns(3)
        if bridge_service_actions[0].button("Start Bridge Loop", width="stretch"):
            set_action_result(
                "Runtime Node bridge loop start requested.",
                start_bridge_daemon(interval_seconds=int(bridge_interval)),
            )
            st.rerun()
        if bridge_service_actions[1].button("Restart Bridge Loop", width="stretch"):
            set_action_result(
                "Runtime Node bridge loop restart requested.",
                restart_bridge_daemon_async(interval_seconds=int(bridge_interval)),
            )
            st.rerun()
        if bridge_service_actions[2].button("Stop Bridge Loop", width="stretch"):
            set_action_result(
                "Runtime Node bridge loop stop requested.",
                stop_bridge_daemon(),
            )
            st.rerun()

        bridge_actions = st.columns(3)
        if bridge_actions[0].button("Publish Heartbeat", width="stretch"):
            set_action_result(
                "Runtime node heartbeat published.",
                publish_remote_bridge_heartbeat(),
            )
            st.rerun()
        if bridge_actions[1].button("Poll Remote Command", width="stretch"):
            set_action_result(
                "Remote bridge command poll finished.",
                process_remote_bridge_command(),
            )
            st.rerun()
        if bridge_actions[2].button("Heartbeat + Poll", width="stretch"):
            heartbeat_result = publish_remote_bridge_heartbeat()
            command_result = process_remote_bridge_command()
            set_action_result(
                "Runtime node bridge roundtrip finished.",
                {
                    "ok": bool(heartbeat_result.get("ok", False)) and bool(command_result.get("ok", True)),
                    "stdout": "\n\n".join(
                        part
                        for part in [
                            heartbeat_result.get("stdout", "").strip(),
                            command_result.get("stdout", "").strip(),
                        ]
                        if part
                    ),
                    "stderr": "\n\n".join(
                        part
                        for part in [
                            heartbeat_result.get("stderr", "").strip(),
                            command_result.get("stderr", "").strip(),
                        ]
                        if part
                    ),
                },
            )
            st.rerun()

    with tab_fleet:
        st.markdown('<div class="panel"><h4>Fleet Bridge</h4><div class="panel-copy">The hosted control center reads node heartbeats from a dedicated GitHub bridge branch and queues commands back to each node. The local node keeps secrets on-device and only polls outward.</div></div>', unsafe_allow_html=True)
        st.caption(
            f"Bridge target: `{bridge_summary.get('repo', 'n/a')}` on branch `{bridge_summary.get('branch', 'n/a')}`"
        )

        if not remote_nodes:
            st.info("No remote nodes have published a heartbeat yet. Use the Runtime Node tab and publish the first heartbeat from the local machine.")
        else:
            node_options = {
                f"{item.get('device_label', 'node')} | {item.get('node_id', 'unknown')}": item
                for item in remote_nodes
            }
            selected_label = st.selectbox("Active node", options=list(node_options.keys()), key="fleet_active_node")
            active_node = node_options[selected_label]
            fleet_top = st.columns(4)
            fleet_top[0].metric("Device", active_node.get("device_label") or "unset")
            fleet_top[1].metric("Phase", active_node.get("phase") or "idle")
            fleet_top[2].metric("Best Public", fmt_value(active_node.get("best_completed_public_score")))
            fleet_top[3].metric("Pending", fmt_value(active_node.get("pending_submission_count"), digits=0))

            action_cols = st.columns(5)
            if action_cols[0].button("Queue Healthcheck", width="stretch"):
                set_action_result(
                    "Remote command queued.",
                    queue_remote_bridge_command(active_node.get("node_id", ""), "run_healthcheck"),
                )
                st.rerun()
            if action_cols[1].button("Queue Sync", width="stretch"):
                set_action_result(
                    "Remote command queued.",
                    queue_remote_bridge_command(active_node.get("node_id", ""), "sync_state", {"history": 10}),
                )
                st.rerun()
            if action_cols[2].button("Queue One Cycle", width="stretch"):
                set_action_result(
                    "Remote command queued.",
                    queue_remote_bridge_command(
                        active_node.get("node_id", ""),
                        "run_single_cycle",
                        {"allow_submit": False, "history": 10, "min_local_delta": 0.0},
                    ),
                )
                st.rerun()
            if action_cols[3].button("Queue Restart", width="stretch"):
                set_action_result(
                    "Remote command queued.",
                    queue_remote_bridge_command(
                        active_node.get("node_id", ""),
                        "restart_daemon",
                        {
                            "allow_submit": True,
                            "history": 10,
                            "min_local_delta": 0.0,
                            "sleep_seconds": 60,
                            "max_pending_submissions": 2,
                        },
                    ),
                )
                st.rerun()
            if action_cols[4].button("Queue Stop", width="stretch"):
                set_action_result(
                    "Remote command queued.",
                    queue_remote_bridge_command(active_node.get("node_id", ""), "stop_daemon", {"reset_state": False}),
                )
                st.rerun()

            note_value = st.text_area(
                "Remote operator note",
                value="",
                key="fleet_remote_note",
                height=120,
                placeholder="Write an operator note that the local node should save before the next cycle.",
            )
            if st.button("Queue Operator Note", width="stretch"):
                set_action_result(
                    "Remote operator note queued.",
                    queue_remote_bridge_command(
                        active_node.get("node_id", ""),
                        "save_operator_note",
                        {"note": note_value},
                    ),
                )
                st.rerun()

            status_left, status_right = st.columns([1.0, 1.0])
            with status_left:
                st.subheader("Heartbeat Payload")
                st.json(active_node)
            with status_right:
                st.subheader("All Nodes")
                st.dataframe(
                    [
                        {
                            "device_label": item.get("device_label"),
                            "node_id": item.get("node_id"),
                            "phase": item.get("phase"),
                            "best_public": item.get("best_completed_public_score"),
                            "pending": item.get("pending_submission_count"),
                            "last_seen_at": item.get("last_seen_at"),
                        }
                        for item in remote_nodes
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with tab_control:
        st.markdown('<div class="panel"><h4>Autonomy Controls</h4><div class="panel-copy">Start, stop, restart, run a single cycle, inject steering, and import Kaggle sources.</div></div>', unsafe_allow_html=True)
        allow_submit = st.checkbox(
            "Allow Kaggle submissions",
            key=control_widget_key("allow_submit"),
        )
        history = st.number_input(
            "History depth",
            min_value=3,
            max_value=100,
            key=control_widget_key("history"),
        )
        delta = st.number_input(
            "Minimum local delta floor (0 = submit any verified gain)",
            min_value=0.0,
            max_value=1000.0,
            step=0.5,
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

        # Adaptive learning controls
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="panel"><h4>Adaptive Learning Loop</h4><div class="panel-copy">Turn recent cycle evidence into planner guidance, weak-task focus lists, and safer automatic operating notes.</div></div>', unsafe_allow_html=True)

        si_enabled = is_self_improvement_enabled()
        si_status = get_improvement_status()

        si_col1, si_col2, si_col3 = st.columns([1, 1, 2])
        with si_col1:
            if st.button("Enable Adaptive Learning" if not si_enabled else "Adaptive Learning Active", width="stretch", disabled=si_enabled):
                enable_self_improvement()
                st.success("Adaptive learning enabled. New cycles will write automatic guidance from recent results.")
                st.rerun()
        with si_col2:
            if st.button("Disable Adaptive Learning" if si_enabled else "Adaptive Learning Inactive", width="stretch", disabled=not si_enabled):
                disable_self_improvement()
                st.warning("Adaptive learning disabled.")
                st.rerun()
        with si_col3:
            st.metric("Learning Updates", si_status["recent_improvements"])

        if si_status['last_attempt']:
            last = si_status['last_attempt']
            st.caption(f"Last update: {last.get('timestamp', 'unknown')} | Applied: {last.get('applied', False)} | {last.get('reason', 'N/A')[:80]}...")
        note_preview = str((si_status.get("state") or {}).get("operator_note", "")).strip()
        if note_preview:
            st.code(note_preview, language="text")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="panel"><h4>Core Maintenance</h4><div class="panel-copy">Archive generated caches, old cycle reports, and source backups without deleting evidence or source files.</div></div>', unsafe_allow_html=True)
        clutter = generated_clutter_summary()
        maint_cols = st.columns([1, 1, 2])
        maint_cols[0].metric("Declutter Items", clutter.get("count", 0))
        maint_cols[1].metric("Archive Size", format_bytes(clutter.get("total_bytes", 0)))
        maint_cols[2].caption(", ".join(f"{kind}: {count}" for kind, count in clutter.get("by_kind", {}).items()) or "No generated clutter detected.")
        maint_buttons = st.columns(2)
        if maint_buttons[0].button("Preview Declutter Archive", width="stretch"):
            set_action_result("Declutter preview complete.", archive_generated_clutter(dry_run=True))
        if maint_buttons[1].button("Archive Generated Clutter", width="stretch"):
            set_action_result("Generated clutter archived.", archive_generated_clutter(dry_run=False))
            st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="panel"><h4>Teacher Distillation Track</h4><div class="panel-copy">Fine-tune or steer a large teacher for rule discovery, then distill each task into the smallest legal static ONNX graph.</div></div>', unsafe_allow_html=True)
        distill = read_distillation_status()
        plan = distill.get("plan", {})
        ft_policy = plan.get("fine_tune_policy", {}) if isinstance(plan.get("fine_tune_policy"), dict) else {}
        dist_cols = st.columns(4)
        dist_cols[0].metric("Teacher Signals", distill.get("teacher_signal_count", 0))
        dist_cols[1].metric("Stages", f"{distill.get('ready_or_done_stages', 0)}/{distill.get('stage_count', 0)}")
        dist_cols[2].metric("Fine Tune", "ON" if distill.get("fine_tune_enabled") else "OFF")
        dist_cols[3].metric("Teacher", distill.get("teacher_model") or "unset")
        teacher_model = st.text_input("Teacher model", value=str(distill.get("teacher_model") or ""), key="distill_teacher_model")
        dataset_path = st.text_input("Fine-tune dataset path", value=str(ft_policy.get("dataset_path", "")), key="distill_dataset_path")
        adapter_path = st.text_input("Adapter/output path", value=str(ft_policy.get("output_adapter_path", "")), key="distill_adapter_path")
        ft_enabled = st.checkbox("Enable fine-tune policy", value=bool(ft_policy.get("enabled", False)), key="distill_ft_enabled")
        if st.button("Save Distillation Settings", width="stretch"):
            set_action_result(
                "Distillation settings saved.",
                save_distillation_settings(
                    enabled=bool(ft_enabled),
                    dataset_path=dataset_path,
                    output_adapter_path=adapter_path,
                    teacher_model=teacher_model,
                ),
            )
            st.rerun()

        top_actions = st.columns(4)
        if top_actions[0].button("Start Autonomy", width="stretch"):
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
        if top_actions[1].button("Stop And Reset", width="stretch"):
            set_action_result(
                "Autonomy stopped. Learning state cleared; the next start will begin fresh.",
                stop_daemon(reset_state=True)
            )
            st.rerun()
        if top_actions[2].button("Run One Cycle", width="stretch"):
            set_action_result(
                "One-cycle run finished.",
                run_single_cycle(allow_submit=allow_submit, history=int(history), min_local_delta=float(delta)),
            )
            st.rerun()
        if top_actions[3].button("Sign Out And Stop", width="stretch"):
            set_action_result("System stopped and control console locked.", shutdown_and_logout())
            st.rerun()

        lower_actions = st.columns(3)
        if lower_actions[0].button("Sync Kaggle State", width="stretch"):
            persist_runtime_preferences(
                allow_submit=bool(allow_submit),
                history=int(history),
                min_local_delta=float(delta),
                sleep_seconds=int(sleep_seconds),
                max_pending_submissions=int(max_pending),
            )
            set_action_result("State sync finished.", sync_state(history=int(history)))
            st.rerun()
        if lower_actions[1].button("Run Healthcheck", width="stretch"):
            set_action_result("Healthcheck finished.", healthcheck())
            st.rerun()
        if lower_actions[2].button("Restart Daemon", width="stretch"):
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
            placeholder="Keep seed-preserving swaps near the strongest valid pack, avoid repeating failed families, and submit any verified public gain immediately.",
        )
        note_buttons = st.columns(3)
        if note_buttons[0].button("Save Operator Note", width="stretch"):
            save_operator_note(note_value)
            st.success("Operator note saved.")
        if note_buttons[1].button("Clear Operator Note", width="stretch"):
            save_operator_note("")
            st.success("Operator note cleared.")
        if note_buttons[2].button("Apply Operator Note", width="stretch"):
            save_operator_note(note_value)
            st.success("Operator note saved for future cycles.")

        st.subheader("Kaggle Source Intake")
        kaggle_seed_text = st.text_area(
            "Paste Kaggle URLs or CLI lines",
            value=st.session_state.get("kaggle_intake_text", ""),
            height=150,
            placeholder="https://www.kaggle.com/code/jiweiliu/kaggle-agent-lb794-inference\nkaggle kernels pull jonathanchan/ngc36-constraint-smart-logic-mix-blending\nhttps://www.kaggle.com/datasets/thisray/neurogolf-4743-93-submission-task-table",
            key="kaggle_intake_text",
        )
        intake_buttons = st.columns(2)
        if intake_buttons[0].button("Import Kaggle Sources", width="stretch"):
            set_action_result("Kaggle source fetch finished.", fetch_kaggle_targets(kaggle_seed_text))
            st.cache_data.clear()
            st.rerun()
        if intake_buttons[1].button("Show Import Location", width="stretch"):
            st.info("Imported sources appear below and in the NeuroGolf scratch gui_imports folder.")

        if imported_sources:
            st.subheader("Imported Seed Controls")
            control_rows = []
            for item in imported_sources:
                inferred_score = item.get("claimed_public_score")
                control_rows.append(
                    {
                        "use_as_seed": bool(item.get("usable_as_seed")),
                        "seed_label": item.get("seed_label", ""),
                        "manual_public_score": inferred_score if item.get("score_source") == "manual" else None,
                        "current_score": inferred_score,
                        "score_source": item.get("score_source", ""),
                        "task_file_count": item.get("task_file_count", 0),
                        "name": item.get("name", ""),
                        "note": item.get("control_note", ""),
                    }
                )
            edited_sources = st.data_editor(
                control_rows,
                width="stretch",
                hide_index=True,
                key="imported_seed_controls_editor",
                column_config={
                    "use_as_seed": st.column_config.CheckboxColumn("Use", help="Deselect to remove this seed from autonomous planning."),
                    "seed_label": st.column_config.TextColumn("Seed", disabled=True),
                    "manual_public_score": st.column_config.NumberColumn("Manual score", min_value=0.0, step=0.01, format="%.2f"),
                    "current_score": st.column_config.NumberColumn("Current score", disabled=True, format="%.2f"),
                    "score_source": st.column_config.TextColumn("Score source", disabled=True),
                    "task_file_count": st.column_config.NumberColumn("Tasks", disabled=True),
                    "name": st.column_config.TextColumn("Import", disabled=True),
                    "note": st.column_config.TextColumn("Note"),
                },
            )
            control_buttons = st.columns(2)
            if control_buttons[0].button("Save Seed Controls", width="stretch"):
                edited_records = edited_sources.to_dict("records") if hasattr(edited_sources, "to_dict") else edited_sources
                rows_to_save = []
                for row in edited_records:
                    rows_to_save.append(
                        {
                            "seed_label": row.get("seed_label", ""),
                            "manual_public_score": row.get("manual_public_score"),
                            "disabled": not bool(row.get("use_as_seed")),
                            "note": row.get("note", ""),
                        }
                    )
                set_action_result("Seed controls saved.", save_imported_seed_controls(rows_to_save))
                st.cache_data.clear()
                st.rerun()
            if control_buttons[1].button("Refresh Seed Table", width="stretch"):
                st.cache_data.clear()
                st.rerun()
            st.caption("Manual scores override scores inferred from notebook names. Deselected seeds remain visible here, but the planner will stop using them.")
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
        if profile_buttons[2].button("Save Profile", width="stretch"):
            update_env_settings(env_updates_for_profile(profile_key))
            st.success("Profile written to .env.")
        if profile_buttons[3].button("Apply Profile And Restart", width="stretch"):
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
        if manual_buttons[0].button("Save Manual Role Overrides", width="stretch"):
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
        st.markdown('<div class="panel"><h4>Competition Telemetry</h4><div class="panel-copy">Competition state, leaderboard standing, recent submissions, and output manifests.</div></div>', unsafe_allow_html=True)

        # Metric V3 Rules Alert Banner
        metric_v3 = state.get("metric_v3_rules", {})
        if metric_v3:
            st.markdown(
                f"""
                <div class="alert-banner">
                    <div class="alert-title">April 28, 2026 Metric Update (V3) - ACTIVE</div>
                    <div class="alert-text">
                        <strong>Dynamic shapes now yield ZERO points.</strong> All networks must have statically-defined shapes.
                        <strong>Constant values now correctly count</strong> toward parameter contributions.
                        Memory footprint is calculated as sum of bytes for all static shapes (excluding input/output layers).
                        Invalid networks (failed shape inference, symbolic dims) score ZERO.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
        st.markdown('<div class="panel"><h4>Cycle Reports</h4><div class="panel-copy">Recent autonomy cycles, submit decisions, and the live daemon trace.</div></div>', unsafe_allow_html=True)
        report_rows = [
            {
                "path": item.get("_path"),
                "started_at": item.get("started_at"),
                "action": (item.get("plan") or {}).get("action"),
                "target": (item.get("plan") or {}).get("target") or (item.get("plan") or {}).get("seed_label"),
                "variant": (item.get("plan") or {}).get("variant_hint") or (item.get("plan") or {}).get("mode"),
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
                    format_func=lambda idx: f"{reports[idx].get('started_at', 'unknown')} | {((reports[idx].get('plan') or {}).get('target') or (reports[idx].get('plan') or {}).get('seed_label', 'n/a'))}",
                )
                st.json(reports[selected])
            else:
                st.info("No cycle reports found yet.")
        with log_col:
            st.subheader("Latest Daemon Log")
            st.code(tail_log(140) or "No daemon log yet.", language="text")

    with tab_arc:
        render_arc_visualizer()

    with tab_chat:
        st.markdown('<div class="panel"><h4>Operator Console</h4><div class="panel-copy">Use the Fast Operator for quick answers, or switch to the Orchestrator when you want deeper strategy synthesis grounded in live system state.</div></div>', unsafe_allow_html=True)
        ensure_chat_state()
        chat_context = build_orchestrator_chat_context(state, status, reports, imported_sources, current_role_map())
        chat_target = st.radio(
            "Chat target",
            options=["operator_fast", "orchestrator"],
            format_func=lambda value: "Fast Operator" if value == "operator_fast" else "Orchestrator",
            horizontal=True,
        )

        chat_actions = st.columns(2)
        if chat_actions[0].button("Clear Chat", width="stretch"):
            st.session_state.pop("orchestrator_chat_messages", None)
            ensure_chat_state()
            st.rerun()
        if chat_actions[1].button("Show Context Note", width="stretch"):
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
