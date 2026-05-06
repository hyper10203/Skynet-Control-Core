# AxiomGraph Operations Core

<p align="center">
  <img src="assets/axiomgraph-operations-core-banner.svg" alt="AxiomGraph Operations Core banner" width="100%" />
</p>

<p align="center">
  <strong>Local-first ARC and NeuroGolf orchestration with Ollama, Kaggle automation, and a professional operations console.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-0f172a?style=for-the-badge&logo=python&logoColor=ffd43b" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/streamlit-control%20center-0f172a?style=for-the-badge&logo=streamlit&logoColor=ff4b4b" alt="Streamlit control center" />
  <img src="https://img.shields.io/badge/ollama-multi--model-0f172a?style=for-the-badge" alt="Ollama multi-model" />
  <img src="https://img.shields.io/badge/kaggle-neurogolf%20automation-0f172a?style=for-the-badge&logo=kaggle&logoColor=20beff" alt="Kaggle NeuroGolf automation" />
</p>

<p align="center">
  <a href="https://github.com/hyper10203/Skynet-Control-Core/actions/workflows/ci.yml">
    <img src="https://github.com/hyper10203/Skynet-Control-Core/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  </a>
</p>

## What This Is

AxiomGraph Operations Core is now tuned as a private owner-operated AI stack for:

- ARC-style symbolic reasoning
- NeuroGolf submission optimization
- multi-model orchestration through Ollama
- Kaggle source intake and score-aware submission workflows
- live monitoring through a private Streamlit control surface

You talk to one front-door system. It decides when to keep work local and when to wake specialist models.

## Why It Exists

Most local agent stacks either feel like a toy chatbot or a loose pile of scripts.

This repo is built to be a proper control surface:

- one orchestrator lane for high-level decisions
- one lightweight fast operator lane for quick questions
- three reasoning lanes for disagreement and synthesis
- one coder lane for implementation
- one critic lane for attack and failure analysis
- a persistent memory layer
- a daemon that can keep working while you step away

## Core Features

- Stylish Streamlit dashboard with live daemon status, rank, scores, logs, reports, and ARC visualization
- Autonomous NeuroGolf loop with seed-preserving pack strategies and Kaggle-aware submission gating
- Kaggle intake box that can pull notebooks and datasets directly from pasted links or CLI snippets
- Model-role editor with profile loading, installed-model detection, and context auto-fill
- Direct orchestrator chat plus a separate fast operator chat lane
- `.env`-backed runtime settings so loop timing and submission controls stay stable across restarts
- Runtime Node tab for local-only Kaggle auth, Ollama/model paths, pairing code rotation, and installed-engine health checks

## Runtime Split

The system is now moving toward a two-part shape:

- **AxiomGraph Runtime Node**: the installed local engine that owns Kaggle auth, Ollama, local model paths, the NeuroGolf workspace, daemon execution, and submissions
- **AxiomGraph Control Center**: the remote-facing website and hosted portal that owns landing/login/device views, telemetry, command history, and operator control

This keeps the powerful parts local while still giving you a clean remote surface that is meant for one owner, not a public multi-user product.

### First bridge implementation

The first private remote-control bridge uses a dedicated GitHub branch as a transport layer:

- the local Runtime Node publishes heartbeat JSON to `runtime-node-bridge`
- the hosted Control Center reads node heartbeats from that branch
- the hosted Control Center can queue JSON commands back to each node
- the local node polls outward, executes allowed commands, and acknowledges the result

That gives us outbound-only private remote control without exposing your laptop directly to the internet.

The local Runtime Node can also keep a small bridge loop running in the background. That loop publishes fresh heartbeats, polls for remote commands, and lets the hosted Control Center steer the machine even when the Streamlit dashboard is closed.

For day-to-day local use, start the full machine-facing surface with:

- `start_runtime_node.ps1` for the dashboard + bridge
- `start_runtime_node.ps1 -StartAutonomy -AllowSubmit` when you want the local solver loop live immediately
- `stop_runtime_node.ps1` to stop the dashboard, bridge, and local daemon together

### OpenClaude on top of Ollama

This Runtime Node can also use OpenClaude as the local prompting layer while still routing model calls through Ollama.

- official OpenClaude install: `npm install -g @gitlawb/openclaude`
- local launch helper: `start_openclaude_ollama.ps1`
- recommended config:
  - `LLM_BACKEND=openclaude`
  - `OPENCLAUDE_PROVIDER=ollama`
  - `OPENCLAUDE_DEFAULT_MODEL=qwen2.5-coder:latest`

Right now the safest competition path is still:

- `LLM_BACKEND=ollama` for the autonomous solver loop
- OpenClaude as an optional local operator surface for short/manual sessions

Why: short probes can be faster through OpenClaude, but the local-first solver path still needs the raw Ollama transport as the most trustworthy baseline while longer planner runs are being hardened.

## System Layout

```mermaid
flowchart LR
    User["Operator"] --> GUI["AxiomGraph Operations Core GUI"]
    GUI --> Orch["Orchestrator"]
    GUI --> Fast["Fast Operator Link"]
    Orch --> R1["Primary Reasoner"]
    Orch --> R2["Secondary Reasoner"]
    Orch --> R3["Tertiary Reasoner"]
    Orch --> Coder["Coder"]
    Orch --> Critic["Critic"]
    Orch --> Memory["Pattern Memory"]
    Orch --> Kaggle["Kaggle Intake + Submission Loop"]
    Kaggle --> NG["NeuroGolf Workspace"]
    NG --> Reports["Reports / Logs / Manifests"]
    Reports --> GUI
```

## Model Team

| Role | Purpose |
|---|---|
| `orchestrator` | Main planner and synthesis layer |
| `operator_fast` | Lightweight operator-facing chat and control helper |
| `reasoning_primary` | Careful symbolic strategist |
| `reasoning_secondary` | Alternative hypothesis generator |
| `reasoning_tertiary` | Overlooked-leverage and resource-fusion strategist |
| `coder` | Implementation specialist |
| `critic` | Counterexample and risk hunter |

## Repository Highlights

- [dashboard.py](dashboard.py): main Streamlit control center
- [autonomous_neurogolf.py](autonomous_neurogolf.py): 24/7 NeuroGolf daemon logic
- [core/](core): orchestration, config, prompts, env handling, runtime control
- [agents/](agents): role-specific agent wrappers
- [CONTROL_CENTER_README.md](CONTROL_CENTER_README.md): full button-by-button GUI guide
- [data/training](data/training): ARC task payloads used by the local stack

## Quick Start

1. Install dependencies with `bootstrap.ps1`
2. Make sure Ollama is running
3. Run `healthcheck.py`
4. Start the dashboard
5. Use the GUI to load roles, control the daemon, and inspect NeuroGolf cycles

Main launcher scripts:

- [bootstrap.ps1](bootstrap.ps1)
- [start_dashboard.ps1](start_dashboard.ps1)
- [stop_dashboard.ps1](stop_dashboard.ps1)
- [run_autonomous_neurogolf.ps1](run_autonomous_neurogolf.ps1)
- [start_autonomous_neurogolf.ps1](start_autonomous_neurogolf.ps1)
- [stop_autonomous_neurogolf.ps1](stop_autonomous_neurogolf.ps1)

## Deployment

This project is deployable, but it is best treated as a **self-hosted AI operations system**, not a generic stateless web app.

Included deployment assets:

- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [.streamlit/config.toml](.streamlit/config.toml)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [.github/workflows/ci.yml](.github/workflows/ci.yml)

Best-fit deployment path:

- run the dashboard in Docker
- keep Ollama running on the host or on a reachable model server
- mount a writable NeuroGolf workspace into the container
- access it over LAN or through Tailscale from any device while your laptop stays on

Fast start:

```bash
cp .env.example .env
docker compose up --build -d
```

Then open `http://localhost:8501`.

For remote control from other devices:

- on the same network, use the LAN URL shown in the dashboard
- across the internet, use Tailscale and open the Tailscale URL shown in the dashboard
- keep `SKYNET_UI_PASSWORD` set in `.env`

## Dashboard Surface

The control center includes:

- live daemon and telemetry overview
- operations controls for starting, stopping, syncing, and running cycles
- model profile management
- NeuroGolf leaderboard and submission state
- ARC visualizer
- report inspection
- direct chat with the orchestrator or fast operator link

See [CONTROL_CENTER_README.md](CONTROL_CENTER_README.md) for the full guide.

## NeuroGolf Automation Notes

The NeuroGolf side is intentionally conservative:

- preserve strong accepted public seeds where possible
- treat Kaggle score as final truth
- avoid trusting local scorer deltas too much
- keep `task000` unless there is strong evidence to remove it
- prefer controlled swaps over full rebuilds

## Deployment Notes

This repo is set up to be GitHub-safe:

- `.env` is not committed
- live daemon logs and local runtime state are ignored
- the tracked codebase includes `.env.example` for setup

If you deploy this elsewhere, copy `.env.example` to `.env` and adjust model and path settings for the target machine.

## Project Philosophy

This is not trying to be a generic chatbot.

It is a working local operations system for ARC and NeuroGolf:

- hypothesis-driven
- score-aware
- tool-using
- memory-backed
- operator-steerable

If your goal is to leave the machine running and remotely steer your own local model team without handing secrets to the cloud, this repo is built for that.
