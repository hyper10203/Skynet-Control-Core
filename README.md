# ARC Agent System

Local Ollama-based ARC and NeuroGolf helper with one front-door orchestrator plus a lightweight fast operator lane.

## Main Idea

You normally talk to one model: the orchestrator.

There is also a lighter operator-facing chat lane for quick questions and control work.

That orchestrator decides when to call:
- `deepseek-r1` for careful rule discovery
- `mistral` for an alternate hypothesis
- `qwen2.5-coder` for Python solver generation
- `llama3` for critique and edge-case checking
- `memory/patterns.json` for pattern retrieval

## Model Roles

- Orchestrator: `qwen2.5`
- Fast operator link: `qwen2.5`
- Primary reasoning: `deepseek-r1`
- Alternate reasoning: `mistral`
- Third reasoning: `llama3`
- Coding: `qwen2.5-coder`
- Critic: `llama3`

## Recommended Context Windows

These are the defaults already wired in:

- Orchestrator: `8192`
- Fast operator link: `6144`
- Primary reasoning: `12288`
- Alternate reasoning: `8192`
- Third reasoning: `8192`
- Coding: `8192`
- Critic: `6144`

If your machine is tight on RAM or VRAM, cut them roughly in half.
Short response caps are also set by default so the models stay practical in local Ollama.

## Cline Settings

If you want one top-level AI in Cline, set:

- Provider: `Ollama`
- Base URL: `http://localhost:11434`
- Model: `qwen2.5`

Reason: `qwen2.5` should be the one you talk to. `qwen2.5-coder` stays as the internal coding specialist inside this project.

## Data Layout

The project is already wired to the NeuroGolf task set on `E:` through:

- [data/neurogolf-2026](E:/LLM/New%20project_main/arc-agent-system/data/neurogolf-2026)

That junction points to:

- [main NeuroGolf data](E:/LLM/New%20project_main/data/neurogolf-2026)

To keep local Ollama responsive, the default prompt pack uses:

- `2` train examples
- `0` test examples
- `0` arc-gen examples

## Quick Start

From [arc-agent-system](E:/LLM/New%20project_main/arc-agent-system):

1. Install dependencies:
   - `powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1`
2. Make sure Ollama is running:
   - `ollama serve`
3. Check models:
   - `python .\healthcheck.py`
4. Run a small NeuroGolf batch:
   - `powershell -ExecutionPolicy Bypass -File .\run_neurogolf.ps1 -Limit 5 -CritiqueRounds 2`

You can also run directly:

- `python .\main.py`
- `python .\main.py --limit 20 --critique-rounds 1`

## Outputs

- Generated solver code: [outputs/generated](E:/LLM/New%20project_main/arc-agent-system/outputs/generated)
- Per-task reports: [outputs/reports](E:/LLM/New%20project_main/arc-agent-system/outputs/reports)
- Pattern memory: [memory/patterns.json](E:/LLM/New%20project_main/arc-agent-system/memory/patterns.json)
- Submission state: [outputs/submission_state.json](E:/LLM/New%20project_main/arc-agent-system/outputs/submission_state.json)

## Submission Automation

This project can also submit to Kaggle and poll the result by itself.

Prereqs:
- Kaggle API credentials already configured on the machine
- `submission.zip` ready

Example:

- `python .\submit_to_kaggle.py --file "E:\path\to\submission.zip" --estimated-score 4800`
- `powershell -ExecutionPolicy Bypass -File .\submit_if_ready.ps1 -SubmissionFile "E:\path\to\submission.zip" -EstimatedScore 4800`

Default behavior:
- competition: `neurogolf-2026`
- minimum jump required before submit: `250` points
- it stores the last submission state and estimate

Useful flags:
- `--threshold 300`
- `--force`
- `--dry-run`
- `--estimate-file path\to\score.json`

## GUI Control Center

There is now a full local dashboard for the ARC + NeuroGolf system.

Main file:
- [dashboard.py](E:/LLM/New%20project_main/arc-agent-system/dashboard.py)
- [CONTROL_CENTER_README.md](E:/LLM/New%20project_main/arc-agent-system/CONTROL_CENTER_README.md)

Launcher scripts:
- [start_dashboard.ps1](E:/LLM/New%20project_main/arc-agent-system/start_dashboard.ps1)
- [stop_dashboard.ps1](E:/LLM/New%20project_main/arc-agent-system/stop_dashboard.ps1)

What the GUI shows:
- daemon start and stop controls
- live progress bars and current phase
- latest submission score, best public score, and current team rank
- recent cycle reports and live daemon logs
- operator prompt injection for steering the planner
- model role assignments, installed Ollama models, and upgrade profiles
- a fast operator chat lane plus the heavier orchestrator chat lane

The dashboard is designed as a sci-fi control room and uses a few web-served assets for styling and branding.
Runtime control values like sleep time, pending submission cap, and submission permission are now persisted so they do not drift after a submission or daemon restart.

## Autonomous NeuroGolf Mode

The agent system can now sync the real NeuroGolf workspace, ingest its manifests and Kaggle results, and run a seed-preserving optimization cycle on its own.

Important behavior:
- it starts from strong public seeds instead of a full scorer-wide rebuild
- it preserves `task000`
- it prefers `processable_best` swaps over aggressive invalid fills
- it stores the synced workspace snapshot in [memory/neurogolf_state.json](E:/LLM/New%20project_main/arc-agent-system/memory/neurogolf_state.json)

Useful commands:

- `python .\sync_neurogolf_artifacts.py --history 10`
- `python .\autonomous_neurogolf.py --sync-only --history 10`
- `powershell -ExecutionPolicy Bypass -File .\run_autonomous_neurogolf.ps1`
- `powershell -ExecutionPolicy Bypass -File .\run_autonomous_neurogolf.ps1 -AllowSubmit`

Autonomy reports are written to:

- [outputs/neurogolf](E:/LLM/New%20project_main/arc-agent-system/outputs/neurogolf)

## Prompt Design

Each model now has a fixed role prompt:

- Qwen orchestrates and chooses who to wake up
- DeepSeek focuses on exact symbolic ARC rules
- Mistral searches for a different rule family
- Qwen-Coder turns final reasoning into short numpy code
- LLaMA3 tries to break the code and surface edge cases

That separation is deliberate so the models do less duplicate work.

Direct-first routing is enabled by default:

- if Qwen already has a concrete rule, it keeps the job local
- it only wakes DeepSeek or Mistral when it genuinely wants extra help
