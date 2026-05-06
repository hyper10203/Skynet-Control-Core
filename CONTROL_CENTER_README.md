# AxiomGraph Operations Core Guide

This file explains the local GUI, the autonomy controls, the model team, and the Kaggle intake workflow.

## What this system is

`AxiomGraph Operations Core` is the Streamlit dashboard for the ARC and NeuroGolf agent stack.

It controls:

- the autonomous NeuroGolf daemon
- the local Ollama model team
- Kaggle source intake
- ARC task visualization
- model role and context configuration

The dashboard refreshes itself every 5 minutes.
If `SKYNET_UI_PASSWORD` is set, the whole control center stays locked until you authenticate.

## Main tabs

### Overview

Shows:

- current daemon status
- current rank
- latest completed public score
- current run phase
- live daemon log tail
- score trend over time
- operator note preview
- remote LAN and Tailscale access URLs

### Control

Used for runtime actions:

- starting and stopping the autonomy daemon
- running one single cycle manually
- syncing Kaggle state
- health checks
- restarting the daemon with current settings
- editing the persistent operator note
- importing Kaggle notebooks and datasets
- previewing and archiving generated clutter
- managing the teacher-to-student distillation plan

### Models

Used for:

- seeing the currently saved role assignments
- seeing installed Ollama models
- loading a recommended profile into the role editor
- auto-detecting the strongest installed profile
- saving role overrides
- resetting role contexts to model-based defaults

### NeuroGolf

Shows:

- leaderboard rank
- latest score
- pending submissions
- recent submissions
- recent output manifests

### Reports

Shows:

- recent autonomy cycle reports
- submit decisions
- live daemon trace

### ARC Visualizer

Lets you:

- point at an ARC dataset folder
- select a task
- inspect train and test grids
- see rough pattern hints like rotation, transpose, crop, or color remap

### Orchestrator Chat

Lets you:

- talk either to the fast operator link or directly to the orchestrator from the GUI
- ask what experiment to run next
- ask what Kaggle sources to fetch
- ask how to improve rank or submission quality
- keep the conversation grounded in live local state, imported sources, recent reports, and the operator note

## Button guide

### Control buttons

- `Refresh Data`
  - Clears cached dashboard data and refreshes the page immediately.

- `Start Autonomy`
  - Starts the autonomous NeuroGolf daemon with the current control settings.

- `Stop And Reset`
  - Stops the autonomous daemon.

- `Run One Cycle`
  - Runs one autonomy cycle once without waiting for the full background loop.

- `Sync Kaggle State`
  - Re-syncs local NeuroGolf outputs and Kaggle submission history into memory.

- `Run Healthcheck`
  - Runs the model health check against the current Ollama stack.

- `Restart Daemon`
  - Restarts the daemon using the current settings in the control tab.

- `Save Operator Note`
  - Saves the operator note.

- `Clear Operator Note`
  - Clears the operator note.

- `Apply Operator Note`
  - Saves the operator note so future autonomy cycles use it.

- `Import Kaggle Sources`
  - Pulls Kaggle notebooks or datasets from pasted URLs or CLI lines.

- `Show Import Location`
  - Explains where imported Kaggle files appear.

- `Preview Declutter Archive`
  - Shows generated caches, timestamped source backups, and old cycle reports that can be archived.

- `Archive Generated Clutter`
  - Moves generated clutter under the configured archive folder without deleting it.

- `Save Distillation Settings`
  - Saves the big-teacher / small-student workflow settings into `memory/distillation_plan.json`.

### Orchestrator chat buttons

- `Purge Chat Memory`
  - Clears the current chat session inside the dashboard.

- `Inject Live State`
  - Explains that live NeuroGolf and model-team context is already attached to each orchestrator reply.

### Model buttons

- `Load Into Roles`
  - Loads the selected model profile into the editable role fields without saving yet.

- `Save Profile`
  - Saves the selected model profile into `.env`.

- `Apply Profile And Restart`
  - Saves the selected model profile into `.env` and restarts the daemon.

- `Use My Installed Best Models`
  - Detects the strongest profile that matches your locally installed Ollama models and loads it into the role editor.

- `Save Manual Role Overrides`
  - Saves the current role dropdown values and their context lengths into `.env`.

- `Reset Contexts To Model Defaults`
  - Recomputes context lengths from the currently selected model for each role.

## Model team

The system now has six active roles:

- `orchestrator`
  - the top-level planner and merger

- `operator_fast`
  - the lightweight direct chat lane for quick questions, summaries, and operator-facing control work

- `reasoning_primary`
  - the main careful strategist

- `reasoning_secondary`
  - the alternative strategist

- `reasoning_tertiary`
  - the third strategist focused on overlooked leverage, imported Kaggle assets, and validator-safe opportunities

- `coder`
  - the implementation specialist

- `critic`
  - the risk checker and counterexample hunter

## How the team works together

The autonomy engine is designed to combine:

- Kaggle submission history
- accepted public seed knowledge
- imported Kaggle notebooks and datasets
- local manifests and output files
- the operator note
- the model team discussions

The intent is not to let one model dominate too early.

The three strategists produce different views, the coder turns them into concrete experiments, the critic attacks weak ideas, and the orchestrator merges the bundle into one operational plan.

## Kaggle intake format

You can paste any mix of these into the `Kaggle Source Intake` box:

- `https://www.kaggle.com/code/owner/slug`
- `https://www.kaggle.com/datasets/owner/slug`
- `kaggle kernels pull owner/slug`
- `kaggle datasets download owner/slug`

Imported files are stored under the NeuroGolf project scratch folder:

- `NEUROGOLF_IMPORTED_SOURCES_DIR`

The current default path is:

- `D:\E into D(12-15-2025_15-33)\coding 25-26\neurogolf_arc_system\scratch\gui_imports`

## Role editing notes

When you change a model in the manual role section:

- the role keeps the selected model name
- the context length can auto-reset to a model-based default
- saving writes to `.env`
- the daemon uses the new values after restart

## Typical safe workflow

1. Open the dashboard.
2. Use `Sync Kaggle State`.
3. Paste new Kaggle references into `Kaggle Source Intake`.
4. Use `Import Kaggle Sources`.
5. Use `Preview Declutter Archive` when the workspace gets noisy.
6. Configure the `Teacher Distillation Track` if you are collecting rule signals.
7. Load or adjust a model profile if needed.
8. Save the operator note.
9. Use `Start Autonomy`.
10. Watch `Reports` and `NeuroGolf`.

## Teacher Distillation Track

The intended NeuroGolf workflow is:

1. Use a large teacher model to infer the symbolic transformation rule.
2. Generate several legal static ONNX graph candidates for the task.
3. Distill the teacher behavior into the smallest student graph.
4. Validate shape inference, banned operators, file size, and graph cost locally.
5. Keep only improvements that solve more examples or reduce `params + memory + MACs`.

The dataset helper can create prompt records for the teacher:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_distillation_dataset.py --include-unlabeled
```

It writes to `outputs/distillation/teacher_prompts.jsonl`. Once teacher signals are recorded in
`memory/distillation_plan.json`, the same script also emits supervised chat records for fine-tuning.

## Important note

The dashboard saves configuration to `.env`, but a running daemon only adopts those changes after restart.

Use:

- `Restart Daemon`
- or `Apply Profile And Restart`

when you want the live autonomy loop to pick up changed roles or contexts.

Control values are persisted, so sleep time, pending submission cap, history depth, and submission permission stay stable across reruns and daemon restarts.
