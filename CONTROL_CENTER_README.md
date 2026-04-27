# Skynet Control Core Guide

This file explains the local GUI, the autonomy controls, the model team, and the Kaggle intake workflow.

## What this system is

`Skynet Control Core` is the Streamlit dashboard for the ARC and NeuroGolf agent stack.

It controls:

- the autonomous NeuroGolf daemon
- the local Ollama model team
- Kaggle source harvesting
- ARC task visualization
- model role and context configuration

The dashboard refreshes itself every 5 minutes.

## Main tabs

### Overview

Shows:

- current daemon status
- current rank
- latest completed public score
- current mission phase
- live daemon log tail
- score trend over time
- operator note preview

### Control

Used for runtime actions:

- starting and stopping the autonomy daemon
- running one single cycle manually
- syncing Kaggle state
- health checks
- restarting the daemon with current settings
- editing the persistent operator note
- importing Kaggle notebooks and datasets

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

- `Refresh Intel`
  - Clears cached dashboard data and refreshes the page immediately.

- `Bring Skynet Online`
  - Starts the autonomous NeuroGolf daemon with the current control settings.

- `Destroy Skynet`
  - Stops the autonomous daemon.

- `Send a T-800`
  - Runs one autonomy cycle once without waiting for the full background loop.

- `Scan the Battlefield`
  - Re-syncs local NeuroGolf outputs and Kaggle submission history into memory.

- `Cyberdyne Diagnostics`
  - Runs the model health check against the current Ollama stack.

- `Reboot the Time Core`
  - Restarts the daemon using the current settings in the control tab.

- `Rewrite the Future`
  - Saves the operator note.

- `Erase the Timeline`
  - Clears the operator note.

- `Arm Judgment Day`
  - Saves the operator note so future autonomy cycles use it.

- `Harvest Future Files`
  - Pulls Kaggle notebooks or datasets from pasted URLs or CLI lines.

- `Review Time Displacement`
  - Explains where imported Kaggle files appear.

### Orchestrator chat buttons

- `Purge Chat Memory`
  - Clears the current chat session inside the dashboard.

- `Inject Live State`
  - Explains that live NeuroGolf and model-team context is already attached to each orchestrator reply.

### Model buttons

- `Load Into Roles`
  - Loads the selected model profile into the editable role fields without saving yet.

- `Install Neural Chip`
  - Saves the selected model profile into `.env`.

- `Judgment Day Upgrade`
  - Saves the selected model profile into `.env` and restarts the daemon.

- `Use My Installed Best Models`
  - Detects the strongest profile that matches your locally installed Ollama models and loads it into the role editor.

- `Override the Neural Net`
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

You can paste any mix of these into the `Kaggle Intel Intake` box:

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
2. Use `Scan the Battlefield`.
3. Paste new Kaggle references into `Kaggle Intel Intake`.
4. Use `Harvest Future Files`.
5. Load or adjust a model profile if needed.
6. Save the operator note.
7. Use `Bring Skynet Online`.
8. Watch `Reports` and `NeuroGolf`.

## Important note

The dashboard saves configuration to `.env`, but a running daemon only adopts those changes after restart.

Use:

- `Reboot the Time Core`
- or `Judgment Day Upgrade`

when you want the live autonomy loop to pick up changed roles or contexts.

Control values in the flight deck are now also persisted, so sleep time, pending submission cap, history depth, and submission permission stay stable across reruns and daemon restarts.
