# Deployment Guide

## Best Deployment Mode

This project is best deployed as a self-hosted dashboard with:

- a running Ollama server
- access to your NeuroGolf workspace
- optional Kaggle credentials for source intake and submissions

The GUI can run in a container, but the model server and competition workspace are usually external resources.

## Recommended Path

Use Docker Compose on your own machine or VPS.

Why:

- the app depends on Ollama
- the app expects a writable NeuroGolf workspace
- the autonomy loop benefits from persistent local files and logs

## Quick Deploy

1. Copy `.env.example` to `.env`
2. Adjust the model names and paths
3. Put your NeuroGolf workspace under `./mounted/neurogolf` or change the mount path in `docker-compose.yml`
4. Start Ollama so the container can reach it
5. Run:

```bash
docker compose up --build -d
```

Then open:

- `http://localhost:8501`

## Remote Control From Any Device

If the laptop is going to stay on and act as the control host, this is the recommended setup:

1. Keep the laptop plugged in
2. Disable sleep and hibernate for plugged-in mode
3. Leave Ollama running
4. Start the Streamlit dashboard
5. Protect the UI with `SKYNET_UI_PASSWORD`
6. Use one of these access paths:
   - same Wi-Fi or LAN: use the dashboard's LAN URL
   - remote internet access: use Tailscale and the dashboard's Tailscale URL

Why Tailscale is preferred:

- no router port-forwarding
- private device-to-device access
- much safer than exposing raw Streamlit to the public internet

The dashboard now binds to `0.0.0.0`, so it is reachable from other devices once the host firewall and network path allow it.

## Ollama Connectivity

The default compose file assumes:

- `OLLAMA_BASE_URL=http://host.docker.internal:11434`

If Ollama is running on another host, change `OLLAMA_BASE_URL` in `.env`.

## Kaggle Credentials

This repo does not commit credentials.

If you want Kaggle source pulls and submissions to work in deployment:

- mount or copy a valid `kaggle.json`
- make sure `KAGGLE_CONFIG_DIR` points to the right folder

## Streamlit Community Cloud

This repo can be shown on Streamlit Cloud, but that is not the recommended full deployment path.

Limitations:

- no local Ollama by default
- no real NeuroGolf workspace mount
- no practical autonomous submission loop

So Streamlit Cloud is useful for a preview UI, not for the full agent system.

## Render / Railway

Possible, but only if:

- Ollama is available remotely
- you provide writable persistent storage
- you wire Kaggle credentials and workspace paths carefully

Without those, the dashboard becomes mostly read-only.

## GitHub Actions

The repo includes a CI workflow that:

- installs Python dependencies
- compiles all Python files
- checks that the main deployment files exist

That gives you a clean baseline for future pushes.
