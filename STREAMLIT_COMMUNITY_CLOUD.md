# Streamlit Community Cloud

This repository is ready for a Streamlit Community Cloud app with:

- **Repository:** `hyper10203/Skynet-Control-Core`
- **Branch:** `codex/skynet-neurogolf-core-distillation` (or `main` after merge)
- **Entrypoint:** `streamlit_app.py`
- **Suggested app URL:** `axiomgraph-operations-core`

## What deploys cleanly

The Streamlit interface, reports, visualizers, and the branded portal surface deploy cleanly to Community Cloud.

The dashboard now includes a **built-in portal fallback**. If the separate localhost website is not reachable, the app still presents the landing and command-surface previews directly from the repo instead of showing a dead iframe.

## What stays local

The full offline control core still depends on:

- local Ollama models
- local Kaggle credentials
- local filesystem artifacts and daemon processes

That means Community Cloud is best used as a hosted portal / observer / control surface, while the heavy local-first autonomy loop continues to run on your machine.

## Recommended secrets

Paste secrets into Community Cloud app settings instead of committing them:

```toml
SKYNET_UI_PASSWORD = "choose-a-password"
```

## Create app values

Use the following values in the Community Cloud Create App flow:

- **GitHub repo:** `hyper10203/Skynet-Control-Core`
- **Branch:** `codex/skynet-neurogolf-core-distillation`
- **Main file path:** `streamlit_app.py`
- **Python version:** `3.12`

## Notes

- The repo root already contains `.streamlit/config.toml`.
- The repo root already contains `requirements.txt`.
- Changes pushed to the selected branch will trigger app updates automatically.
- Official Streamlit Community Cloud deployment still requires the account-side browser workflow at `share.streamlit.io` to create the app.
