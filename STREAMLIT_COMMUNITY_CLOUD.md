# Streamlit Community Cloud

This repository is ready for a Streamlit Community Cloud app with:

- **Repository:** `hyper10203/Skynet-Control-Core`
- **Branch:** `codex/skynet-neurogolf-core-distillation` (or `main` after merge)
- **Entrypoint:** `streamlit_app.py`

## What deploys cleanly

The Streamlit interface, reports, visualizers, and website portal deploy cleanly to Community Cloud.

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

## Notes

- The repo root already contains `.streamlit/config.toml`.
- The repo root already contains `requirements.txt`.
- Changes pushed to the selected branch will trigger app updates automatically.
