from __future__ import annotations

import argparse
import json

from core.config import NEUROGOLF_SYNC_STATE_PATH
from core.neurogolf_context import sync_neurogolf_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync NeuroGolf repo outputs and Kaggle history into agent memory.")
    parser.add_argument("--history", type=int, default=10, help="How many Kaggle submissions to include.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = sync_neurogolf_state(limit_history=args.history)
    print(json.dumps({"state_path": str(NEUROGOLF_SYNC_STATE_PATH), "best_completed_public_score": summary.get("best_completed_public_score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
