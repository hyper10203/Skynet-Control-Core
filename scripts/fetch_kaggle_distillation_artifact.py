from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Kaggle distillation kernel outputs.")
    parser.add_argument("--kernel", default="subhampaulchoudhury/axiomgraph-neurogolf-student-distillation-v10")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "distillation" / "kaggle_artifacts")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kaggle = PROJECT_ROOT / ".venv" / "Scripts" / "kaggle.exe"
    command = [str(kaggle if kaggle.exists() else "kaggle"), "kernels", "output", args.kernel, "-p", str(args.out), "--force"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
