from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys

from core.config import (
    KAGGLE_COMPETITION,
    KAGGLE_POLL_ATTEMPTS,
    KAGGLE_POLL_SECONDS,
    KAGGLE_SUBMISSION_FILE,
    NEUROGOLF_PROJECT_ROOT,
    SUBMISSION_MIN_DELTA,
    SUBMISSION_STATE_PATH,
)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'kaggle' package is not installed. Run bootstrap.ps1 first.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a NeuroGolf zip to Kaggle and poll the result.")
    parser.add_argument("--file", type=Path, default=KAGGLE_SUBMISSION_FILE)
    parser.add_argument("--competition", default=KAGGLE_COMPETITION)
    parser.add_argument("--message", default="")
    parser.add_argument("--estimated-score", type=float, default=None)
    parser.add_argument("--estimate-file", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=SUBMISSION_MIN_DELTA)
    parser.add_argument("--poll-seconds", type=int, default=KAGGLE_POLL_SECONDS)
    parser.add_argument("--poll-attempts", type=int, default=KAGGLE_POLL_ATTEMPTS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_state() -> dict:
    if not SUBMISSION_STATE_PATH.exists():
        return {}
    return json.loads(SUBMISSION_STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    SUBMISSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _score_from_json(data: dict) -> float | None:
    for key in ("estimated_live_score", "known_score", "public_score", "score", "estimated_score"):
        value = data.get(key)
        if value is not None:
            return float(value)
    return None


def resolve_estimated_score(args: argparse.Namespace) -> float | None:
    if args.estimated_score is not None:
        return float(args.estimated_score)
    if args.estimate_file and args.estimate_file.exists():
        suffix = args.estimate_file.suffix.lower()
        if suffix == ".json":
            return _score_from_json(json.loads(args.estimate_file.read_text(encoding="utf-8")))
        return float(args.estimate_file.read_text(encoding="utf-8").strip())
    return None


def serialize_submission(submission: object) -> dict:
    date_value = getattr(submission, "date", None)
    if hasattr(date_value, "isoformat"):
        date_value = date_value.isoformat()
    status_value = getattr(submission, "status", None)
    if status_value is not None:
        status_value = str(status_value)
    public_score = getattr(submission, "publicScore", None)
    if public_score is not None:
        public_score = float(public_score)
    private_score = getattr(submission, "privateScore", None)
    if private_score is not None:
        private_score = float(private_score)
    return {
        "ref": getattr(submission, "ref", None),
        "description": getattr(submission, "description", None),
        "status": status_value,
        "public_score": public_score,
        "private_score": private_score,
        "file_name": getattr(submission, "fileName", None),
        "date": date_value,
    }


def latest_submission(api: KaggleApi, competition: str) -> object | None:
    submissions = list(api.competition_submissions(competition))
    return submissions[0] if submissions else None


def ensure_kaggle_auth_context() -> None:
    kaggle_json = NEUROGOLF_PROJECT_ROOT / "kaggle.json"
    if kaggle_json.exists() and "KAGGLE_CONFIG_DIR" not in os.environ:
        os.environ["KAGGLE_CONFIG_DIR"] = str(NEUROGOLF_PROJECT_ROOT)


@contextmanager
def staged_submission_file(submission_file: Path):
    if submission_file.name.lower() == "submission.zip":
        yield submission_file
        return
    with tempfile.TemporaryDirectory(prefix="neurogolf_submit_") as tmpdir:
        staged = Path(tmpdir) / "submission.zip"
        shutil.copyfile(submission_file, staged)
        yield staged


def submission_history(api: KaggleApi, competition: str, limit: int = 10) -> list[dict]:
    kaggle_exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle.exe"))
    try:
        completed = subprocess.run(
            [kaggle_exe, "competitions", "submissions", competition, "-v", "--page-size", str(limit)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    lines = [line for line in completed.stdout.splitlines() if line.strip() and not line.startswith("Warning:")]
    if not lines:
        return rows
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return rows
    cli_rows: list[dict] = []
    for parts in reader:
        if len(parts) < 6:
            continue
        cli_rows.append(
            {
                "file_name": parts[0],
                "date": parts[1],
                "description": parts[2],
                "status": parts[3],
                "public_score": float(parts[4]) if parts[4] else None,
                "private_score": float(parts[5]) if parts[5] else None,
            }
        )
    if cli_rows and any(item.get("public_score") is not None for item in cli_rows):
        return cli_rows
    return [serialize_submission(item) for item in list(api.competition_submissions(competition))[:limit]]


def _exception_details(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    parts: list[str] = []
    try:
        if response.status_code:
            parts.append(f"status={response.status_code}")
    except Exception:
        pass
    try:
        text = (response.text or "").strip()
        if text:
            parts.append(text[:1000])
    except Exception:
        pass
    return " | ".join(parts)


def main() -> int:
    args = build_parser().parse_args()
    submission_file = args.file.resolve()
    if not submission_file.exists() and not args.dry_run:
        raise SystemExit(f"Submission file not found: {submission_file}")

    estimated_score = resolve_estimated_score(args)
    state = load_state()
    best_estimate = float(state.get("best_estimated_score", 0.0))

    if estimated_score is None and not args.force:
        raise SystemExit("Provide --estimated-score or --estimate-file, or use --force.")

    if estimated_score is not None and not args.force:
        improvement = estimated_score - best_estimate
        if state and improvement < args.threshold:
            print(
                f"Skipping submit. Estimated score {estimated_score:.2f} improves only {improvement:.2f}, "
                f"below threshold {args.threshold:.2f}."
            )
            return 0

    message = args.message.strip() or (
        f"auto submit {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        + (f" est={estimated_score:.2f}" if estimated_score is not None else "")
    )

    if args.dry_run:
        print(f"Would submit {submission_file} to {args.competition} with message: {message}")
        return 0

    ensure_kaggle_auth_context()
    api = KaggleApi()
    api.authenticate()
    before = latest_submission(api, args.competition)
    before_ref = getattr(before, "ref", None)

    print(f"Submitting {submission_file} to {args.competition}")
    print(f"Message: {message}")
    with staged_submission_file(submission_file) as staged_file:
        try:
            api.competition_submit(str(staged_file), message, args.competition)
        except Exception as exc:
            detail = _exception_details(exc)
            suffix = f" | {detail}" if detail else ""
            print(f"API submit failed, falling back to Kaggle CLI: {type(exc).__name__}: {exc}{suffix}")
            kaggle_exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle.exe"))
            completed = subprocess.run(
                [kaggle_exe, "competitions", "submit", "-c", args.competition, "-f", str(staged_file), "-m", message],
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                print(completed.stdout.strip())
            if completed.returncode != 0:
                stderr = completed.stderr.strip() if completed.stderr else "unknown Kaggle CLI error"
                raise SystemExit(f"Kaggle submit failed: {stderr}") from exc

    current = None
    for attempt in range(args.poll_attempts):
        time.sleep(args.poll_seconds)
        current = latest_submission(api, args.competition)
        if current is None:
            continue
        current_ref = getattr(current, "ref", None)
        if before_ref and current_ref == before_ref and attempt < args.poll_attempts - 1:
            continue
        status = str(getattr(current, "status", "")).lower()
        snapshot = serialize_submission(current)
        print(json.dumps(snapshot, indent=2))
        if status and all(word not in status for word in ("pending", "running", "processing")):
            break

    latest = serialize_submission(current) if current else {}
    history = submission_history(api, args.competition, limit=10)
    completed_scores = [float(item["public_score"]) for item in history if item.get("public_score") is not None]
    if estimated_score is not None:
        state["best_estimated_score"] = max(best_estimate, estimated_score)
        state["last_estimated_score"] = estimated_score
    if completed_scores:
        state["best_public_score"] = max(completed_scores)
        state["recent_submissions"] = history
    state["last_submission"] = latest
    state["last_message"] = message
    state["last_file"] = str(submission_file)
    state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
