from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from core.config import (
    AXIOMGRAPH_BRIDGE_LOG_PATH,
    AXIOMGRAPH_BRIDGE_LOOP_SECONDS,
    AXIOMGRAPH_BRIDGE_PID_PATH,
    AXIOMGRAPH_BRIDGE_STATUS_PATH,
)
from core.remote_bridge import process_runtime_node_command, publish_runtime_node_heartbeat


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    AXIOMGRAPH_BRIDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AXIOMGRAPH_BRIDGE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now()}] {message}\n")


def _write_status(phase: str, message: str, *, progress: float, extra: dict | None = None) -> None:
    payload = {
        "updated_at": _utc_now(),
        "phase": phase,
        "message": message,
        "progress": max(0.0, min(1.0, progress)),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    AXIOMGRAPH_BRIDGE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AXIOMGRAPH_BRIDGE_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _combine_outputs(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if str(part).strip())


def run_bridge_once(*, interval_seconds: int, iteration: int) -> dict:
    _write_status(
        "bridge_sync",
        "Publishing heartbeat and polling for remote commands.",
        progress=0.2,
        extra={"interval_seconds": interval_seconds, "iteration": iteration},
    )
    heartbeat = publish_runtime_node_heartbeat()
    _write_status(
        "bridge_poll",
        "Heartbeat published. Checking for queued remote commands.",
        progress=0.55,
        extra={
            "interval_seconds": interval_seconds,
            "iteration": iteration,
            "last_heartbeat_path": heartbeat.get("path", ""),
        },
    )
    command = process_runtime_node_command()
    ok = bool(heartbeat.get("ok", False)) and bool(command.get("ok", True))
    stdout = _combine_outputs(heartbeat.get("stdout", ""), command.get("stdout", ""))
    stderr = _combine_outputs(heartbeat.get("stderr", ""), command.get("stderr", ""))
    ack = command.get("command") if isinstance(command.get("command"), dict) else {}
    result = {
        "ok": ok,
        "stdout": stdout,
        "stderr": stderr,
        "heartbeat": heartbeat,
        "command": ack,
    }
    _write_status(
        "bridge_idle",
        "Runtime node bridge loop healthy.",
        progress=1.0 if ok else 0.0,
        extra={
            "interval_seconds": interval_seconds,
            "iteration": iteration,
            "ok": ok,
            "last_heartbeat_path": heartbeat.get("path", ""),
            "last_command_id": ack.get("id", ""),
            "last_command_action": ack.get("action", ""),
            "last_command_status": ack.get("status", ""),
            "stdout": stdout[:2000],
            "stderr": stderr[:2000],
        },
    )
    _log(stdout or "Bridge roundtrip completed.")
    if stderr:
        _log(f"stderr: {stderr}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AxiomGraph Runtime Node bridge loop.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--interval-seconds", type=int, default=AXIOMGRAPH_BRIDGE_LOOP_SECONDS)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever when --loop is enabled.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    AXIOMGRAPH_BRIDGE_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    AXIOMGRAPH_BRIDGE_PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    _log("Runtime node bridge starting.")

    cycle = 0
    try:
        if not args.loop:
            run_bridge_once(interval_seconds=max(15, int(args.interval_seconds)), iteration=1)
            return 0

        _write_status(
            "bridge_boot",
            "Runtime node bridge loop starting.",
            progress=0.0,
            extra={"interval_seconds": max(15, int(args.interval_seconds)), "iteration": 0},
        )
        while True:
            cycle += 1
            run_bridge_once(interval_seconds=max(15, int(args.interval_seconds)), iteration=cycle)
            if args.max_cycles and cycle >= args.max_cycles:
                break
            time.sleep(max(15, int(args.interval_seconds)))
        return 0
    except KeyboardInterrupt:
        _write_status("bridge_stopped", "Runtime node bridge loop interrupted.", progress=0.0, extra={"iteration": cycle})
        _log("Runtime node bridge interrupted by keyboard.")
        return 130
    except Exception:
        error_text = traceback.format_exc()
        _write_status(
            "bridge_error",
            "Runtime node bridge loop crashed.",
            progress=0.0,
            extra={"iteration": cycle, "stderr": error_text[-4000:]},
        )
        _log(error_text)
        return 1
    finally:
        try:
            if AXIOMGRAPH_BRIDGE_PID_PATH.exists():
                AXIOMGRAPH_BRIDGE_PID_PATH.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
