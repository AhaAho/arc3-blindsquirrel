from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from arc_agi import Arcade, OperationMode


def _get_local_games() -> list[str]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    return sorted({env.game_id for env in arc.get_environments() if env.game_id})


def _filter_games(games: list[str], filters: str | None) -> list[str]:
    if not filters:
        return games
    prefixes = [prefix.strip() for prefix in filters.split(",") if prefix.strip()]
    return [gid for gid in games if any(gid.startswith(prefix) for prefix in prefixes)]


def _chunk(games: list[str], batch_size: int) -> list[list[str]]:
    return [games[i : i + batch_size] for i in range(0, len(games), batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Blind Squirrel offline in sequential batches."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="How many games to run in each batch.",
    )
    parser.add_argument(
        "--games",
        type=str,
        default=None,
        help="Optional comma-separated game prefixes to include.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional device cycle, for example 'cuda:0,cuda:1'.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Optional comma-separated scorecard tags to pass through.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any batch exits non-zero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned batches without launching them.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    repo_root = Path(__file__).resolve().parent
    games = _filter_games(_get_local_games(), args.games)
    if not games:
        print("No offline games matched the requested filters.", file=sys.stderr)
        return 1

    batches = _chunk(games, args.batch_size)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_log_dir = repo_root / "batch_logs" / timestamp

    print(f"offline_games={len(games)}")
    print(f"batch_size={args.batch_size}")
    print(f"num_batches={len(batches)}")
    print(f"batch_log_dir={batch_log_dir}")

    for index, batch in enumerate(batches, start=1):
        batch_games = ",".join(batch)
        print(f"batch_{index:02d}: {batch_games}")

    if args.dry_run:
        return 0

    batch_log_dir.mkdir(parents=True, exist_ok=True)
    python_bin = repo_root / ".venv" / "bin" / "python"
    overall_rc = 0

    for index, batch in enumerate(batches, start=1):
        batch_games = ",".join(batch)
        batch_log_path = batch_log_dir / f"batch_{index:02d}.log"
        cmd = [
            str(python_bin),
            "main.py",
            "--agent=blindsquirrel",
            f"--game={batch_games}",
        ]
        if args.tags:
            cmd.append(f"--tags={args.tags}")

        env = os.environ.copy()
        env["OPERATION_MODE"] = "offline"
        env["RUN_LOG_PATH"] = str(batch_log_path)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        if args.devices:
            env["BLINDSQUIRREL_DEVICES"] = args.devices

        print(
            f"[batch {index:02d}/{len(batches)}] starting games={batch_games} log={batch_log_path}"
        )
        result = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
        print(f"[batch {index:02d}/{len(batches)}] exit_code={result.returncode}")

        if result.returncode != 0:
            overall_rc = result.returncode
            if args.stop_on_error:
                return overall_rc

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
