"""换牌配对分片的本地速度基准。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_swap_shard import main as run_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="测量换牌分片运行速度")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--dealer", type=int, choices=range(4), default=0)
    parser.add_argument("--target", type=int, choices=range(4), default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark_swap_shard.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    shard_path = args.out.with_suffix(".shard.json")
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["run_swap_shard", "--seed-start", str(args.seed_start), "--games", str(args.games), "--dealer", str(args.dealer), "--target", str(args.target), "--out", str(shard_path), "--force"]
        run_shard()
    finally:
        sys.argv = old_argv
    elapsed = time.perf_counter() - started
    result = {
        "games": args.games,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_game": round(elapsed / args.games, 6) if args.games else 0.0,
        "estimated_games_per_hour": round(args.games * 3600 / elapsed, 1) if elapsed else 0.0,
        "shard": str(shard_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
