"""运行可恢复的换牌策略配对分片任务。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from compare_swap_policies import _run
from policies.heuristics import choose_swap_tiles, choose_swap_tiles_legacy


CHOOSERS = {"legacy": choose_swap_tiles_legacy, "principled": choose_swap_tiles}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行旧版/新版换牌配对分片")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--dealer", type=int, choices=range(4), required=True)
    parser.add_argument("--target", type=int, choices=range(4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        raise SystemExit(f"输出已存在，使用 --force 覆盖：{args.out}")

    started = time.perf_counter()
    checkpoint = args.out.with_suffix(args.out.suffix + ".checkpoint")
    rows = []
    failures = []
    if checkpoint.exists() and not args.force:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("seed_start") == args.seed_start and saved.get("games") == args.games and saved.get("dealer") == args.dealer and saved.get("target") == args.target:
            rows = saved.get("rows", [])
            failures = saved.get("failures", [])
    completed_seeds = {row["seed"] for row in rows} | {item["seed"] for item in failures}
    for offset in range(args.games):
        if args.seed_start + offset in completed_seeds:
            continue
        seed = args.seed_start + offset
        try:
            legacy = _run(seed, args.dealer, args.target, CHOOSERS["legacy"])
            principled = _run(seed, args.dealer, args.target, CHOOSERS["principled"])
            rows.append({
                "seed": seed,
                "dealer": args.dealer,
                "target": args.target,
                "suit_count_bucket": legacy["suit_count_bucket"],
                "legacy": legacy,
                "principled": principled,
                "same_action": int(legacy["action"] == principled["action"]),
                "score_delta": principled["score"] - legacy["score"],
                "receiver_score_delta": principled["receiver_score"] - legacy["receiver_score"],
                "net_score_delta": principled["net_score"] - legacy["net_score"],
            })
        except Exception as exc:  # 保留失败 seed，便于重跑定位
            failures.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        checkpoint.write_text(json.dumps({"seed_start": args.seed_start, "games": args.games, "dealer": args.dealer, "target": args.target, "rows": rows, "failures": failures}, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.games - 1,
        "requested_games": args.games,
        "completed_games": len(rows),
        "failed_games": len(failures),
        "dealer": args.dealer,
        "target": args.target,
        "policies": ["legacy", "principled"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {"metadata": metadata, "rows": rows, "failures": failures}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
