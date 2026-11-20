"""配对评估:同一副牌、同一座位,候选和基准各打一遍,比差值。

配对消掉了牌运的方差,差值的 95% 置信区间下界大于 0 才算候选确实更强。

    python tools/eval_paired.py --games 2000 --variant smart_kong --variant void_order --variant all
    python tools/eval_paired.py --games 200 --variant none     # 自检:全关应当和 S3 完全一样,差值恒为 0

对手是三家 S3。规则读 configs/rules.yaml(或环境变量 MAHJONG_RULES),结果里记录用的是哪份规则。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.driver import run_game
from engine.config import active_rule_config
from policies.rule_policy import RulePolicy
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy


def play(seed: int, variant: str | None) -> dict:
    """variant 为 None 时四家都是 S3;否则座位 seed%4 换成 S3+。"""
    seat = seed % 4
    policies = [RulePolicy() for _ in range(4)]
    if variant is not None:
        policies[seat] = RulePlusPolicy(PlusConfig.parse(variant))
    game, _ = run_game(seed, policies=policies)
    state = game.state
    record = state.win_records[seat]
    return {
        "seed": seed,
        "score": state.scores[seat],
        "won": state.won[seat],
        "self_draw": bool(record and record.kind == "self_draw"),
        "dealt_in": sum(1 for r in state.win_records if r is not None and r.from_player == seat and r.kind == "discard"),
        "kongs": sum(1 for g in state.gang_records if g.gang_player == seat),
        "finished": state.finished,
        "sum": sum(state.scores),
    }


def _play_star(args: tuple[int, str | None]) -> dict:
    return play(*args)


def _paired_ci(values: list[float]) -> tuple[float, float, float]:
    """(均值, 95% 区间半宽, 单局标准差)。"""
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, n - 1))
    return mean, 1.96 * sd / math.sqrt(n), sd


def summarize(base: list[dict], cand: list[dict]) -> dict:
    diffs = [c["score"] - b["score"] for b, c in zip(base, cand)]
    n = len(diffs)
    mean, half, sd = _paired_ci(diffs)
    win_mean, win_half, _ = _paired_ci([float(c["won"]) - float(b["won"]) for b, c in zip(base, cand)])
    deal_mean, deal_half, _ = _paired_ci([float(c["dealt_in"] - b["dealt_in"]) for b, c in zip(base, cand)])
    rate = lambda rows, key: sum(bool(r[key]) if key != "dealt_in" else r[key] for r in rows) / n  # noqa: E731
    return {
        "games": n,
        "diff_mean": round(mean, 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
        "significant": mean - half > 0,
        "diff_sd": round(sd, 3),
        "win_rate_diff_ci95": [round(win_mean - win_half, 4), round(win_mean + win_half, 4)],
        "deal_in_diff_ci95": [round(deal_mean - deal_half, 4), round(deal_mean + deal_half, 4)],
        "changed_games": sum(1 for d in diffs if d != 0),
        "score": [round(sum(r["score"] for r in rows) / n, 4) for rows in (base, cand)],
        "win_rate": [round(rate(rows, "won"), 4) for rows in (base, cand)],
        "self_draw_rate": [round(rate(rows, "self_draw"), 4) for rows in (base, cand)],
        "deal_in_per_game": [round(rate(rows, "dealt_in"), 4) for rows in (base, cand)],
        "kongs_per_game": [round(sum(r["kongs"] for r in rows) / n, 4) for rows in (base, cand)],
        "bad_games": sum(1 for r in cand if not r["finished"] or r["sum"] != 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="S3+ 配对评估")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--variant", action="append", required=True, help="开关组合,如 smart_kong 或 smart_kong,void_order;all 全开;none 全关")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.games)
    results: dict = {"rules": asdict(active_rule_config()), "seed_start": args.seed_start, "variants": {}}
    with mp.Pool(args.workers) as pool:
        started = time.time()
        base = pool.map(_play_star, [(s, None) for s in seeds], chunksize=20)
        print(f"基准 S3 完成,用时 {time.time() - started:.0f}s", flush=True)
        for variant in args.variant:
            started = time.time()
            cand = pool.map(_play_star, [(s, variant) for s in seeds], chunksize=20)
            summary = summarize(base, cand)
            results["variants"][variant] = summary
            verdict = "显著更强" if summary["significant"] else ("显著更弱" if summary["ci95"][1] < 0 else "分不出高低")
            print(
                f"{variant:<40} 差值 {summary['diff_mean']:+.3f} 区间 {summary['ci95']} {verdict}"
                f" | 胡牌率 {summary['win_rate'][0]:.3f}->{summary['win_rate'][1]:.3f} {summary['win_rate_diff_ci95']}"
                f" 放炮/局 {summary['deal_in_per_game'][0]:.3f}->{summary['deal_in_per_game'][1]:.3f} {summary['deal_in_diff_ci95']}"
                f" 杠/局 {summary['kongs_per_game'][0]:.3f}->{summary['kongs_per_game'][1]:.3f}"
                f" 有差异局 {summary['changed_games']} 用时 {time.time() - started:.0f}s",
                flush=True,
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
