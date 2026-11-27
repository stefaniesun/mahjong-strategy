"""影子对比: 同一局面下让两个冻结版本各自决策, 统计分歧率与分歧类型 (tools/shadow_diff_versions.py)。

用途: 竞技场大样本只能告诉你"净分差", 分辨率受限于局数 (1 万局仅 ±0.09)。
      本工具直接比较两个版本的逐决策分歧率, 用来判断一次升级到底有没有落地:
      - 分歧率接近 0 → 升级条件基本不触发, 再多局也测不出差别, 应先修触发条件而非加样本;
      - 某类动作(碰/杠/弃牌)分歧集中 → 定位到具体升级项。

对局走向由 --base 版本推着走, --cand 只在同一 state 上做影子决策, 不影响牌局。

    python tools/shadow_diff_versions.py --deals 400
    python tools/shadow_diff_versions.py --base v3_goodwait_read --cand v4_waitq_tactics --deals 400 \\
        --out outputs/shadow_diff_v3_v4.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.recorder import PublicRecorder
from assistant.zoo import load_version
from engine.actions import ActionKind
from engine.game import Game
from play.session import next_actor

# 换三张的 tiles 顺序不稳定, 需按集合比较
_UNORDERED_KINDS = {ActionKind.SWAP_THREE}


def _norm(act) -> tuple:
    if act.kind in _UNORDERED_KINDS:
        return (act.kind, tuple(sorted(act.tiles)))
    return (act.kind, act.tile, act.suit, act.kong_kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v3_goodwait_read", help="推动对局走向的基准版本")
    ap.add_argument("--cand", default="v4_waitq_tactics", help="做影子决策的候选版本")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=900_000)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "shadow_diff.json")
    args = ap.parse_args()

    base = load_version(args.base, stochastic=False, exact_swap_live=False)
    cand = load_version(args.cand, stochastic=False, exact_swap_live=False)

    total: collections.Counter = collections.Counter()
    diff: collections.Counter = collections.Counter()
    detail: collections.Counter = collections.Counter()
    examples: list[dict] = []

    for i in range(args.deals):
        seed = args.seed_start + i
        game = Game(seed=seed, dealer=seed % 4)
        game.reset()
        rec = PublicRecorder(game)
        steps = 0
        while not game.state.finished and steps < 1000:
            actor = next_actor(game)
            if actor is None:
                break
            legal = game.legal_actions(actor)
            a_base = base.choose_from_game(game.state, rec, actor, legal)
            a_cand = a_base if len(legal) == 1 else cand.choose_from_game(game.state, rec, actor, legal)
            kb, kc = _norm(a_base), _norm(a_cand)
            total[kb[0].value] += 1
            if kb != kc:
                diff[kb[0].value] += 1
                detail[f"{kb[0].value}->{kc[0].value}"] += 1
                if len(examples) < args.examples and kb[0] is not ActionKind.PASS:
                    examples.append({"seed": seed, "seat": actor, "base": str(kb), "cand": str(kc)})
            rec.step(actor, a_base)
            steps += 1

    report = {
        "base": args.base,
        "cand": args.cand,
        "deals": args.deals,
        "overall_diff_rate": sum(diff.values()) / max(1, sum(total.values())),
        "diff_by_kind": {
            k: {"diff": diff[k], "total": total[k], "rate": diff[k] / total[k] if total[k] else 0.0} for k in total
        },
        "diff_detail": dict(detail.most_common()),
        "examples": examples,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.base} vs {args.cand}: {args.deals} 副牌, 共 {sum(total.values())} 个决策")
    print(f"总体分歧率 {report['overall_diff_rate'] * 100:.2f}%")
    for kind, row in sorted(report["diff_by_kind"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {kind:<12} {row['total']:>7} 决策, 分歧 {row['diff']:>6} ({row['rate'] * 100:5.2f}%)")
    print("分歧类型:", dict(detail.most_common(10)))


if __name__ == "__main__":
    main()
