"""在同一批初始牌局上比较旧版和新版换牌策略。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import legal_swap_actions
from engine.game import Game
from engine.tiles import SUITS
from policies.heuristics import choose_swap_tiles, choose_swap_tiles_legacy
from policies.rule_policy import RulePolicy
from tools.evaluate_swap_counterfactual import _action_key, _play_fixed_swap


def _find_action(actions, tiles):
    key = tuple(sorted(tile.index for tile in tiles))
    return next(action for action in actions if _action_key(action) == key)


def _run(seed: int, dealer: int, target: int, chooser):
    initial = Game(seed=seed, dealer=dealer)
    state = initial.reset()
    hand = state.hands[target]
    actions = legal_swap_actions(hand)
    action = _find_action(actions, chooser(hand))
    policies = [RulePolicy() for _ in range(4)]
    game, _, _ = _play_fixed_swap(seed, dealer, target, action, policies)
    final = game.state
    receiver = (target - state.swap_direction) % 4
    suit_count_bucket = "/".join(str(count) for count in sorted(sum(1 for tile in hand.tiles() if tile.suit is suit) for suit in SUITS))
    return {
        "suit_count_bucket": suit_count_bucket,
        "action": [f"{tile.rank}{tile.suit.value}" for tile in action.tiles],
        "score": final.scores[target],
        "receiver_score": final.scores[receiver],
        "net_score": final.scores[target] - final.scores[receiver],
        "won": int(final.won[target]),
        "receiver_won": int(final.won[receiver]),
        "finished": int(final.finished),
        "win_order": final.win_order.index(target) + 1 if target in final.win_order else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="配对比较旧版和新版换牌策略")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--dealer", type=int, choices=range(4), default=0)
    parser.add_argument("--target", type=int, choices=range(4), default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/swap_policy_comparison.json"))
    args = parser.parse_args()

    rows = []
    for offset in range(args.games):
        seed = args.seed_start + offset
        legacy = _run(seed, args.dealer, args.target, choose_swap_tiles_legacy)
        principled = _run(seed, args.dealer, args.target, choose_swap_tiles)
        rows.append({
            "seed": seed,
            "dealer": args.dealer,
            "target": args.target,
            "legacy": legacy,
            "principled": principled,
            "same_action": int(legacy["action"] == principled["action"]),
            "score_delta": principled["score"] - legacy["score"],
            "receiver_score_delta": principled["receiver_score"] - legacy["receiver_score"],
            "net_score_delta": principled["net_score"] - legacy["net_score"],
            "suit_count_bucket": legacy["suit_count_bucket"],
        })

    def mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    by_bucket = {}
    for bucket in sorted({row["suit_count_bucket"] for row in rows}):
        values = [row for row in rows if row["suit_count_bucket"] == bucket]
        by_bucket[bucket] = {
            "games": len(values),
            "average_score_delta": sum(row["score_delta"] for row in values) / len(values),
            "average_receiver_score_delta": sum(row["receiver_score_delta"] for row in values) / len(values),
            "average_net_score_delta": sum(row["net_score_delta"] for row in values) / len(values),
            "principled_better_score_rate": sum(row["score_delta"] > 0 for row in values) / len(values),
        }

    result = {
        "games": len(rows),
        "dealer": args.dealer,
        "target": args.target,
        "same_action_rate": mean("same_action"),
        "principled_win_rate": sum(row["principled"]["won"] for row in rows) / len(rows) if rows else 0.0,
        "legacy_win_rate": sum(row["legacy"]["won"] for row in rows) / len(rows) if rows else 0.0,
        "average_score_delta": mean("score_delta"),
        "average_receiver_score_delta": mean("receiver_score_delta"),
        "average_net_score_delta": mean("net_score_delta"),
        "principled_better_score_rate": sum(row["score_delta"] > 0 for row in rows) / len(rows) if rows else 0.0,
        "by_suit_count_bucket": by_bucket,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
