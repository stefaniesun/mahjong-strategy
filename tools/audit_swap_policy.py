"""审计换三张策略的基本行为，不跑完整牌局。

示例:
    python tools/audit_swap_policy.py --games 10000 --policy principled
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import legal_swap_actions
from engine.game import Game
from engine.hand import Hand
from engine.tiles import SUITS, Tile
from policies.heuristics import choose_swap_tiles, choose_swap_tiles_legacy, choose_void_suit
from state.hand_analysis import shanten


def _remove(hand: Hand, tiles) -> Hand:
    result = Hand(counts=list(hand.counts), melds=list(hand.melds))
    for tile in tiles:
        result.remove(tile)
    return result


def _summary(hand: Hand) -> tuple[int, int, int]:
    after = _remove(hand, tuple())
    current = shanten(after)
    useful = 0
    for index in range(27):
        tile = Tile.from_index(index)
        if after.count(tile) < 4:
            candidate = Hand(counts=list(after.counts), melds=list(after.melds))
            candidate.add(tile)
            if shanten(candidate) < current:
                useful += 4 - after.count(tile)
    return current, useful, sum(1 for tile in hand.tiles() if hand.count(tile) >= 2)


def _bucket_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)
    result = {}
    for bucket, values in sorted(grouped.items()):
        count = len(values)
        result[bucket] = {
            "scenarios": count,
            "average_shanten_delta": sum(row["shanten_delta"] for row in values) / count,
            "average_useful_tiles_delta": sum(row["useful_delta"] for row in values) / count,
            "average_shanten_gap_to_best_candidate": sum(row["shanten_gap"] for row in values) / count,
            "selected_shanten_worse_than_best_rate": sum(row["shanten_gap"] > 0 for row in values) / count,
            "pair_break_rate": sum(row["broke_pair"] for row in values) / count,
            "void_suit_match_rate": sum(row["void_match"] for row in values) / count,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="审计基本换牌策略")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--policy", choices=("principled", "legacy"), default="principled")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    counters = Counter()
    selected_shantens: list[int] = []
    selected_useful: list[int] = []
    rank_gaps: list[int] = []
    examples = []
    stratified_rows: list[dict[str, float]] = []
    for offset in range(args.games):
        seed = args.seed_start + offset
        game = Game(seed=seed, dealer=seed % 4)
        state = game.reset()
        for actor, hand in enumerate(state.hands):
            actions = legal_swap_actions(hand)
            if not actions:
                counters["no_legal_swap"] += 1
                continue
            chooser = choose_swap_tiles if args.policy == "principled" else choose_swap_tiles_legacy
            preferred = chooser(hand)
            selected = next(action for action in actions if tuple(sorted(action.tiles)) == tuple(sorted(preferred)))
            before_shanten, before_useful, _ = _summary(hand)
            after = _remove(hand, selected.tiles)
            after_shanten, after_useful, _ = _summary(after)
            remaining_scores = [_summary(_remove(hand, action.tiles))[0] for action in actions]
            best_remaining = min(remaining_scores)
            rank_gaps.append(after_shanten - best_remaining)
            selected_counts = Counter(selected.tiles)
            pair_excess = sum(max(0, count - 1) for count in selected_counts.values())
            broke_pair = any(hand.count(tile) >= 2 and selected_counts[tile] < hand.count(tile) for tile in selected_counts)
            counters["scenarios"] += 1
            suit_bucket = "/".join(str(sum(1 for tile in hand.tiles() if tile.suit is suit)) for suit in SUITS)
            counters[f"suit_count_{suit_bucket}"] += 1
            counters["selected_pair_excess"] += pair_excess
            counters["selected_broke_pair"] += int(broke_pair)
            counters["selected_middle_tiles"] += sum(2 <= tile.rank <= 8 for tile in selected.tiles)
            void_match = int(selected.tiles[0].suit is choose_void_suit(hand))
            counters["preferred_void_match"] += void_match
            selected_shantens.append(after_shanten - before_shanten)
            selected_useful.append(after_useful - before_useful)
            stratified_rows.append({
                "bucket": f"{'dealer' if actor == state.dealer else 'non_dealer'}|{'/'.join(sorted(suit_bucket.split('/'), key=int))}",
                "shanten_delta": after_shanten - before_shanten,
                "useful_delta": after_useful - before_useful,
                "shanten_gap": after_shanten - best_remaining,
                "broke_pair": int(broke_pair),
                "void_match": void_match,
            })
            if len(examples) < 10 and (broke_pair or after_shanten > before_shanten):
                examples.append({
                    "seed": seed,
                    "actor": actor,
                    "hand": [f"{tile.rank}{tile.suit.value}" for tile in hand.tiles()],
                    "selected": [f"{tile.rank}{tile.suit.value}" for tile in selected.tiles],
                    "before_shanten": before_shanten,
                    "after_shanten": after_shanten,
                    "before_useful": before_useful,
                    "after_useful": after_useful,
                    "broke_pair": broke_pair,
                })

    scenarios = counters["scenarios"]
    result = {
        "games": args.games,
        "by_role_and_suit_count": _bucket_stats(stratified_rows),
        "scenarios": scenarios,
        "policy": args.policy,
        "average_shanten_delta": sum(selected_shantens) / len(selected_shantens) if selected_shantens else 0.0,
        "average_useful_tiles_delta": sum(selected_useful) / len(selected_useful) if selected_useful else 0.0,
        "average_shanten_gap_to_best_candidate": sum(rank_gaps) / len(rank_gaps) if rank_gaps else 0.0,
        "selected_shanten_worse_than_best_rate": sum(gap > 0 for gap in rank_gaps) / len(rank_gaps) if rank_gaps else 0.0,
        "pair_break_rate": counters["selected_broke_pair"] / scenarios if scenarios else 0.0,
        "average_selected_pair_excess": counters["selected_pair_excess"] / scenarios if scenarios else 0.0,
        "average_selected_middle_tiles": counters["selected_middle_tiles"] / scenarios if scenarios else 0.0,
        "void_suit_match_rate": counters["preferred_void_match"] / scenarios if scenarios else 0.0,
        "examples": examples,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
