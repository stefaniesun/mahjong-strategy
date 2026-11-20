"""定缺分布反事实统计：固定换牌结果，只替换目标玩家定缺。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import Action, ActionKind, legal_swap_actions
from engine.game import Game
from engine.hand import Hand
from engine.tiles import SUITS, Suit, tile_to_str
from policies.heuristics import choose_swap_tiles, choose_swap_tiles_legacy, choose_void_suit


@dataclass(frozen=True)
class VoidDistribution:
    target: int
    voids: tuple[Suit, Suit, Suit, Suit]
    distribution: dict[str, int]
    single_lane: bool
    opponent_single_lane: bool
    single_lane_suit: Suit | None
    unique_non_void_player: int | None


def classify_void_distribution(*, target: int, voids: tuple[Suit, Suit, Suit, Suit]) -> VoidDistribution:
    counts = Counter(voids)
    distribution = {suit.value: counts.get(suit, 0) for suit in SUITS}
    lanes = [suit for suit in SUITS if counts.get(suit, 0) == 3]
    opponent_lane = [suit for suit in lanes if voids[target] is not suit]
    lane = lanes[0] if lanes else None
    unique = next((player for player, suit in enumerate(voids) if lane is not None and suit is not lane), None)
    return VoidDistribution(
        target=target,
        voids=voids,
        distribution=distribution,
        single_lane=bool(lanes),
        opponent_single_lane=bool(opponent_lane),
        single_lane_suit=lane,
        unique_non_void_player=unique,
    )


def _suit_counts(hand: Hand) -> tuple[int, int, int]:
    return tuple(sum(1 for tile in hand.tiles() if tile.suit is suit) for suit in SUITS)


def _bucket(hand: Hand) -> str:
    return "/".join(str(value) for value in sorted(_suit_counts(hand)))


def _action_for_swap(hand: Hand, *, policy: str) -> Action:
    chooser = choose_swap_tiles if policy == "principled" else choose_swap_tiles_legacy
    selected = tuple(sorted(chooser(hand)))
    for action in legal_swap_actions(hand):
        if tuple(sorted(action.tiles)) == selected:
            return action
    raise RuntimeError("swap chooser returned an illegal action")


def _prepare(seed: int, dealer: int, target: int, policy: str) -> tuple[Game, tuple[Action, ...], tuple[Hand, ...], str]:
    game = Game(seed=seed, dealer=dealer)
    state = game.reset()
    before = tuple(Hand(counts=list(hand.counts), melds=list(hand.melds)) for hand in state.hands)
    actions: list[Action] = []
    for player, hand in enumerate(state.hands):
        actions.append(_action_for_swap(hand, policy=policy))
    target_out = actions[target].tiles[0].suit.value
    for player, action in enumerate(actions):
        game.step(player, action)
    incoming_giver = (target + state.swap_direction) % 4
    incoming = actions[incoming_giver].tiles[0].suit.value
    collision = target_out == incoming
    return game, tuple(actions), before, str(collision).lower()


def _scenario(seed: int, dealer: int, target: int, policy: str) -> dict:
    game, actions, before, collision_text = _prepare(seed, dealer, target, policy)
    state = game.state
    assert state is not None and state.phase == "declare_void"
    target_hand = state.hands[target]
    base_voids = tuple(choose_void_suit(hand) for hand in state.hands)
    receiver = (target - state.swap_direction) % 4
    incoming_giver = (target + state.swap_direction) % 4
    rows = []
    for selected in SUITS:
        voids = list(base_voids)
        voids[target] = selected
        result = classify_void_distribution(target=target, voids=tuple(voids))
        rows.append({
            "my_void_suit": selected.value,
            "choice_relation": "outgoing" if selected.value == actions[target].tiles[0].suit.value else ("incoming" if selected.value == actions[incoming_giver].tiles[0].suit.value else "third"),
            "voids": [suit.value for suit in result.voids],
            "void_distribution": result.distribution,
            "single_lane": result.single_lane,
            "opponent_single_lane": result.opponent_single_lane,
            "single_lane_suit": result.single_lane_suit.value if result.single_lane_suit else None,
            "unique_non_void_player": result.unique_non_void_player,
        })
    return {
        "seed": seed,
        "dealer": dealer,
        "target": target,
        "policy": policy,
        "target_is_dealer": target == dealer,
        "target_hand_size": target_hand.size,
        "suit_count_bucket": _bucket(target_hand),
        "collision": collision_text == "true",
        "collision_suit": actions[target].tiles[0].suit.value if collision_text == "true" else None,
        "outgoing_suit": actions[target].tiles[0].suit.value,
        "incoming_suit": actions[incoming_giver].tiles[0].suit.value,
        "outgoing_equals_incoming": actions[target].tiles[0].suit is actions[incoming_giver].tiles[0].suit,
        "target_hand": [tile_to_str(tile) for tile in target_hand.tiles()],
        "rows": rows,
    }


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for option in row["rows"]:
            key = f"{row['target_hand_size']}张|{row['suit_count_bucket']}|{'collision' if row['collision'] else 'non_collision'}|{option['choice_relation']}"
            grouped[key].append(option)
    result = {}
    for key, values in sorted(grouped.items()):
        result[key] = {
            "scenarios": len(values),
            "opponent_single_lane_rate": sum(item["opponent_single_lane"] for item in values) / len(values),
            "single_lane_rate": sum(item["single_lane"] for item in values) / len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="定缺分布反事实统计")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=990000)
    parser.add_argument("--dealer", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--policy", choices=("legacy", "principled"), default="principled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [_scenario(args.seed_start + offset, args.dealer, args.target, args.policy) for offset in range(args.games)]
    payload = {
        "schema_version": 1,
        "games": args.games,
        "seed_start": args.seed_start,
        "counting_basis": "post_swap_declare_void_hand_size",
        "suit_count_bucket_basis": "sorted_counts_with_actual_hand_size",
        "seed_end": args.seed_start + args.games - 1,
        "dealer": args.dealer,
        "target": args.target,
        "policy": args.policy,
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
