"""离线换三张场景生成与配对反事实评估。

示例:
    python tools/evaluate_swap_counterfactual.py --games 1000 --out outputs/swap_eval.jsonl

每个初始牌局会枚举目标座位的全部合法换法。所有候选使用同一初始牌局、
同一换牌动作以及同一套对手策略，比较换牌后短期牌效和完整牌局结果。
当前版本先固定目标玩家的换法，其他三家使用 RulePolicy；用于建立可复现基线，
不是最终的真人策略模型。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.driver import run_game
from engine.actions import Action, ActionKind, legal_swap_actions
from engine.hand import Hand
from engine.tiles import SUITS, Tile, tile_to_str
from policies.opponent_pool import GreedyPolicy, PrincipledSwapPolicy
from policies.protocol_actions import actions_from_mask
from policies.rule_policy import RulePolicy
from state.hand_analysis import shanten


def _action_key(action: Action) -> tuple[int, ...]:
    return tuple(tile.index for tile in action.tiles)


def _remove_tiles(hand: Hand, tiles: tuple[Tile, ...]) -> Hand:
    rest = Hand(counts=list(hand.counts), melds=list(hand.melds))
    for tile in tiles:
        rest.remove(tile)
    return rest


def _exchange_hand(hand: Hand, outgoing: tuple[Tile, ...], incoming: tuple[Tile, ...]) -> Hand:
    exchanged = _remove_tiles(hand, outgoing)
    for tile in incoming:
        exchanged.add(tile)
    return exchanged


def _useful_summary(hand: Hand) -> dict[str, int]:
    """用换出后手牌评估每种摸牌对向听的改善,不使用未来牌墙信息。"""
    current = shanten(hand)
    useful = 0
    kinds = 0
    for tile in (Tile.from_index(i) for i in range(27)):
        if hand.count(tile) >= 4:
            continue
        candidate = Hand(counts=list(hand.counts))
        candidate.add(tile)
        if shanten(candidate) < current:
            kinds += 1
            useful += 4 - hand.count(tile)
    return {"shanten": current, "useful_kinds": kinds, "useful_tiles": useful}


def _feature_row(hand: Hand, action: Action, actor: int, receiver: int, dealer: int, direction: int) -> dict:
    rest = _remove_tiles(hand, action.tiles)
    summary = _useful_summary(rest)
    counts = {suit.value: sum(1 for tile in rest.tiles() if tile.suit is suit) for suit in SUITS}
    outgoing_counts = {tile_to_str(tile): action.tiles.count(tile) for tile in set(action.tiles)}
    outgoing_pairs = sum(max(0, count - 1) for count in outgoing_counts.values())
    outgoing_triplet = int(any(count >= 3 for count in outgoing_counts.values()))
    outgoing_middle = sum(int(tile.suit in SUITS and 2 <= tile.rank <= 8) for tile in action.tiles)
    return {
        "swap": [tile_to_str(tile) for tile in action.tiles],
        "swap_suit": action.tiles[0].suit.value,
        "swap_count_by_suit": counts,
        "outgoing_tile_counts": outgoing_counts,
        "outgoing_pair_excess": outgoing_pairs,
        "outgoing_triplet": outgoing_triplet,
        "outgoing_middle_count": outgoing_middle,
        "remaining_hand_size": rest.size,
        "post_swap": summary,
        "actor": actor,
        "actor_is_dealer": actor == dealer,
        "actor_hand_size": hand.size,
        "receiver": receiver,
        "receiver_is_dealer": receiver == dealer,
        "swap_direction": direction,
    }


def _make_policies(policy_mode: str, seed: int) -> list:
    if policy_mode not in {"rule", "principled", "mixed"}:
        raise ValueError(f"unknown policy mode: {policy_mode}")
    if policy_mode == "rule":
        return [RulePolicy() for _ in range(4)]
    if policy_mode == "principled":
        return [PrincipledSwapPolicy() for _ in range(4)]
    rng = random.Random(seed)
    return [PrincipledSwapPolicy() if rng.random() < 0.5 else RulePolicy() for _ in range(4)]


def _play_fixed_swap(seed: int, dealer: int, target: int, action: Action, policies: list, *, capture_swap: bool = False):
    chosen: dict[int, Action] = {}

    def on_decision(game, recorder, player, decision):
        if capture_swap and game.state is not None and game.state.phase == "swap_three":
            chosen[player] = decision.action

    class FixedSwapPolicy(RulePolicy):
        def choose_action(self, protocol_state, legal_mask):
            legal = actions_from_mask(legal_mask)
            swaps = [candidate for candidate in legal if candidate.kind is ActionKind.SWAP_THREE]
            if swaps:
                return next(candidate for candidate in swaps if _action_key(candidate) == _action_key(action))
            return super().choose_action(protocol_state, legal_mask)

    active = list(policies)
    active[target] = FixedSwapPolicy()
    game, recorder = run_game(seed, policies=active, dealer=dealer, on_decision=on_decision if capture_swap else None)
    return game, recorder, chosen


def _scenario(seed: int, dealer: int, target: int, policy_mode: str) -> dict:
    # 先获取同一 seed 的初始牌面和所有候选动作。
    from engine.game import Game

    initial_game = Game(seed=seed, dealer=dealer)
    state = initial_game.reset()
    hand = state.hands[target]
    actions = legal_swap_actions(hand)
    receiver = (target - state.swap_direction) % 4
    candidates = [_feature_row(hand, action, target, receiver, dealer, state.swap_direction) for action in actions]
    receiver_hand = state.hands[receiver]
    receiver_before = _useful_summary(receiver_hand)
    baseline_policies = _make_policies(policy_mode, seed)
    _, _, baseline_swaps = _play_fixed_swap(seed, dealer, target, actions[0], baseline_policies, capture_swap=True)
    results = []
    for action, candidate in zip(actions, candidates):
        policies = _make_policies(policy_mode, seed)
        game, _, _ = _play_fixed_swap(seed, dealer, target, action, policies)
        final = game.state
        candidate = dict(candidate)
        candidate["score"] = final.scores[target]
        candidate["won"] = final.won[target]
        candidate["finished"] = final.finished
        candidate["win_order"] = final.win_order.index(target) + 1 if target in final.win_order else None
        receiver_action = baseline_swaps.get(receiver)
        receiver_outgoing = receiver_action.tiles if receiver_action is not None else tuple()
        receiver_after = _useful_summary(_exchange_hand(receiver_hand, receiver_outgoing, action.tiles))
        candidate["receiver_swap"] = [tile_to_str(tile) for tile in receiver_outgoing]
        candidate["receiver_score"] = final.scores[receiver]
        candidate["receiver_before"] = receiver_before
        candidate["receiver_after"] = receiver_after
        candidate["receiver_shanten_delta"] = receiver_after["shanten"] - receiver_before["shanten"]
        candidate["receiver_useful_tiles_delta"] = receiver_after["useful_tiles"] - receiver_before["useful_tiles"]
        candidate["receiver_helped"] = int(
            receiver_after["shanten"] < receiver_before["shanten"]
            or receiver_after["useful_tiles"] > receiver_before["useful_tiles"]
        )
        candidate["net_score"] = candidate["score"] - candidate["receiver_score"]
        results.append(candidate)
    return {
        "seed": seed,
        "dealer": dealer,
        "target": target,
        "policy_mode": policy_mode,
        "target_is_dealer": target == dealer,
        "initial_hand": [tile_to_str(tile) for tile in hand.tiles()],
        "suit_counts": {suit.value: sum(1 for tile in hand.tiles() if tile.suit is suit) for suit in SUITS},
        "swap_direction": state.swap_direction,
        "receiver": receiver,
        "candidate_count": len(actions),
        "candidates": results,
    }


def _mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, 1.96 * sd / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并评估换三张反事实场景")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--dealer", type=int, choices=range(4), default=0)
    parser.add_argument("--target", type=int, choices=range(4), default=0)
    parser.add_argument("--policy-mode", choices=("rule", "principled", "mixed"), default="rule")
    parser.add_argument("--out", type=Path, default=Path("outputs/swap_counterfactual.jsonl"))
    parser.add_argument("--append", action="store_true", help="追加到已有 JSONL；默认覆盖旧文件")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        args.out.write_text("", encoding="utf-8")
    rows = []
    for offset in range(args.games):
        row = _scenario(args.seed_start + offset, args.dealer, args.target, args.policy_mode)
        rows.append(row)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores = [candidate["score"] for row in rows for candidate in row["candidates"]]
    mean, half = _mean_ci(scores)
    print(json.dumps({"games": len(rows), "candidates": len(scores), "average_candidate_score": mean, "ci95": [mean - half, mean + half], "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
