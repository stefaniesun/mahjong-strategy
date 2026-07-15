"""大规模交叉验证:S2 legality 纯函数 vs S1 引擎 legal_actions,并顺带守恒/不卡死检查。

S2 spec 要求"逐决策点比对 legal_mask 与引擎、零不一致";原有单测只有少量手工场景,
这里用随机合法动作驱动大量真实对局,把该验收沉淀成回归网。
"""
from __future__ import annotations

import random

from engine.game import Game
from state.adapters.from_engine import from_engine
from state.legality import legal_actions as s2_legal_actions


# 60 局约覆盖 7000+ 决策点,足以做回归网又不拖垮默认套件;
# 需要更强保证时把 GAMES 调到 1000+ 单独跑。
GAMES = 60
MAX_STEPS = 1000


def _engine_kinds(actions) -> set:
    result = set()
    for action in actions:
        tile = None if action.tile is None else f"{action.tile.rank}{action.tile.suit.value}"
        kong_kind = None if action.kong_kind is None else action.kong_kind.value
        result.add((action.kind.value, tile, kong_kind))
    return result


def _protocol_kinds(action_dicts) -> set:
    return {(item["kind"], item.get("tile"), item.get("kong_kind")) for item in action_dicts}


def _select_actor(game: Game):
    state = game.state
    if state.phase in ("swap_three", "declare_void"):
        for player in range(4):
            if state.phase == "swap_three" and state.swap_choices[player] is None:
                return player
            if state.phase == "declare_void" and state.void_suits[player] is None:
                return player
        return None
    if state.pending_rob_kong is not None:
        return state.pending_rob_kong.winners[0]
    if state.pending_discard is not None:
        if state.pending_winners:
            return state.pending_winners[0]
        for offset in range(1, 4):
            player = (state.pending_discard.discarder + offset) % 4
            if game.legal_actions(player):
                return player
        return state.pending_discard.discarder
    if state.phase == "play":
        return state.current_player
    return None


def test_legality_matches_engine_and_conservation_over_many_games():
    mismatches = 0
    zero_sum_violations = 0
    unfinished = 0
    checked_points = 0

    for seed in range(GAMES):
        game = Game(seed=seed)
        state = game.reset()
        rng = random.Random(seed)
        steps = 0

        while not state.finished and steps < MAX_STEPS:
            steps += 1
            actor = _select_actor(game)
            if actor is None:
                break
            engine_actions = game.legal_actions(actor)
            if not engine_actions:
                break

            # 只在对局/抢杠/点炮响应阶段比对(换三张/定缺阶段 legality 语义不同)
            if state.phase == "play" or state.pending_rob_kong is not None or state.pending_discard is not None:
                protocol_state = from_engine(state, player_id=actor)
                s2_actions = s2_legal_actions(protocol_state)
                if _protocol_kinds(s2_actions) != _engine_kinds(engine_actions):
                    mismatches += 1
                checked_points += 1

            game.step(actor, rng.choice(engine_actions))

        if steps >= MAX_STEPS and not state.finished:
            unfinished += 1
        if sum(state.scores) != 0:
            zero_sum_violations += 1

    assert checked_points > 5000, f"覆盖的决策点太少: {checked_points}"
    assert mismatches == 0, f"legality 与引擎不一致 {mismatches} 处(共 {checked_points} 点)"
    assert zero_sum_violations == 0, f"零和被破坏 {zero_sum_violations} 局"
    assert unfinished == 0, f"有 {unfinished} 局跑满 {MAX_STEPS} 步仍未结束(疑似卡死)"
