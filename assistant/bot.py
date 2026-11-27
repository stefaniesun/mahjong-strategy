"""基于辅助器全套策略的完整玩家 AI (AssistantPolicy)。

严格通过 build_state(game_state, recorder, actor) 仅读取该座位视角可见的公开信息,
绝不偷看牌墙或他家暗手牌; 全流程复用辅助器已标定的五大模块并附带受控随机性(防过拟合/保持牌局多样性):
- 换三张: sample_swap_v2_indices
- 定缺:   sample_void_v3_idx
- 读牌:   opponent_read + winner_inference + swap_floors
- 碰/杠:  tactics.advise_pong / advise_kong
- 出牌:   discard_v2 (score_discard_candidates_v2 + sample_discard_v2_index)
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.state import GameState
from engine.tiles import SUITS, Tile

from assistant.adapters.from_engine import build_state
from assistant.alerts import _best_shanten_after_discard
from assistant.counting import compute_counts
from assistant.danger import analyze_danger
from assistant.discard_v2 import (
    DEFAULT_DISCARD_V2_CONFIG,
    DiscardV2Config,
    sample_discard_v2_index,
    score_discard_candidates_v2,
)
from assistant.known_holdings import swap_floors
from assistant.opponent_read import read_all_opponents
from assistant.own_hand import analyze_own_hand, build_own_hand, fast_shanten_hand
from assistant.recorder import PublicRecorder
from assistant.swap_v2 import choose_swap_v2_indices, sample_swap_v2_indices
from assistant.tactics import advise_kong, advise_pong
from assistant.tiles import from_label, label, tidx
from assistant.void_v3 import choose_void_v3_idx, sample_void_v3_idx
from assistant.winner_inference import infer_winners
from policies.base_policy import BasePolicy
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy


class AssistantPolicy(BasePolicy):
    """完整辅助器 AI 策略(支持确定性模式或受控随机抽样模式)。"""

    def __init__(
        self,
        *,
        stochastic: bool = True,
        seed: int | None = None,
        discard_config: DiscardV2Config = DEFAULT_DISCARD_V2_CONFIG,
        exact_swap_live: bool = True,
    ) -> None:
        self.stochastic = stochastic
        self.rng = random.Random(seed)
        self.discard_config = discard_config
        self.exact_swap_live = exact_swap_live
        self._fallback = RulePlusPolicy(PlusConfig.parse("all"))

    def choose_action(self, protocol_state, legal_mask: Sequence[bool]) -> Action:
        # 若通过旧协议接口调用(缺少完整 recorder 上下文), 安全回退到 RulePlusPolicy
        return self._fallback.choose_action(protocol_state, legal_mask)

    def choose_from_game(
        self,
        engine_state: GameState,
        recorder: PublicRecorder,
        actor: int,
        legal_actions: Sequence[Action],
    ) -> Action:
        if not legal_actions:
            raise ValueError("no legal actions")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 1) 能胡必胡(抢杠胡 / 点炮胡 / 自摸)
        for kind in (ActionKind.ROB_KONG_WIN, ActionKind.WIN, ActionKind.SELF_WIN):
            for act in legal_actions:
                if act.kind is kind:
                    return act

        # 2) 换三张 (sample_swap_v2)
        if engine_state.phase == "swap_three":
            recv = (actor - engine_state.swap_direction) % 4
            if self.stochastic:
                idxs = sample_swap_v2_indices(
                    engine_state.hands[actor].counts,
                    rng=self.rng,
                    is_dealer=(engine_state.dealer == actor),
                    receiver_is_dealer=(engine_state.dealer == recv),
                    swap_direction=engine_state.swap_direction,
                    exact_live=self.exact_swap_live,
                )
            else:
                idxs = choose_swap_v2_indices(
                    engine_state.hands[actor].counts,
                    is_dealer=(engine_state.dealer == actor),
                    receiver_is_dealer=(engine_state.dealer == recv),
                    swap_direction=engine_state.swap_direction,
                    exact_live=self.exact_swap_live,
                )
            wanted = sorted(idxs)
            for act in legal_actions:
                if act.kind is ActionKind.SWAP_THREE and sorted(t.index for t in act.tiles) == wanted:
                    return act
            return legal_actions[0]

        # 3) 换后定缺 (sample_void_v3)
        if engine_state.phase == "declare_void":
            out_s = SUITS.index(engine_state.swap_choices[actor][0].suit)
            giver = (actor + engine_state.swap_direction) % 4
            in_s = SUITS.index(engine_state.swap_choices[giver][0].suit)
            if self.stochastic:
                vi = sample_void_v3_idx(
                    engine_state.hands[actor].counts,
                    rng=self.rng,
                    is_dealer=(engine_state.dealer == actor),
                    swap_out_suit=out_s,
                    swap_in_suit=in_s,
                    swap_direction=engine_state.swap_direction,
                )
            else:
                vi = choose_void_v3_idx(
                    engine_state.hands[actor].counts,
                    is_dealer=(engine_state.dealer == actor),
                    swap_out_suit=out_s,
                    swap_in_suit=in_s,
                    swap_direction=engine_state.swap_direction,
                )
            target_suit = SUITS[vi]
            for act in legal_actions:
                if act.kind is ActionKind.DECLARE_VOID and act.suit is target_suit:
                    return act
            return legal_actions[0]

        # 4) 行牌阶段: 严格通过 build_state 构造仅本家可见的状态
        state = build_state(engine_state, recorder, actor)
        counts = compute_counts(state)
        floors = swap_floors(state)
        winners = infer_winners(state, counts, floors=floors)
        opp_reads = read_all_opponents(state, floors)
        danger_rows, _ = analyze_danger(state, counts, winners, floors, opponent_reads=opp_reads)
        hand = build_own_hand(state)
        void_suit = engine_state.void_suits[actor]
        void_idx = SUITS.index(void_suit) if void_suit is not None else None

        risk_by_lbl = {r["tile"]: r.get("total_dealin_prob", 0.0) for r in danger_rows}
        risk_by_idx = {tidx(from_label(t)): p for t, p in risk_by_lbl.items()}
        locked = [0] * 27
        for inf in winners:
            for t in range(27):
                locked[t] += inf.certain[t]
        eff_unseen = tuple(max(0, counts.unseen[t] - locked[t]) for t in range(27))

        # 4a) 响应别人弃牌: 明杠 / 碰 / 过
        if engine_state.pending_discard is not None:
            tile = engine_state.pending_discard.tile
            discarder = engine_state.pending_discard.discarder
            kong_act = next((a for a in legal_actions if a.kind is ActionKind.KONG), None)
            pong_act = next((a for a in legal_actions if a.kind is ActionKind.PONG), None)
            pass_act = next((a for a in legal_actions if a.kind is ActionKind.PASS), legal_actions[0])

            if kong_act is not None:
                sh_b = fast_shanten_hand(hand.counts, len(hand.melds), void_idx)
                c_a = list(hand.counts)
                c_a[tile.index] -= 3
                sh_a = fast_shanten_hand(c_a, len(hand.melds) + 1, void_idx)
                ka = advise_kong(state, hand, tile, "exposed", sh_b, sh_a, opp_reads)
                if ka["verdict"] in ("建议明杠",):
                    return kong_act

            if pong_act is not None:
                pa = advise_pong(
                    state,
                    hand,
                    tile,
                    discarder,
                    eff_unseen,
                    risk_by_idx,
                    counts=counts,
                    danger_rows=danger_rows,
                    opp_reads=opp_reads,
                    floors=floors,
                )
                if pa["verdict"] in ("强烈建议碰", "建议碰", "可碰"):
                    return pong_act
            return pass_act

        # 4b) 自己摸牌回合: 先判暗杠 / 补杠
        kong_acts = [a for a in legal_actions if a.kind is ActionKind.KONG and a.tile is not None]
        if kong_acts:
            sh_b = _best_shanten_after_discard(hand, void_suit)
            for ka_act in kong_acts:
                ktile = ka_act.tile
                assert ktile is not None
                ktype = "concealed" if ka_act.kong_kind is GangKind.CONCEALED else "added"
                c_a = list(hand.counts)
                c_a[ktile.index] -= 4 if ktype == "concealed" else 1
                m_a = len(hand.melds) + (1 if ktype == "concealed" else 0)
                sh_a = fast_shanten_hand(c_a, m_a, void_idx)
                adv = advise_kong(
                    state,
                    hand,
                    ktile,
                    ktype,
                    sh_b,
                    sh_a,
                    opp_reads,
                    tile_dealin_prob=risk_by_lbl.get(label(ktile.index), 0.0),
                )
                if adv["verdict"] in ("建议暗杠", "建议补杠"):
                    return ka_act

        # 4c) 出牌 (discard_v2 + 紧致受控随机性: 每局12-15次弃牌, 仅在极接近 <=0.10 EV 时探索次优, 避免随机叠加掉分)
        own_info, _ = analyze_own_hand(
            state, counts, winners=winners, floors=floors, danger_rows=danger_rows, opp_reads=opp_reads
        )
        if self.discard_config == DEFAULT_DISCARD_V2_CONFIG and own_info.get("discard_v2_options"):
            ranked = own_info["discard_v2_options"]
        else:
            ranked = score_discard_candidates_v2(
                state, hand, counts, own_info["candidates"], danger_rows, opp_reads, config=self.discard_config, floors=floors
            )
        if self.stochastic:
            chosen_idx = sample_discard_v2_index(ranked, rng=self.rng, top_k=2, max_gap=0.10, temperature=0.03)
        else:
            chosen_idx = ranked[0]["index"]

        target_tile = Tile.from_index(chosen_idx)
        for act in legal_actions:
            if act.kind is ActionKind.DISCARD and act.tile == target_tile:
                return act
        return next((a for a in legal_actions if a.kind is ActionKind.DISCARD), legal_actions[0])
