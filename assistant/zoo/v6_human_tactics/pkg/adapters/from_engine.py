"""引擎状态 -> 辅助器状态(某位玩家视角)。

关键:按真实打法,胡牌者只亮胡牌张、手牌扣着。这里只取胡牌者的张数,绝不读取其牌面。
"""
from __future__ import annotations

from engine.config import RuleConfig, active_rule_config
from engine.meld import MeldKind
from engine.state import GameState
from engine.tiles import tile_to_str

from assistant.recorder import PublicRecorder
from ..state import AssistantState, MeldView, PlayerView, SwapInfo


def build_state(
    engine_state: GameState,
    recorder: PublicRecorder,
    perspective: int,
    rule_config: RuleConfig | None = None,
) -> AssistantState:
    s = engine_state
    own = [tile_to_str(t) for t in s.hands[perspective].tiles()]

    pre_play = s.phase in ("swap_three", "declare_void")
    players = []
    for seat in range(4):
        hand = s.hands[seat]
        melds = tuple(
            MeldView(
                kind="pong" if meld.kind is MeldKind.PONG else "kong",
                tile=tile_to_str(meld.tiles[0]),
                concealed=meld.kind is MeldKind.KONG and not meld.exposed,
                from_seat=meld.from_player,
                added=(seat, tile_to_str(meld.tiles[0])) in recorder.added_kongs,
                income=recorder.kong_income.get((seat, tile_to_str(meld.tiles[0]))) if meld.kind is MeldKind.KONG else None,
            )
            for meld in hand.melds
        )
        if seat == perspective:
            hidden = 0
        elif s.won[seat]:
            hidden = 13 - 3 * len(hand.melds)
        else:
            hidden = hand.size
            if s.phase == "swap_three" and s.swap_choices[seat] is not None:
                hidden += len(s.swap_choices[seat])  # 已选好换出的牌还在途中,仍是他看不见的牌
        void = s.void_suits[seat]
        if pre_play and seat != perspective:
            void = None  # 定缺是同时亮出的,定缺阶段看不到别人的选择
        players.append(
            PlayerView(
                seat=seat,
                melds=melds,
                river=tuple(tile_to_str(t) for t in s.rivers[seat]),
                discards=tuple(recorder.discards[seat]),
                void_suit=void.value if void is not None else None,
                won=s.won[seat],
                hidden_count=hidden,
                discard_tsumogiri=tuple(recorder.discard_tsumogiri[seat]),
                passed_since_tedashi=tuple(sorted(recorder.passed_since_tedashi[seat])),
            )
        )

    swap = None
    if s.phase != "swap_three" and all(choice is not None for choice in s.swap_choices):
        giver = (perspective + s.swap_direction) % 4
        swap = SwapInfo(
            given=tuple(tile_to_str(t) for t in s.swap_choices[perspective]),
            given_to=(perspective - s.swap_direction) % 4,
            received=tuple(tile_to_str(t) for t in s.swap_choices[giver]),
            received_from=giver,
        )

    is_current = s.current_player == perspective
    return AssistantState(
        perspective=perspective,
        phase=s.phase,
        own_hand=tuple(own),
        players=tuple(players),
        wins=tuple(recorder.wins),
        wall_count=len(s.wall),
        current_player=s.current_player,
        pending_discard=(s.pending_discard.discarder, tile_to_str(s.pending_discard.tile)) if s.pending_discard else None,
        pending_after_kong=bool(s.pending_discard and s.pending_discard_after_kong),
        pending_last_wall=bool(s.pending_discard and s.pending_discard_last_wall),
        pending_rob_kong=(s.pending_rob_kong.kong_player, tile_to_str(s.pending_rob_kong.tile)) if s.pending_rob_kong else None,
        draw_after_kong=is_current and s.current_draw_after_kong,
        draw_was_last=is_current and s.current_draw_last_wall,
        passed_hu_lock=s.passed_hu_lock[perspective],
        passed_fan=s.passed_fan[perspective],
        swap=swap,
        rule_config=rule_config or s.rule_config or active_rule_config(),
        swap_direction=s.swap_direction,
        dealer=s.dealer,
    )
