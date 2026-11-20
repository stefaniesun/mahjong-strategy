"""辅助器测试用的局面构造工具。"""
from __future__ import annotations

from dataclasses import replace

from assistant.counting import seen_counts
from assistant.state import AssistantState, MeldView, PlayerView, WinRecord


def make_state(
    own,
    *,
    perspective: int = 0,
    rivers: dict | None = None,
    melds: dict | None = None,
    voids: dict | None = None,
    won: dict | None = None,
    wins=(),
    hidden: dict | None = None,
    phase: str = "play",
    current_player: int | None = None,
    **kwargs,
) -> AssistantState:
    """构造一个张数守恒的局面:牌墙张数按"未现 − 对手暗手牌"自动算出。"""
    rivers = rivers or {}
    melds = melds or {}
    voids = voids or {}
    won = won or {}
    hidden = hidden or {}
    players = []
    for seat in range(4):
        seat_melds = tuple(melds.get(seat, ()))
        default_hidden = 13 - 3 * len(seat_melds)
        players.append(
            PlayerView(
                seat=seat,
                melds=seat_melds,
                river=tuple(rivers.get(seat, ())),
                discards=tuple(rivers.get(seat, ())),
                void_suit=voids.get(seat),
                won=won.get(seat, False),
                hidden_count=0 if seat == perspective else hidden.get(seat, default_hidden),
            )
        )
    draft = AssistantState(
        perspective=perspective,
        phase=phase,
        own_hand=tuple(own),
        players=tuple(players),
        wins=tuple(wins),
        wall_count=0,
        current_player=perspective if current_player is None else current_player,
        **kwargs,
    )
    unseen = sum(max(0, 4 - c) for c in seen_counts(draft))
    wall = unseen - sum(view.hidden_count for view in players if view.seat != perspective)
    assert wall >= 0, f"构造的局面张数不对:牌墙会是 {wall}"
    return replace(draft, wall_count=wall)


def pong(tile: str, from_seat: int | None = None) -> MeldView:
    return MeldView("pong", tile, from_seat=from_seat)


def kong(tile: str, concealed: bool = False) -> MeldView:
    return MeldView("kong", tile, concealed=concealed)


__all__ = ["make_state", "pong", "kong", "WinRecord"]
