"""换三张带来的确定信息:你换给某家的牌,只要还没从他手里出来,就一定还在他的暗手牌里。

换牌发生在开局,之后牌离开一个人的手只有几条路,而且都摆在明面上:打出(进牌河或被碰杠走)、
碰杠时从手里拿出、补杠被抢杠胡。所以对每种牌:
    他手里至少还有 = 你换给他的张数 − 他打出过的张数 − 他副露里从手里拿出的张数 − 被抢杠胡抢走的张数
这是严格的下限:他打出的同种牌就算其实是他原来的那张,只会让真实张数比下限更多,不会更少。
"""
from __future__ import annotations

from assistant.state import AssistantState, MeldView
from assistant.tiles import label, tidx


def _from_hand(meld: MeldView) -> int:
    """这组副露里有几张是从他自己手里拿出来的。"""
    if meld.kind == "pong":
        return 2
    if meld.concealed:
        return 4
    return 3  # 直明杠 3 张;补杠 = 碰时的 2 张 + 后加的 1 张


def swap_floors(state: AssistantState) -> dict[int, tuple[int, ...]]:
    """{座位: 每种牌在他暗手牌里一定有的最少张数}。只在行牌阶段、且换给的是别人时给出。"""
    swap = state.swap
    if swap is None or state.phase != "play" or swap.given_to == state.perspective:
        return {}
    view = state.players[swap.given_to]
    floor = [0] * 27
    for tile in swap.given:
        floor[tidx(tile)] += 1
    for tile in view.discards:
        floor[tidx(tile)] -= 1
    for meld in view.melds:
        floor[tidx(meld.tile)] -= _from_hand(meld)
    for record in state.wins:
        if record.robbing_kong and record.from_seat == view.seat and record.tile is not None:
            floor[tidx(record.tile)] -= 1
    floor = [max(0, c) for c in floor]
    if not any(floor) or sum(floor) > view.hidden_count:
        # 张数对不上说明输入有误,宁可不给结论
        return {}
    return {view.seat: tuple(floor)}


def floor_text(floor: tuple[int, ...]) -> str:
    return "、".join(label(t) + (f"×{c}" if c > 1 else "") for t, c in enumerate(floor) if c)
