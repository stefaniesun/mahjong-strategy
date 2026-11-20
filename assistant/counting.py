"""计数层:已现、未现、绝张、未知区规模,以及守恒校验。"""
from __future__ import annotations

from dataclasses import dataclass

from assistant.state import AssistantState
from assistant.tiles import label, tidx


@dataclass(frozen=True)
class TileCounts:
    seen: tuple[int, ...]  # 27 种牌各自已现张数(视角玩家看得见的)
    unseen: tuple[int, ...]  # 4 - 已现
    total_unseen: int
    wall: int
    hidden: dict[int, int]  # 对手座位 -> 看不见的暗手牌张数

    @property
    def exhausted(self) -> list[int]:
        return [index for index, count in enumerate(self.unseen) if count == 0]


def seen_counts(state: AssistantState) -> list[int]:
    counts = [0] * 27
    for tile in state.own_hand:
        counts[tidx(tile)] += 1
    for view in state.players:
        for tile in view.river:
            counts[tidx(tile)] += 1
        for meld in view.melds:
            counts[tidx(meld.tile)] += meld.size
    for record in state.wins:
        # 点炮胡的那张还在放炮者牌河里,已计入;自摸和抢杠胡的那张是单独亮出来的。
        # 视角玩家自己自摸的那张在自己手里,已计入。
        if record.tile is not None and (record.self_draw or record.robbing_kong) and record.seat != state.perspective:
            counts[tidx(record.tile)] += 1
    return counts


def compute_counts(state: AssistantState) -> TileCounts:
    seen = seen_counts(state)
    unseen = tuple(max(0, 4 - count) for count in seen)
    hidden = {view.seat: view.hidden_count for view in state.opponents()}
    return TileCounts(
        seen=tuple(seen),
        unseen=unseen,
        total_unseen=sum(unseen),
        wall=state.wall_count,
        hidden=hidden,
    )


def conservation_errors(state: AssistantState) -> list[str]:
    """录错一张牌就可能算出假绝张,所以任何不守恒都要报警并停止输出确定结论。"""
    errors: list[str] = []
    seen = seen_counts(state)
    for index, count in enumerate(seen):
        if count > 4:
            errors.append(f"{label(index)} 已现 {count} 张,超过 4 张,牌面信息有误")
    total_unseen = sum(max(0, 4 - count) for count in seen)
    hidden_total = sum(view.hidden_count for view in state.opponents())
    expected = state.wall_count + hidden_total
    if total_unseen != expected:
        errors.append(
            f"张数对不上:未现 {total_unseen} 张,但牌墙 {state.wall_count} 张 + 对手暗手牌 {hidden_total} 张 = {expected} 张"
        )
    return errors


def seen_breakdown(state: AssistantState) -> list[dict[str, int]]:
    """每种牌已现张数的来源:自己手里、牌河、副露、单独亮出的胡牌张。"""
    rows = [{"own": 0, "river": 0, "meld": 0, "win": 0} for _ in range(27)]
    for tile in state.own_hand:
        rows[tidx(tile)]["own"] += 1
    for view in state.players:
        for tile in view.river:
            rows[tidx(tile)]["river"] += 1
        for meld in view.melds:
            rows[tidx(meld.tile)]["meld"] += meld.size
    for record in state.wins:
        if record.tile is not None and (record.self_draw or record.robbing_kong) and record.seat != state.perspective:
            rows[tidx(record.tile)]["win"] += 1
    return rows


def breakdown_text(row: dict[str, int]) -> str:
    names = {"own": "你手里", "river": "牌河", "meld": "副露", "win": "亮出的胡牌张"}
    return "、".join(f"{names[key]} {count}" for key, count in row.items() if count)
