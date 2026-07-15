from __future__ import annotations

from dataclasses import dataclass, field

from engine.config import RuleConfig
from engine.hand import Hand
from engine.meld import MeldKind
from engine.tiles import Tile
from engine.win_check import is_seven_pairs, is_standard_win


@dataclass(frozen=True)
class WinContext:
    self_draw: bool = False
    after_kong: bool = False
    robbing_kong: bool = False
    haidi: bool = False


@dataclass(frozen=True)
class FanResult:
    fan: int
    score_per_payer: int
    patterns: list[str] = field(default_factory=list)


def _all_tiles(hand: Hand) -> list[Tile]:
    result = hand.tiles()
    for meld in hand.melds:
        result.extend(meld.tiles)
    return result


def _is_pure_suit(hand: Hand) -> bool:
    tiles = _all_tiles(hand)
    return bool(tiles) and len({tile.suit for tile in tiles}) == 1


def _root_count(hand: Hand) -> int:
    counts = list(hand.counts)
    for meld in hand.melds:
        for tile in meld.tiles:
            counts[tile.index] += 1
    return sum(1 for count in counts if count >= 4)


def _all_triplets(counts: tuple[int, ...]) -> bool:
    # 每种牌至多 4 张,故"全刻子"等价于每种牌张数为 0 或 3(4 张会留单张,无法成刻)。
    return all(count % 3 == 0 for count in counts)


def _is_pongs_and_pair(counts: tuple[int, ...]) -> bool:
    # 手内牌能否拆成"若干刻子 + 恰好一对将"。逐个尝试将牌位置,余下必须全是刻子。
    work = list(counts)
    for index, count in enumerate(work):
        if count >= 2:
            work[index] -= 2
            if _all_triplets(tuple(work)):
                return True
            work[index] += 2
    return False


def _is_all_pongs(hand: Hand) -> bool:
    if not is_standard_win(hand):
        return False
    if any(meld.kind == MeldKind.CHOW for meld in hand.melds):
        return False
    # 必须真正存在"全刻子"的拆解,不能只看张数奇偶——两副相同顺子(如 234 234)
    # 每张恰好出现 2 次,会骗过奇偶判断却根本不是对对胡。
    return _is_pongs_and_pair(tuple(hand.counts))


def _is_jin_gou_diao(hand: Hand) -> bool:
    fixed_meld_count = sum(1 for meld in hand.melds if meld.kind in {MeldKind.PONG, MeldKind.KONG})
    return fixed_meld_count == 4 and hand.size == 2 and any(count == 2 for count in hand.counts)


def _score(fan: int, config: RuleConfig) -> int:
    return config.base_score * (2 ** fan)


def calculate_fan(hand: Hand, context: WinContext, config: RuleConfig | None = None) -> FanResult:
    config = config or RuleConfig()
    patterns: list[str] = []
    fan = 0

    if is_seven_pairs(hand):
        fan += 2
        patterns.append("seven_pairs")
    elif _is_jin_gou_diao(hand):
        fan += 2
        patterns.append("jin_gou_diao")
    elif _is_all_pongs(hand):
        fan += 1
        patterns.append("all_pongs")
    elif is_standard_win(hand):
        patterns.append("ping_hu")
    else:
        raise ValueError("hand is not a winning hand")

    if _is_pure_suit(hand):
        fan += 2
        patterns.append("pure_suit")

    roots = _root_count(hand)
    for _ in range(roots):
        fan += 1
        patterns.append("root")

    if context.after_kong:
        fan += 1
        patterns.append("after_kong")
    if context.robbing_kong:
        fan += 1
        patterns.append("robbing_kong")
    if context.haidi:
        fan += 1
        patterns.append("haidi")

    fan = min(fan, config.max_fan)
    score = _score(fan, config)

    if context.self_draw:
        if config.self_draw_mode == "add_fan" and fan < config.max_fan:
            # 加番:未顶番时自摸 +1 番再算分。
            fan += 1
            score = _score(fan, config)
        else:
            # 加底:add_di 模式,以及"加番模式但已顶番"的牌,自摸一律加一个底分
            # (顶番牌自摸不能白摸,封顶封的是番、不封自摸的底)。
            score += config.base_score

    return FanResult(fan=fan, score_per_payer=score, patterns=patterns)
