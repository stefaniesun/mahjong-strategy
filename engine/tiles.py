from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(str, Enum):
    WAN = "W"
    TIAO = "T"
    BING = "B"


SUITS: tuple[Suit, ...] = (Suit.WAN, Suit.TIAO, Suit.BING)


@dataclass(frozen=True, order=True)
class Tile:
    suit: Suit
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.suit, Suit):
            object.__setattr__(self, "suit", Suit(self.suit))
        if self.rank < 1 or self.rank > 9:
            raise ValueError("tile rank must be in 1..9")
        # index 预计算并缓存:牌对象在热点路径被海量使用,property 每次重算曾是 profile 热点。
        object.__setattr__(self, "_index", SUITS.index(self.suit) * 9 + self.rank - 1)

    @property
    def index(self) -> int:
        return self._index

    @classmethod
    def from_index(cls, index: int) -> "Tile":
        if index < 0 or index >= 27:
            raise ValueError("tile index must be in 0..26")
        return _TILES_BY_INDEX[index]


# 全部 27 种牌的驻留单例表:牌是不可变值对象,domain 固定,
# 反复 new 是纯开销(profile 显示 5 局构造 20 万次)。相等性按值比较,与新建实例完全等价。
_TILES_BY_INDEX: tuple[Tile, ...] = tuple(Tile(SUITS[i // 9], i % 9 + 1) for i in range(27))
_TILES_BY_TEXT: dict[str, Tile] = {f"{tile.rank}{tile.suit.value}": tile for tile in _TILES_BY_INDEX}
_TILE_TEXTS: tuple[str, ...] = tuple(f"{tile.rank}{tile.suit.value}" for tile in _TILES_BY_INDEX)


def parse_tile(text: str) -> Tile:
    if isinstance(text, str):
        tile = _TILES_BY_TEXT.get(text)
        if tile is not None:
            return tile
    # 慢路径只为保留与原实现完全一致的报错行为
    if len(text) != 2:
        raise ValueError(f"invalid tile text: {text!r}")
    rank_text, suit_text = text[0], text[1]
    if not rank_text.isdigit():
        raise ValueError(f"invalid tile rank: {text!r}")
    try:
        suit = Suit(suit_text)
    except ValueError as exc:
        raise ValueError(f"invalid tile suit: {text!r}") from exc
    return Tile(suit, int(rank_text))


def tile_to_str(tile: Tile) -> str:
    return _TILE_TEXTS[tile.index]


def full_wall() -> list[Tile]:
    return [tile for tile in _TILES_BY_INDEX for _ in range(4)]
