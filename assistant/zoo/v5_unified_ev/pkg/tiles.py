"""辅助器内部用的牌面小工具:文本/编号互转与中文显示。"""
from __future__ import annotations

from engine.tiles import Tile, parse_tile, tile_to_str

SUIT_ORDER: tuple[str, ...] = ("W", "T", "B")
SUIT_NAMES: dict[str, str] = {"W": "万", "T": "条", "B": "筒"}


def tidx(text: str) -> int:
    return parse_tile(text).index


def ttext(index: int) -> str:
    return tile_to_str(Tile.from_index(index))


def suit_of(index: int) -> str:
    return SUIT_ORDER[index // 9]


def rank_of(index: int) -> int:
    return index % 9 + 1


def label(tile: str | int) -> str:
    index = tile if isinstance(tile, int) else tidx(tile)
    return f"{rank_of(index)}{SUIT_NAMES[suit_of(index)]}"


def from_label(text: str) -> str:
    """"3万" -> "3W"。"""
    code = {name: suit for suit, name in SUIT_NAMES.items()}[text[-1]]
    return f"{text[:-1]}{code}"


def labels(tiles) -> str:
    return "、".join(label(tile) for tile in tiles)


def suit_label(suit: str) -> str:
    return SUIT_NAMES[suit]


def seat_name(seat: int, perspective: int) -> str:
    return ("自己", "下家", "对家", "上家")[(seat - perspective) % 4]
