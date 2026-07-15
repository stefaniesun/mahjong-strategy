from __future__ import annotations

from dataclasses import dataclass, field

from engine.meld import Meld
from engine.tiles import Tile, parse_tile


@dataclass
class Hand:
    counts: list[int] = field(default_factory=lambda: [0] * 27)
    melds: list[Meld] = field(default_factory=list)

    @classmethod
    def from_tiles(cls, tiles: list[Tile] | tuple[Tile, ...]) -> "Hand":
        hand = cls()
        for tile in tiles:
            hand.add(tile)
        return hand

    @classmethod
    def from_strings(cls, tiles: list[str] | tuple[str, ...]) -> "Hand":
        return cls.from_tiles([parse_tile(text) for text in tiles])

    @property
    def size(self) -> int:
        return sum(self.counts)

    def count(self, tile: Tile) -> int:
        return self.counts[tile.index]

    def add(self, tile: Tile) -> None:
        if self.counts[tile.index] >= 4:
            raise ValueError(f"cannot hold more than four copies of {tile}")
        self.counts[tile.index] += 1

    def remove(self, tile: Tile) -> None:
        if self.counts[tile.index] <= 0:
            raise ValueError(f"tile {tile} not in hand")
        self.counts[tile.index] -= 1

    def tiles(self) -> list[Tile]:
        result: list[Tile] = []
        for index, count in enumerate(self.counts):
            result.extend([Tile.from_index(index)] * count)
        return result

    def add_meld(self, meld: Meld) -> None:
        self.melds.append(meld)
