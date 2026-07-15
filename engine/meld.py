from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.tiles import Tile


class MeldKind(str, Enum):
    CHOW = "chow"
    PONG = "pong"
    KONG = "kong"
    PAIR = "pair"


@dataclass(frozen=True)
class Meld:
    kind: MeldKind
    tiles: tuple[Tile, ...]
    exposed: bool = False
    from_player: int | None = None

    def __post_init__(self) -> None:
        expected_sizes = {
            MeldKind.CHOW: 3,
            MeldKind.PONG: 3,
            MeldKind.KONG: 4,
            MeldKind.PAIR: 2,
        }
        if len(self.tiles) != expected_sizes[self.kind]:
            raise ValueError(f"{self.kind.value} requires {expected_sizes[self.kind]} tiles")
