from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

from engine.gang import GangKind
from engine.hand import Hand
from engine.tiles import SUITS, Suit, Tile



class ActionKind(str, Enum):
    SWAP_THREE = "swap_three"
    DECLARE_VOID = "declare_void"
    DRAW = "draw"
    DISCARD = "discard"
    PONG = "pong"
    KONG = "kong"
    WIN = "win"
    SELF_WIN = "self_win"
    ROB_KONG_WIN = "rob_kong_win"
    PASS = "pass"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    tiles: tuple[Tile, ...] = field(default_factory=tuple)
    tile: Tile | None = None
    suit: Suit | None = None
    kong_kind: GangKind | None = None



def swap_direction_from_dice_sum(total: int) -> int:
    if total in {2, 6, 10}:
        return 1
    if total in {3, 5, 7, 9, 11}:
        return 2
    if total in {4, 8, 12}:
        return -1
    raise ValueError("dice sum must be in 2..12")


def legal_swap_actions(hand: Hand) -> list[Action]:
    actions: list[Action] = []
    for suit in SUITS:
        suited_tiles = [tile for tile in hand.tiles() if tile.suit is suit]
        unique_combos = set(combinations(suited_tiles, 3))
        for combo in sorted(unique_combos):
            actions.append(Action(ActionKind.SWAP_THREE, tiles=combo))
    return actions


def legal_declare_void_actions() -> list[Action]:
    return [Action(ActionKind.DECLARE_VOID, suit=suit) for suit in SUITS]
