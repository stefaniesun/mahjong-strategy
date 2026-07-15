from __future__ import annotations

from engine.hand import Hand
from engine.meld import Meld
from engine.tiles import Tile, tile_to_str
from state.protocol import ObservedValue


PLAYER_COUNT = 4


def relative_position(player_id: int, perspective_player: int) -> int:
    return (player_id - perspective_player) % PLAYER_COUNT


def absolute_player(relative: int, perspective_player: int) -> int:
    return (perspective_player + relative) % PLAYER_COUNT


def players_in_relative_order(perspective_player: int) -> list[int]:
    return [absolute_player(relative, perspective_player) for relative in range(PLAYER_COUNT)]


def tiles_to_strings(tiles: list[Tile] | tuple[Tile, ...]) -> list[str]:
    return [tile_to_str(tile) for tile in tiles]


def hand_tiles_to_strings(hand: Hand) -> list[str]:
    return tiles_to_strings(hand.tiles())


def meld_to_dict(meld: Meld) -> dict:
    return {
        "kind": meld.kind.value,
        "tiles": tiles_to_strings(meld.tiles),
        "exposed": meld.exposed,
        "from_player": meld.from_player,
    }


def visible_concealed_hand(hand: Hand, *, is_self: bool, has_won: bool) -> ObservedValue[list[str]]:
    if is_self or has_won:
        return ObservedValue.observed(hand_tiles_to_strings(hand))
    return ObservedValue.unknown()
