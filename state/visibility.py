from __future__ import annotations

from engine.hand import Hand
from engine.meld import Meld
from engine.state import GameState
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


def visible_concealed_hand(hand: Hand, *, is_self: bool) -> ObservedValue[list[str]]:
    # 按真实打法,胡牌者手牌扣着不亮,所以别人的暗手牌一律不可见,胡没胡都一样
    if is_self:
        return ObservedValue.observed(hand_tiles_to_strings(hand))
    return ObservedValue.unknown()


def revealed_win_tile(engine_state: GameState, player: int) -> Tile | None:
    """胡牌者亮出、且不在别处可见的那张胡牌张。

    自摸、抢杠胡亮出胡牌张;点炮胡的那张留在放炮者牌河里,已经可见,这里不重复给出。
    """
    record = engine_state.win_records[player]
    if not engine_state.won[player] or record is None or record.kind == "discard":
        return None
    return record.tile


def hidden_hand_tiles(engine_state: GameState, player: int) -> list[Tile]:
    """别家看不见的暗手牌(真实牌面,只供上帝视角标签使用)。胡牌者扣掉亮出的胡牌张。"""
    tiles = list(engine_state.hands[player].tiles())
    revealed = revealed_win_tile(engine_state, player)
    if revealed is not None and engine_state.win_records[player].kind == "self_draw":
        tiles.remove(revealed)  # 自摸的那张还在引擎手牌里;抢杠胡、点炮胡的胡牌张不在
    return tiles
