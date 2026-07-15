from __future__ import annotations

from engine.config import RuleConfig
from engine.fan_calc import WinContext, calculate_fan
from engine.hand import Hand
from engine.tiles import Suit, Tile

from engine.win_check import can_win


def ting_tiles(hand: Hand, void_suit: Suit | None = None) -> set[Tile]:

    if hand.size % 3 != 1:
        return set()
    result: set[Tile] = set()
    for index in range(27):
        tile = Tile.from_index(index)
        if hand.count(tile) >= 4:
            continue
        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.add(tile)
        if can_win(trial, void_suit):

            result.add(tile)
    return result


def max_ting_fan(
    hand: Hand,
    config: RuleConfig | None = None,
    void_suit: Suit | None = None,
) -> int:
    max_fan = -1
    for tile in ting_tiles(hand, void_suit):

        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.add(tile)
        result = calculate_fan(trial, WinContext(), config)
        max_fan = max(max_fan, result.fan)
    if max_fan < 0:
        raise ValueError("hand is not ting")
    return max_fan
