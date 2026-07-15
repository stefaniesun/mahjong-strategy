from __future__ import annotations

from engine.hand import Hand
from engine.tiles import Suit
from state.hand_analysis import best_discards as _best_discards
from state.hand_analysis import shanten as _shanten
from state.hand_analysis import useful_tiles as _useful_tiles


def shanten(hand: Hand, void_suit: Suit | None = None) -> int:
    return _shanten(hand, void_suit=void_suit)


def best_discards(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    return _best_discards(hand, void_suit=void_suit)


def useful_tiles(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    return _useful_tiles(hand, void_suit=void_suit)
