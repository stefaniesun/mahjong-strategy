import pytest

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, parse_tile
from engine.win_check import can_win, is_seven_pairs, is_standard_win


def hand(text: str) -> Hand:
    return Hand.from_strings(text.split())


def test_standard_win_with_sequences_and_pair():
    h = hand("1W 2W 3W 2W 3W 4W 3T 4T 5T 7B 8B 9B 9W 9W")

    assert can_win(h, void_suit=Suit.BING) is False
    assert can_win(h, void_suit=None) is True
    assert is_standard_win(h) is True


def test_seven_pairs_win():
    h = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W")

    assert is_seven_pairs(h) is True
    assert can_win(h) is True


def test_incomplete_hand_cannot_win():
    h = hand("1W 2W 3W 2W 3W 4W 3T 4T 5T 7B 8B 9B 9W 8W")

    assert can_win(h) is False


def test_void_suit_blocks_win():
    h = hand("1W 2W 3W 2W 3W 4W 3T 4T 5T 7B 8B 9B 9W 9W")

    assert can_win(h, void_suit=Suit.WAN) is False


def test_seven_pairs_disallows_melds():
    h = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W")
    h.add_meld(Meld(MeldKind.PONG, (parse_tile("7W"),) * 3, exposed=True))

    assert is_seven_pairs(h) is False
