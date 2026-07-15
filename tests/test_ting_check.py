from engine.hand import Hand
from engine.tiles import Suit, parse_tile

from engine.ting_check import max_ting_fan, ting_tiles


def hand(text: str) -> Hand:
    return Hand.from_strings(text.split())


def test_ting_tiles_for_edge_wait():
    h = hand("1W 2W 3W 2W 3W 4W 3T 4T 5T 7B 8B 9B 9W")

    assert ting_tiles(h) == {parse_tile("9W")}


def test_ting_tiles_for_seven_pairs():
    h = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W")

    assert ting_tiles(h) == {parse_tile("9W")}


def test_ting_tiles_excludes_winning_tiles_from_void_suit():
    h = hand("1W 1W 2W 2W 3W 3W 4B 4B 5B 5B 6B 6B 9W")

    assert ting_tiles(h, Suit.WAN) == set()
    assert ting_tiles(h, Suit.TIAO) == {parse_tile("9W")}



def test_max_ting_fan_uses_best_completion():

    h = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W")

    assert max_ting_fan(h) == 2
