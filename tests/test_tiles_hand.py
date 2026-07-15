import pytest

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, Tile, full_wall, parse_tile, tile_to_str


def test_parse_and_format_tile():
    tile = parse_tile("3T")

    assert tile == Tile(Suit.TIAO, 3)
    assert tile_to_str(tile) == "3T"
    assert tile.index == 11


@pytest.mark.parametrize("text", ["0T", "10W", "3Z", "TT", ""])
def test_reject_invalid_tile_text(text):
    with pytest.raises(ValueError):
        parse_tile(text)


def test_full_wall_has_108_tiles_and_four_copies_each():
    wall = full_wall()

    assert len(wall) == 108
    assert wall.count(parse_tile("1W")) == 4
    assert wall.count(parse_tile("9B")) == 4


def test_hand_add_remove_and_counts():
    hand = Hand.from_strings(["1W", "1W", "2W"])

    assert hand.count(parse_tile("1W")) == 2
    hand.remove(parse_tile("1W"))
    assert hand.count(parse_tile("1W")) == 1
    hand.add(parse_tile("3W"))
    assert hand.count(parse_tile("3W")) == 1
    assert hand.size == 3


def test_remove_missing_tile_raises():
    hand = Hand.from_strings(["1W"])

    with pytest.raises(ValueError, match="not in hand"):
        hand.remove(parse_tile("2W"))


def test_meld_validation():
    pong = Meld(MeldKind.PONG, (parse_tile("5B"),) * 3, exposed=True, from_player=2)

    assert pong.kind is MeldKind.PONG
    assert pong.tiles == (parse_tile("5B"), parse_tile("5B"), parse_tile("5B"))
