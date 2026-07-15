from engine.hand import Hand
from engine.tiles import Suit
from policies import shanten as s3_shanten
from state.hand_analysis import best_discards, shanten, useful_tiles



def hand(tiles: list[str]) -> Hand:
    return Hand.from_strings(tiles)


def test_complete_standard_hand_is_minus_one_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T", "5T"])) == -1


def test_ready_standard_hand_is_zero_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T"])) == 0


def test_one_away_standard_hand_is_one_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])) == 1


def test_complete_seven_pairs_is_minus_one_shanten():
    assert shanten(hand(["1W", "1W", "2W", "2W", "3W", "3W", "4T", "4T", "5T", "5T", "6B", "6B", "9B", "9B"])) == -1


def test_ready_seven_pairs_is_zero_shanten():
    assert shanten(hand(["1W", "1W", "2W", "2W", "3W", "3W", "4T", "4T", "5T", "5T", "6B", "6B", "9B"])) == 0


def test_void_suit_tiles_do_not_count_as_progress():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T"])
    assert shanten(candidate) == 0
    assert shanten(candidate, void_suit=Suit.WAN) > 0


def test_best_discards_keep_lowest_shanten_after_discard():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    discards = best_discards(candidate)
    assert "5T" in discards
    assert "9B" not in discards


def test_best_discards_prioritize_void_suit_when_present():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    discards = best_discards(candidate, void_suit=Suit.WAN)
    assert set(discards).issubset({"1W", "2W", "3W", "7W", "8W"})


def test_useful_tiles_are_tiles_that_reduce_shanten():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    useful = useful_tiles(candidate)
    assert "5T" in useful
    assert "9W" in useful


def test_useful_tiles_exclude_void_suit():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    useful = useful_tiles(candidate, void_suit=Suit.WAN)
    assert all(not tile.endswith("W") for tile in useful)


def test_s3_shanten_adapter_matches_s2_source():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    assert s3_shanten.shanten(candidate) == shanten(candidate)
    assert s3_shanten.best_discards(candidate) == best_discards(candidate)
    assert s3_shanten.useful_tiles(candidate) == useful_tiles(candidate)


def test_s3_shanten_adapter_forwards_void_suit():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    assert s3_shanten.shanten(candidate, void_suit=Suit.WAN) == shanten(candidate, void_suit=Suit.WAN)
    assert s3_shanten.best_discards(candidate, void_suit=Suit.WAN) == best_discards(candidate, void_suit=Suit.WAN)



