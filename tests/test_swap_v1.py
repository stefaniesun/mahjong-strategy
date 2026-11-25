"""测试 assistant.swap_v1 的向听一致性、速度与典型牌型决策。"""
from __future__ import annotations

import random

from assistant.swap_v1 import (
    choose_swap_v1,
    choose_void_v1,
    fast_live_useful_two_suits,
    fast_shanten_counts,
    split_suits,
)
from engine.hand import Hand
from engine.tiles import SUITS, Suit, full_wall, parse_tile, tile_to_str
from state.hand_analysis import shanten, useful_tiles


def _live_ref(hand: Hand, void_suit: Suit) -> int:
    return sum(max(0, 4 - hand.count(parse_tile(t))) for t in useful_tiles(hand, void_suit=void_suit))


def test_fast_shanten_and_live_match_engine_exactly():
    rng = random.Random(20261125)
    base = full_wall()
    for _ in range(300):
        pool = list(base)
        rng.shuffle(pool)
        size = rng.choice((10, 11, 13, 14))
        hand = Hand()
        for t in pool[:size]:
            hand.add(t)
        s3 = split_suits(hand.counts)
        assert fast_shanten_counts(s3, None) == shanten(hand, None)
        for idx, suit in enumerate(SUITS):
            s_fast = fast_shanten_counts(s3, idx)
            s_ref = shanten(hand, suit)
            assert s_fast == s_ref
            other = ((1, 2), (0, 2), (0, 1))[idx]
            live_fast = fast_live_useful_two_suits(s3[other[0]], s3[other[1]], s_fast)
            assert live_fast == _live_ref(hand, suit)


def test_v1_keeps_complete_chow_in_346():
    hand = Hand.from_strings("4W 5W 6W 1T 4T 7T 9T 2B 3B 4B 6B 7B 8B".split())
    chosen = [tile_to_str(t) for t in choose_swap_v1(hand)]
    assert chosen == ["1T", "4T", "9T"]
    assert choose_void_v1(hand) == Suit("T")


def test_v1_prefers_clean_trash_and_keeps_pair_in_445():
    hand = Hand.from_strings("2W 5W 5W 8W 1T 4T 6T 9T 3B 4B 6B 7B 7B".split())
    chosen = [tile_to_str(t) for t in choose_swap_v1(hand)]
    assert chosen == ["1T", "4T", "9T"]


def test_v1_discovers_double_short_suit_in_256():
    hand = Hand.from_strings("1W 9W 2T 3T 5T 7T 8T 1B 3B 4B 5B 8B 9B".split())
    chosen = choose_swap_v1(hand)
    assert all(t.suit == Suit("T") for t in chosen)
