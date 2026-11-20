from engine.tiles import Suit

from tools.void_distribution_counterfactual import (
    VoidDistribution,
    classify_void_distribution,
)


def test_three_other_players_same_void_is_opponent_single_lane():
    result = classify_void_distribution(
        target=0,
        voids=(Suit.WAN, Suit.TIAO, Suit.TIAO, Suit.TIAO),
    )

    assert result.distribution == {"W": 1, "T": 3, "B": 0}
    assert result.single_lane is True
    assert result.opponent_single_lane is True
    assert result.single_lane_suit is Suit.TIAO
    assert result.unique_non_void_player == 0


def test_target_sharing_a_void_is_not_opponent_single_lane():
    result = classify_void_distribution(
        target=0,
        voids=(Suit.WAN, Suit.WAN, Suit.TIAO, Suit.TIAO),
    )

    assert result.single_lane is False
    assert result.opponent_single_lane is False


def test_void_distribution_preserves_each_players_choices():
    result = classify_void_distribution(
        target=2,
        voids=(Suit.BING, Suit.BING, Suit.WAN, Suit.TIAO),
    )

    assert isinstance(result, VoidDistribution)
    assert result.voids == (Suit.BING, Suit.BING, Suit.WAN, Suit.TIAO)
    assert result.distribution == {"W": 1, "T": 1, "B": 2}
