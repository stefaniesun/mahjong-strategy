import pytest

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.state import GameState
from engine.tiles import parse_tile
from state.adapters.from_engine import from_engine
from state.protocol import ObservedValue
from state.tile_counting import (
    compute_seen_counts,
    compute_tile_statistics,
    hypergeometric_location_prior,
    reconcile_soft_counts,
)


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def test_compute_seen_counts_uses_only_visible_tiles_and_revealed_win_hands():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W"]),
            Hand.from_strings(["3W", "3W"]),
            Hand.from_strings(["6W", "6W"]),
            Hand.from_strings(["7W", "8W"]),
        ],
        wall=_tiles(["9W", "9T", "9B"]),
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.rivers = [[parse_tile("4W")], [], [], []]
    state.hands[1].add_meld(Meld(MeldKind.PONG, tuple(_tiles(["5W", "5W", "5W"])), exposed=True, from_player=0))
    state.won[2] = True

    protocol_state = from_engine(state, player_id=0)
    counts = compute_seen_counts(protocol_state)

    assert counts[parse_tile("1W").index] == 1
    assert counts[parse_tile("2W").index] == 1
    assert counts[parse_tile("3W").index] == 0
    assert counts[parse_tile("4W").index] == 1
    assert counts[parse_tile("5W").index] == 3
    assert counts[parse_tile("6W").index] == 2
    assert counts[parse_tile("7W").index] == 0


def test_compute_tile_statistics_returns_remaining_counts_and_unknown_pool_breakdown():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "1W"]),
            Hand.from_strings(["2W", "2W", "2W"]),
            Hand.from_strings(["3W", "3W", "3W", "3W"]),
            Hand.from_strings(["4W", "4W", "4W", "5W", "5W"]),

        ],
        wall=_tiles(["9W", "9T"]),
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.won[1] = True

    protocol_state = from_engine(state, player_id=0)
    statistics = compute_tile_statistics(protocol_state)

    assert statistics.remaining_tile_counts.status == protocol_state.facts.seen_counts.status
    assert statistics.remaining_tile_counts.value[parse_tile("1W").index] == 2
    assert statistics.remaining_tile_counts.value[parse_tile("2W").index] == 1
    assert statistics.unknown_pool_breakdown.value == {"wall": 2, "opponents": {"2": 4, "3": 5}}


def test_reconcile_soft_counts_clamps_counts_and_preserves_global_conservation():
    result = reconcile_soft_counts([4.6, 2.0] + [0.0] * 25, unknown_pool_total=102.0)

    assert result.contradiction is True
    assert result.seen_counts[0] == 4.0
    assert result.remaining_counts[0] == 0.0
    assert sum(result.seen_counts) + result.unknown_pool_total == pytest.approx(108.0)


def test_hypergeometric_prior_degrades_uniformly_when_wall_count_unknown():
    prior = hypergeometric_location_prior(
        remaining_counts=[2.0] + [0.0] * 26,
        unknown_pool_breakdown=ObservedValue.unknown(),
    )

    assert prior["1W"] == {"wall": pytest.approx(0.25), "1": pytest.approx(0.25), "2": pytest.approx(0.25), "3": pytest.approx(0.25)}


def test_hypergeometric_prior_splits_by_unknown_pool_sizes():
    prior = hypergeometric_location_prior(
        remaining_counts=[3.0] + [0.0] * 26,
        unknown_pool_breakdown=ObservedValue.observed({"wall": 20, "opponents": {"1": 10, "2": 5}}),
    )

    assert prior["1W"] == {"wall": pytest.approx(20 / 35), "1": pytest.approx(10 / 35), "2": pytest.approx(5 / 35)}
