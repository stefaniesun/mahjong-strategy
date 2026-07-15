from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.state import GameState
from engine.tiles import parse_tile
from state.adapters.from_engine import from_engine
from state.observation_degradation import (
    DegradationPipeline,
    MaskExchange,
    MaskField,
    MidGameSnapshot,
    VisionNoise,
)
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState
from state.tile_counting import compute_seen_counts, compute_tile_statistics


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _sample_state():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W", "3W"]),
            Hand.from_strings(["4W", "5W", "6W"]),
            Hand.from_strings(["7W", "8W", "9W"]),
            Hand.from_strings(["1T", "2T", "3T"]),
        ],
        wall=_tiles(["9B", "8B", "7B", "6B"]),
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.rivers = [[parse_tile("4W"), parse_tile("5W")], [parse_tile("6W")], [], []]
    state.hands[1].add_meld(Meld(MeldKind.PONG, tuple(_tiles(["7T", "7T", "7T"])), exposed=True, from_player=0))
    state.event_log = [
        {"seq": 0, "type": "discard", "tile": "1W"},
        {"seq": 1, "type": "discard", "tile": "2W"},
        {"seq": 2, "type": "discard", "tile": "3W"},
    ]
    return from_engine(state, player_id=0)


def test_mid_game_snapshot_truncates_history_but_keeps_static_visible_tiles_countable():
    protocol_state = _sample_state()
    degraded = MidGameSnapshot(k=2).apply(protocol_state)

    assert degraded is not protocol_state
    assert degraded.observation_start.value == 2
    assert degraded.facts.event_history.status is ObservationStatus.OBSERVED
    assert degraded.facts.event_history.value == [{"seq": 2, "type": "discard", "tile": "3W"}]
    assert degraded.facts.exchange_tracking.status is ObservationStatus.UNKNOWN
    assert degraded.facts.seen_counts.value[parse_tile("4W").index] == 1
    assert degraded.facts.seen_counts.value[parse_tile("7T").index] == 3
    assert compute_tile_statistics(degraded).remaining_tile_counts.status is ObservationStatus.OBSERVED


def test_mask_exchange_is_seeded_and_masks_whole_exchange_tracking():
    protocol_state = _sample_state()

    assert MaskExchange(p=1.0, seed=7).apply(protocol_state).facts.exchange_tracking.status is ObservationStatus.UNKNOWN
    assert MaskExchange(p=0.0, seed=7).apply(protocol_state).facts.exchange_tracking.status is ObservationStatus.OBSERVED


def test_vision_noise_is_seeded_estimates_and_recomputes_soft_counts():
    protocol_state = _sample_state()
    noise = VisionNoise(confusion_matrix={"4W": {"5W": 1.0}}, miss_rate=0.0, seed=11)

    degraded = noise.apply(protocol_state)
    own_river = degraded.facts.players.value[0]["rivers"]

    assert own_river.status is ObservationStatus.ESTIMATED
    assert own_river.confidence < 1.0
    assert own_river.value[0] == "5W"
    assert degraded.facts.seen_counts.status is ObservationStatus.ESTIMATED
    assert degraded.facts.seen_counts.value == compute_seen_counts(degraded)


def test_vision_noise_can_drop_observations_without_touching_hidden_hands():
    protocol_state = _sample_state()
    degraded = VisionNoise(miss_rate=1.0, seed=3).apply(protocol_state)

    assert degraded.facts.players.value[0]["rivers"].value == []
    assert degraded.facts.players.value[1]["melds"].status is ObservationStatus.UNKNOWN
    assert degraded.facts.players.value[1]["concealed_hand"].status is ObservationStatus.UNKNOWN


def test_vision_noise_marks_partial_meld_observation_unknown_so_legal_actions_remain_safe():
    protocol_state = _sample_state()
    pipeline = DegradationPipeline([VisionNoise(miss_rate=0.5, seed=0)])

    degraded = pipeline.apply(protocol_state)

    assert degraded.facts.players.value[1]["melds"].status is ObservationStatus.UNKNOWN
    assert degraded.legal_actions.status is ObservationStatus.OBSERVED






def test_mask_field_masks_supported_fields_and_pipeline_is_deterministic_round_trip_safe():
    protocol_state = _sample_state()
    pipeline = DegradationPipeline(
        [MaskField("wall_count", p=1.0, seed=13), MaskField("dealer", p=1.0, seed=13)],
        recompute_statistics=True,
    )

    degraded = pipeline.apply(protocol_state)
    restored = S2ProtocolState.from_dict(degraded.to_dict())

    assert degraded.facts.wall_count.status is ObservationStatus.UNKNOWN
    assert degraded.facts.dealer.status is ObservationStatus.UNKNOWN
    assert degraded.statistics.unknown_pool_breakdown.value["wall"] is None
    assert restored == degraded


def test_degradation_pipeline_produces_same_output_for_same_seeded_operators():
    protocol_state = _sample_state()
    left = DegradationPipeline([MaskExchange(p=0.5, seed=42), VisionNoise(miss_rate=0.2, seed=42)]).apply(protocol_state)
    right = DegradationPipeline([MaskExchange(p=0.5, seed=42), VisionNoise(miss_rate=0.2, seed=42)]).apply(protocol_state)

    assert left.to_dict() == right.to_dict()
