import pytest

from state.protocol import (
    Beliefs,
    Facts,
    ObservedValue,
    ObservationStatus,
    S2ProtocolState,
    Statistics,
)


def test_observed_value_distinguishes_unknown_from_empty_observed_value():
    empty_list = ObservedValue.observed([])
    unknown = ObservedValue.unknown()

    assert empty_list.status is ObservationStatus.OBSERVED
    assert empty_list.value == []
    assert empty_list.confidence == 1.0
    assert unknown.status is ObservationStatus.UNKNOWN
    assert unknown.value is None
    assert unknown.confidence == 0.0
    assert empty_list.to_dict() != unknown.to_dict()


def test_observed_value_validates_status_confidence_contracts():
    with pytest.raises(ValueError, match="observed values must have confidence 1.0"):
        ObservedValue(value=1, status=ObservationStatus.OBSERVED, confidence=0.9)

    with pytest.raises(ValueError, match="unknown values must not carry a value"):
        ObservedValue(value=[], status=ObservationStatus.UNKNOWN, confidence=0.0)

    with pytest.raises(ValueError, match="confidence must be in"):
        ObservedValue.estimated("1W", confidence=1.5)


def test_protocol_state_round_trips_through_json_compatible_dict():
    state = S2ProtocolState(
        perspective_player=2,
        phase=ObservedValue.observed("play"),
        current_player=ObservedValue.observed(2),
        current_player_relative=ObservedValue.observed(0),
        facts=Facts(
            players=ObservedValue.observed([
                {"player_id": 2, "relative_position": 0, "concealed_hand": ObservedValue.observed(["1W", "2W"]).to_dict()},
                {"player_id": 3, "relative_position": 1, "concealed_hand": ObservedValue.unknown().to_dict()},
            ]),
            seen_counts=ObservedValue.observed([1] + [0] * 26),
            exchange_tracking=ObservedValue.unknown(),
        ),
        statistics=Statistics(
            remaining_tile_counts=ObservedValue.observed([3] + [4] * 26),
            unknown_pool_breakdown=ObservedValue.observed({"wall": 55, "opponents": {"1": 13, "2": 13, "3": 13}}),
        ),
        beliefs=Beliefs(source=ObservedValue.observed("prior")),
        legal_actions=ObservedValue.observed([]),
        observation_start=ObservedValue.observed(0),
        rule_config=ObservedValue.observed({"base_score": 1, "max_fan": 4, "self_draw_mode": "add_di"}),
    )

    restored = S2ProtocolState.from_dict(state.to_dict())

    assert restored == state
    assert restored.version == "s2.v4"
    assert restored.facts.exchange_tracking.status is ObservationStatus.UNKNOWN
