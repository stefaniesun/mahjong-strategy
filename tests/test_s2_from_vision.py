import pytest

from engine.tiles import parse_tile
from state.adapters.from_vision import VisionEvent, VisionSnapshot, from_vision_events

from state.protocol import ObservationStatus, S2ProtocolState



def test_vision_events_build_protocol_with_confidence_and_observation_start():
    events = [
        VisionEvent(seq=5, timestamp=10.0, event_type="discard", player=1, tile="4W", confidence=0.72),
        VisionEvent(
            seq=6,
            timestamp=11.0,
            event_type="pong",
            player=2,
            tile="4W",
            confidence=0.81,
            alternatives=[("5W", 0.19)],
        ),
        VisionEvent(seq=7, timestamp=12.0, event_type="dingque", player=0, tile="wan", confidence=0.9),
    ]

    state, report = from_vision_events(events, perspective_player=0, current_player=0, wall_count=100)

    assert isinstance(state, S2ProtocolState)
    assert report.contradictions == []
    assert state.observation_start.value == 5
    assert state.facts.event_history.value == [event.to_dict() for event in events]
    assert state.facts.players.value[1]["rivers"].status is ObservationStatus.ESTIMATED
    assert state.facts.players.value[1]["rivers"].confidence == 0.72
    assert state.facts.players.value[1]["rivers"].value == ["4W"]
    assert state.facts.players.value[2]["melds"].value == [{"kind": "pong", "tiles": ["4W", "4W", "4W"], "from_player": None}]
    assert state.facts.players.value[0]["void_suit"].value == "wan"
    assert state.facts.seen_counts.status is ObservationStatus.ESTIMATED
    assert state.facts.seen_counts.value[parse_tile("4W").index] == pytest.approx(0.72 + 0.81 * 3)




def test_vision_snapshot_initializes_midgame_static_observations():
    snapshot = VisionSnapshot(
        seq=20,
        timestamp=30.0,
        rivers={0: [("1W", 0.9)], 3: [("9T", 0.8)]},
        melds={1: [{"kind": "kong", "tiles": ["2B", "2B", "2B", "2B"], "confidence": 0.7}]},
        wall_count=(48, 0.6),
    )

    state, report = from_vision_events([snapshot], perspective_player=0)

    assert report.contradictions == []
    assert state.observation_start.value == 20
    assert state.facts.wall_count.status is ObservationStatus.ESTIMATED
    assert state.facts.wall_count.value == 48
    assert state.facts.players.value[0]["rivers"].value == ["1W"]
    assert state.facts.players.value[1]["melds"].value[0]["tiles"] == ["2B", "2B", "2B", "2B"]
    assert state.facts.players.value[3]["rivers"].confidence == 0.8


def test_vision_reconciliation_reports_and_rolls_back_fifth_copy():
    events = [
        VisionEvent(seq=0, timestamp=0.0, event_type="discard", player=0, tile="1W", confidence=0.95),
        VisionEvent(seq=1, timestamp=1.0, event_type="discard", player=1, tile="1W", confidence=0.95),
        VisionEvent(seq=2, timestamp=2.0, event_type="discard", player=2, tile="1W", confidence=0.95),
        VisionEvent(seq=3, timestamp=3.0, event_type="discard", player=3, tile="1W", confidence=0.95),
        VisionEvent(seq=4, timestamp=4.0, event_type="discard", player=0, tile="1W", confidence=0.95),
    ]

    state, report = from_vision_events(events, perspective_player=0)

    assert len(report.contradictions) == 1
    assert report.contradictions[0]["seq"] == 4
    assert report.contradictions[0]["reason"] == "tile_count_exceeds_four"
    assert state.facts.players.value[0]["rivers"].value == ["1W"]
    assert state.facts.event_history.value == [event.to_dict() for event in events[:4]]
    assert state.facts.seen_counts.value[parse_tile("1W").index] == pytest.approx(0.95 * 4)


