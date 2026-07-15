from engine.hand import Hand
from engine.state import GameState
from engine.tiles import Suit, parse_tile
from state.action_features import compute_candidate_action_features
from state.adapters.from_engine import from_engine
from state.hand_analysis import analyze_own_hand
from state.memory import ObservationMemory
from state.observation_degradation import MidGameSnapshot
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _base_state() -> S2ProtocolState:
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "1W", "2W", "3W", "4W", "5T", "6T", "7T", "2B", "3B", "4B", "9W", "9W", "9W"]),
            Hand.from_strings(["3W", "3W", "4T", "5T"]),
            Hand.from_strings(["6W", "6W", "7B", "8B"]),
            Hand.from_strings(["1T", "2T", "3T", "4B", "5B"]),
        ],
        wall=_tiles(["9T", "9B", "8W"]),
        dealer=0,
        current_player=0,
        phase="play",
        void_suits=[Suit.BING, Suit.WAN, Suit.TIAO, Suit.BING],
    )
    state.rivers = [[parse_tile("5W")], [parse_tile("2W")], [], []]
    state.event_log = [
        {"seq": 0, "type": "discard", "player": 2, "tile": "8T"},
        {"seq": 3, "type": "discard", "player": 1, "tile": "2W"},
    ]
    return from_engine(state, player_id=0)


def test_analyze_own_hand_reports_basic_shape_and_unknown_when_hand_hidden():
    state = _base_state()

    analysis = analyze_own_hand(state)

    assert analysis.status is ObservationStatus.OBSERVED
    assert analysis.value["tile_count"] == 14
    assert analysis.value["suit_counts"] == {"W": 8, "T": 3, "B": 3}

    assert analysis.value["void_suit"] == "B"
    assert analysis.value["void_tile_count"] == 3

    assert analysis.value["pairs"] >= 1
    assert analysis.value["triplets"] >= 1
    assert analysis.value["ting_tiles"] == []

    hidden_self = dict(state.facts.players.value[0])
    hidden_self["concealed_hand"] = ObservedValue.unknown()
    hidden_state = S2ProtocolState(
        perspective_player=state.perspective_player,
        phase=state.phase,
        current_player=state.current_player,
        current_player_relative=state.current_player_relative,
        facts=type(state.facts)(**{**state.facts.__dict__, "players": ObservedValue.observed([hidden_self] + state.facts.players.value[1:])}),
        statistics=state.statistics,
        beliefs=state.beliefs,
        legal_actions=state.legal_actions,
        observation_start=state.observation_start,
        rule_config=state.rule_config,
    )

    assert analyze_own_hand(hidden_state).status is ObservationStatus.UNKNOWN


def test_candidate_action_features_are_unknown_aware_and_do_not_mutate_legal_actions():
    state = _base_state()
    original_actions = [dict(action) for action in state.legal_actions.value]

    features = compute_candidate_action_features(state)

    discard_1w = next(item for item in features.value if item["kind"] == "discard" and item["tile"] == "1W")
    assert features.status is ObservationStatus.OBSERVED
    assert discard_1w["tile_remaining_count"] == state.statistics.remaining_tile_counts.value[parse_tile("1W").index]
    assert discard_1w["is_void_suit"] is False
    assert discard_1w["keeps_ting"] is False
    assert state.legal_actions.value == original_actions

    degraded = MidGameSnapshot(k=2).apply(state)
    degraded_features = compute_candidate_action_features(degraded)
    degraded_discard = next(item for item in degraded_features.value if item["kind"] == "discard" and item["tile"] == "1W")

    assert degraded_features.status is ObservationStatus.ESTIMATED
    assert degraded_discard["tile_remaining_count"] == degraded.statistics.remaining_tile_counts.value[parse_tile("1W").index]
    assert degraded_discard["estimated_inputs"] == ["observation_start"]



def test_observation_memory_starts_at_midgame_without_inventing_prior_events_and_tracks_exchange():
    state = MidGameSnapshot(k=2).apply(_base_state())

    memory = ObservationMemory.from_state(state)

    assert memory.observation_start == 2
    assert memory.events == [{"seq": 3, "type": "discard", "player": 1, "tile": "2W"}]
    assert memory.exchange_tracking.status is ObservationStatus.UNKNOWN
    assert memory.to_observed_value().status is ObservationStatus.OBSERVED

    grown = memory.update({"seq": 4, "type": "pong", "player": 0, "tile": "2W"})
    assert memory.events == [{"seq": 3, "type": "discard", "player": 1, "tile": "2W"}]
    assert grown.events[-1] == {"seq": 4, "type": "pong", "player": 0, "tile": "2W"}


def test_observation_memory_preserves_observed_exchange_tracking():
    state = _base_state()

    memory = ObservationMemory.from_state(state)

    assert memory.observation_start == 0
    assert memory.exchange_tracking.status is ObservationStatus.OBSERVED
    assert memory.exchange_tracking.value["swap_direction"] == 1
