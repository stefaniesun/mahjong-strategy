from engine.hand import Hand

from engine.state import GameState
from engine.tiles import Suit, full_wall, parse_tile

from state.action_space import legal_mask
from state.legality import legal_actions

from state.adapters.from_engine import from_engine
from state.adapters.from_vision import VisionEvent, VisionSnapshot, from_vision_events
from state.encoder import encode_state
from state.observation_degradation import DegradationPipeline, MaskField, MidGameSnapshot, VisionNoise
from state.protocol import Facts, ObservationStatus, ObservedValue, S2ProtocolState

from state.tile_belief import generate_belief_labels, with_prior_beliefs


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _engine_state() -> GameState:
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "1W", "2W", "3W", "9W", "5T", "6T", "7T", "2B", "3B", "4B", "9W", "9W", "9W"]),

            Hand.from_strings(["3W", "3W", "4T", "5T"]),
            Hand.from_strings(["6W", "6W", "7B", "8B"]),
            Hand.from_strings(["1T", "2T", "3T", "4B", "5B"]),
        ],
        wall=_tiles(["9T", "9B", "8W", "8T", "7W"]),
        dealer=0,
        current_player=0,
        phase="play",
        void_suits=[Suit.BING, Suit.WAN, Suit.TIAO, Suit.BING],
    )
    state.rivers = [[parse_tile("5W")], [parse_tile("2W")], [], []]
    state.event_log = [
        {"seq": 0, "type": "discard", "player": 2, "tile": "8T"},
        {"seq": 1, "type": "discard", "player": 1, "tile": "2W"},
    ]
    for tile in full_wall():
        physical_count = sum(hand.count(tile) for hand in state.hands)
        physical_count += sum(river.count(tile) for river in state.rivers)
        physical_count += sum(wall_tile == tile for wall_tile in state.wall)
        if physical_count < 4:
            state.rivers[0].extend([tile] * (4 - physical_count))
    return state



def _assert_protocol_round_trip_and_encoding(state: S2ProtocolState) -> None:
    restored = S2ProtocolState.from_dict(state.to_dict())
    encoded = encode_state(restored)

    assert restored == state
    assert restored.version == "s2.v4"
    assert encoded.version == "s2.v4.encoder.v2"

    assert encoded.size == len(encoded.values)


def test_engine_degraded_label_pipeline_preserves_visibility_and_fixed_encoding():
    engine_state = _engine_state()
    perfect = from_engine(engine_state, player_id=0)
    degraded = DegradationPipeline(
        [
            MidGameSnapshot(k=1),
            VisionNoise(confusion_matrix={"5W": {"6W": 1.0}}, miss_rate=0.0, seed=17),
            MaskField("wall_count", p=1.0, seed=17),
        ]
    ).apply(perfect)
    degraded = with_prior_beliefs(degraded)
    labels = generate_belief_labels(engine_state, degraded)

    _assert_protocol_round_trip_and_encoding(perfect)
    _assert_protocol_round_trip_and_encoding(degraded)
    assert len(encode_state(perfect).values) == len(encode_state(degraded).values)
    assert degraded.observation_start.value == 1
    assert degraded.facts.wall_count.status is ObservationStatus.UNKNOWN
    assert degraded.facts.seen_counts.status is ObservationStatus.ESTIMATED
    assert any(action.get("conditionally_legal") for action in degraded.legal_actions.value)
    assert len(legal_mask(degraded)) > 0
    tile_locations = labels["tile_locations"]
    assert tile_locations["counts"][parse_tile("9T").index] == [1, 0, 0, 0]
    assert tile_locations["counts"][parse_tile("3W").index] == [0, 2, 0, 0]

    assert degraded.facts.players.value[1]["concealed_hand"].status is ObservationStatus.UNKNOWN
    assert degraded.facts.players.value[2]["concealed_hand"].status is ObservationStatus.UNKNOWN
    assert degraded.facts.players.value[3]["concealed_hand"].status is ObservationStatus.UNKNOWN


def test_same_visible_information_has_same_encoding_when_hidden_opponent_hands_differ():
    first_engine = _engine_state()
    second_engine = _engine_state()
    second_engine.hands[1] = Hand.from_strings(["7W", "7W", "8T", "9T"])
    second_engine.hands[2] = Hand.from_strings(["1B", "1B", "2T", "2T"])
    second_engine.hands[3] = Hand.from_strings(["3B", "3B", "4T", "4T", "5T"])

    first = from_engine(first_engine, player_id=0)
    second = from_engine(second_engine, player_id=0)

    assert encode_state(first).values == encode_state(second).values


def test_simulated_vision_events_enter_protocol_statistics_legality_and_encoder():
    state, report = from_vision_events(
        [
            VisionSnapshot(
                seq=40,
                timestamp=10.0,
                rivers={0: [("1W", 0.91)], 1: [("2W", 0.83)]},
                melds={2: [{"kind": "pong", "tiles": ["3W", "3W", "3W"], "confidence": 0.76}]},
                void_suits={0: ("B", 0.9), 1: ("W", 0.8)},
                wall_count=(55, 0.7),
            ),
            VisionEvent(seq=41, timestamp=11.0, event_type="discard", player=3, tile="4W", confidence=0.66),
            VisionEvent(seq=42, timestamp=12.0, event_type="dingque", player=2, tile="T", confidence=0.72),
        ],
        perspective_player=0,
        current_player=0,
    )

    _assert_protocol_round_trip_and_encoding(state)
    assert report.contradictions == []
    assert state.observation_start.value == 40
    assert state.facts.wall_count.status is ObservationStatus.ESTIMATED
    assert state.facts.seen_counts.status is ObservationStatus.ESTIMATED
    assert state.statistics.remaining_tile_counts.status is ObservationStatus.ESTIMATED
    assert state.beliefs.source.value == "prior"
    assert state.legal_actions.value == []
    assert state.facts.players.value[1]["concealed_hand"].status is ObservationStatus.UNKNOWN


def test_legal_mask_is_conditional_when_observation_missing_but_shape_stable():
    state = from_engine(_engine_state(), player_id=0)
    unknown_facts = Facts(
        players=state.facts.players,
        dealer=state.facts.dealer,
        dealer_relative_position=state.facts.dealer_relative_position,
        is_dealer=state.facts.is_dealer,
        wall_count=ObservedValue.unknown(),
        is_last_tile=state.facts.is_last_tile,
        pending_discard=state.facts.pending_discard,
        pending_rob_kong=state.facts.pending_rob_kong,
        exchange_tracking=state.facts.exchange_tracking,
        event_history=state.facts.event_history,
        revealed_win_hands=state.facts.revealed_win_hands,
        seen_counts=state.facts.seen_counts,
    )
    unknown_wall = S2ProtocolState(
        perspective_player=state.perspective_player,
        phase=state.phase,
        current_player=state.current_player,
        current_player_relative=state.current_player_relative,
        facts=unknown_facts,
        statistics=state.statistics,
        beliefs=state.beliefs,
        legal_actions=ObservedValue.observed([]),
        observation_start=state.observation_start,
        rule_config=state.rule_config,
    )
    unknown_wall = S2ProtocolState(
        perspective_player=unknown_wall.perspective_player,
        phase=unknown_wall.phase,
        current_player=unknown_wall.current_player,
        current_player_relative=unknown_wall.current_player_relative,
        facts=unknown_wall.facts,
        statistics=unknown_wall.statistics,
        beliefs=unknown_wall.beliefs,
        legal_actions=ObservedValue.observed(legal_actions(unknown_wall)),
        observation_start=unknown_wall.observation_start,
        rule_config=unknown_wall.rule_config,
    )

    encoded = encode_state(unknown_wall)

    actions = unknown_wall.legal_actions.value

    assert len(encoded.values) == encode_state(state).size
    assert any(action.get("kind") == "kong" and action.get("conditionally_legal") for action in actions)
    assert sum(legal_mask(unknown_wall)) == len(actions)

