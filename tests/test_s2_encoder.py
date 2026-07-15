from dataclasses import replace

from engine.hand import Hand
from engine.state import GameState
from engine.tiles import Suit, parse_tile
from state.adapters.from_engine import from_engine
from state.encoder import ENCODER_VERSION, encode_state, encoding_table

from state.observation_degradation import MidGameSnapshot, VisionNoise
from state.protocol import ObservedValue
from state.tile_belief import with_prior_beliefs


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _base_state():
    engine_state = GameState(
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
    engine_state.rivers = [[parse_tile("5W")], [parse_tile("2W")], [], []]
    engine_state.event_log = [{"seq": 0, "type": "discard", "player": 2, "tile": "8T"}]
    return from_engine(engine_state, player_id=0)


def test_encode_state_is_fixed_shape_deterministic_and_has_documented_sections():
    state = _base_state()

    first = encode_state(state)
    second = encode_state(state)
    table = encoding_table()

    assert first.version == ENCODER_VERSION == "s2.v4.encoder.v2"

    assert first.values == second.values
    assert len(first.values) == first.size == table[-1]["end"]
    assert [section["offset"] for section in table] == sorted(section["offset"] for section in table)
    assert {section["name"] for section in table} >= {
        "own_hand_counts",
        "seen_counts",
        "remaining_tile_counts",
        "unknown_pool_breakdown",
        "player_void_suits",
        "tile_location_beliefs",

        "observation_summary",
    }


def test_unknown_and_estimated_values_have_flags_and_confidence_without_shape_changes():
    state = _base_state()
    masked = replace(state, facts=replace(state.facts, wall_count=ObservedValue.unknown()))
    noisy = VisionNoise(confusion_matrix={"5W": {"6W": 1.0}}, seed=1).apply(state)

    encoded = encode_state(state)
    masked_encoded = encode_state(masked)
    noisy_encoded = encode_state(noisy)

    assert len(encoded.values) == len(masked_encoded.values) == len(noisy_encoded.values)

    wall_range = encoded.section("wall_count")
    assert encoded.values[wall_range.offset] == 3 / 108
    assert encoded.values[wall_range.offset + 1] == 0.0
    assert encoded.values[wall_range.offset + 2] == 1.0
    assert masked_encoded.values[wall_range.offset] == 0.0
    assert masked_encoded.values[wall_range.offset + 1] == 1.0
    assert masked_encoded.values[wall_range.offset + 2] == 0.0

    seen_range = noisy_encoded.section("seen_counts")
    assert noisy_encoded.values[seen_range.offset + 27] == 0.0
    assert noisy_encoded.values[seen_range.offset + 28] == 0.8


def test_encoder_handles_midgame_degraded_observations_with_same_shape():
    state = _base_state()
    midgame = with_prior_beliefs(MidGameSnapshot(k=1).apply(state))

    encoded = encode_state(state)
    degraded = encode_state(midgame)

    assert len(encoded.values) == len(degraded.values)
    observation_range = degraded.section("observation_summary")
    assert degraded.values[observation_range.offset] == 1 / 108
    assert degraded.values[observation_range.offset + 3] == 1.0
    assert degraded.values[observation_range.offset + 4] < encoded.values[observation_range.offset + 4]



def test_encoder_tracks_public_void_suits_by_relative_position_only():
    state = _base_state()
    changed_players = [dict(player) for player in state.facts.players.value]
    changed_players[2]["void_suit"] = ObservedValue.estimated("W", 0.6)
    changed = replace(state, facts=replace(state.facts, players=ObservedValue.observed(changed_players)))

    original = encode_state(state)
    updated = encode_state(changed)
    section = original.section("player_void_suits")

    assert section.size == 20
    assert original.values[: section.offset] == updated.values[: section.offset]
    assert original.values[section.end :] == updated.values[section.end :]
    assert updated.values[section.offset + 10 : section.offset + 15] == (1.0, 0.0, 0.0, 0.0, 0.6)


def test_encoder_does_not_use_hidden_opponent_concealed_hands():

    state = _base_state()
    changed_players = [dict(player) for player in state.facts.players.value]
    changed_players[1]["concealed_hand"] = ObservedValue.unknown()
    changed_players[1]["hand_count"] = ObservedValue.observed(4)
    hidden_changed = replace(state, facts=replace(state.facts, players=ObservedValue.observed(changed_players)))

    assert encode_state(state).values == encode_state(hidden_changed).values
