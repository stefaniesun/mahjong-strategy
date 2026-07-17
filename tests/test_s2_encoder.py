from dataclasses import replace
import hashlib
import struct
import time

import pytest

from engine.hand import Hand
from engine.meld import Meld, MeldKind

from engine.state import GameState
from engine.tiles import Suit, Tile, parse_tile
from state.hand_analysis import shanten, useful_tiles
from state.adapters.from_engine import from_engine
from state.encoder import ENCODER_VERSION, encode_state, encoding_table

from state.observation_degradation import MidGameSnapshot, VisionNoise
from state.protocol import ObservationStatus, ObservedValue

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

    assert first.version == ENCODER_VERSION == "s2.v4.encoder.v4"

    assert first.values == second.values
    assert len(first.values) == first.size == table[-1]["end"] == 893
    assert [section["offset"] for section in table] == sorted(section["offset"] for section in table)
    assert all(left["end"] == right["offset"] for left, right in zip(table, table[1:]))
    assert [(section["name"], section["size"]) for section in table[-6:]] == [
        ("opponent_discard_counts", 87),
        ("opponent_discard_phases", 249),
        ("opponent_meld_tiles", 87),
        ("last_discards", 120),
        ("own_hand_shanten", 4),
        ("candidate_discard_features", 83),
    ]
    assert table[-6]["offset"] == 263
    assert table[-2]["offset"] == 806
    assert table[-1]["offset"] == 810
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
    first_players = [dict(player) for player in state.facts.players.value]
    second_players = [dict(player) for player in state.facts.players.value]
    first_players[1]["concealed_hand"] = ObservedValue.observed(["1W", "1W", "2W", "2W"])
    second_players[1]["concealed_hand"] = ObservedValue.observed(["8B", "8B", "9B", "9B"])
    first = replace(state, facts=replace(state.facts, players=ObservedValue.observed(first_players)))
    second = replace(state, facts=replace(state.facts, players=ObservedValue.observed(second_players)))

    assert encode_state(first).values == encode_state(second).values



def _state_with_public_reading_features():
    state = _base_state()
    players = [dict(player) for player in state.facts.players.value]
    rivers = {
        0: [],
        1: ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"],
        2: ["9B", "9B"],
        3: [],
    }
    melds = {
        0: [],
        1: [{"kind": "pong", "tiles": ["5B"] * 3}],
        2: [{"kind": "kong", "tiles": ["6B"] * 4}],
        3: [],
    }
    for player in players:
        relative = player["relative_position"]
        player["rivers"] = ObservedValue.observed(rivers[relative])
        player["melds"] = ObservedValue.observed(melds[relative])
    return replace(state, facts=replace(state.facts, players=ObservedValue.observed(players)))


def _section_values(encoded, name):
    section = encoded.section(name)
    return encoded.values[section.offset : section.end]


def test_encoder_encodes_public_discards_phase_boundaries_melds_and_last_discards():
    encoded = encode_state(_state_with_public_reading_features())

    discard_counts = _section_values(encoded, "opponent_discard_counts")
    assert discard_counts[parse_tile("1W").index] == 0.25
    assert discard_counts[parse_tile("4T").index] == 0.25
    assert discard_counts[29 + parse_tile("9B").index] == 0.5
    assert discard_counts[27:29] == (0.0, 1.0)

    phases = _section_values(encoded, "opponent_discard_phases")
    assert phases[parse_tile("6W").index] == 0.25
    assert phases[27 + parse_tile("7W").index] == 0.25
    assert phases[27 + parse_tile("3T").index] == 0.25
    assert phases[54 + parse_tile("4T").index] == 0.25
    assert phases[81:83] == (0.0, 1.0)

    meld_tiles = _section_values(encoded, "opponent_meld_tiles")
    assert meld_tiles[parse_tile("5B").index] == 0.75
    assert meld_tiles[29 + parse_tile("6B").index] == 1.0

    last_discards = _section_values(encoded, "last_discards")
    assert last_discards[27:30] == (1.0, 0.0, 1.0)
    assert last_discards[30 + parse_tile("4T").index] == 1.0
    assert last_discards[60 + parse_tile("9B").index] == 1.0
    assert last_discards[90 + 27 : 90 + 30] == (1.0, 0.0, 1.0)


def test_encoder_marks_unknown_and_preserves_estimated_confidence_for_public_features():
    state = _state_with_public_reading_features()
    players = [dict(player) for player in state.facts.players.value]
    players[1]["rivers"] = ObservedValue.unknown()
    players[1]["melds"] = ObservedValue.unknown()
    players[2]["rivers"] = ObservedValue(
        value=["9B", "8B"], status=ObservationStatus.ESTIMATED, confidence=0.65
    )
    degraded = replace(state, facts=replace(state.facts, players=ObservedValue.observed(players)))
    encoded = encode_state(degraded)

    for name, per_player_size in (
        ("opponent_discard_counts", 29),
        ("opponent_discard_phases", 83),
        ("opponent_meld_tiles", 29),
    ):
        values = _section_values(encoded, name)
        assert all(value == 0.0 for value in values[: per_player_size - 2])
        assert values[per_player_size - 2 : per_player_size] == (1.0, 0.0)

    discard_counts = _section_values(encoded, "opponent_discard_counts")
    assert discard_counts[29 + parse_tile("9B").index] == 0.25
    assert discard_counts[29 + 27 : 29 + 29] == (0.0, 0.65)
    last_discards = _section_values(encoded, "last_discards")
    assert last_discards[30 + 28 : 30 + 30] == (1.0, 0.0)
    assert last_discards[60 + parse_tile("8B").index] == 1.0
    assert last_discards[60 + 28 : 60 + 30] == (0.0, 0.65)


def test_new_sections_preserve_the_legacy_prefix_golden_output():
    encoded = encode_state(_state_with_public_reading_features())
    legacy_prefix = encoded.values[:263]
    digest = hashlib.sha256(struct.pack("<263d", *legacy_prefix)).hexdigest()

    assert digest == "9ca61b79d267f3acdd3a978db33088aa217dd6f85d99614f7c1dce6acfb21fe6"
    assert encoded.size == 893



def _expected_candidate_values(state):
    own = next(player for player in state.facts.players.value if player["relative_position"] == 0)
    hand = Hand.from_strings(own["concealed_hand"].value)
    for meld in own["melds"].value:
        hand.add_meld(
            Meld(
                kind=MeldKind(meld["kind"]),
                tiles=tuple(parse_tile(text) for text in meld["tiles"]),
                exposed=meld.get("exposed", False),
                from_player=meld.get("from_player"),
            )
        )
    void = None if own["void_suit"].value is None else Suit(own["void_suit"].value)
    remaining = state.statistics.remaining_tile_counts.value
    expected = []
    for index, count in enumerate(hand.counts):
        if count == 0:
            expected.extend((0.0, 0.0, 0.0))
            continue
        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.remove(Tile.from_index(index))
        useful_remaining = sum(remaining[parse_tile(text).index] for text in useful_tiles(trial, void))
        expected.extend((1.0, max(0.0, shanten(trial, void) / 8.0), min(1.0, useful_remaining / 40.0)))
    return hand, void, tuple(expected)


@pytest.mark.parametrize(
    "tiles,void_suit,meld",
    [
        (["1W", "1W", "2W", "3W", "4W", "5T", "6T", "7T", "2B", "3B", "4B", "9W", "9W", "9W"], Suit.BING, None),
        (["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "5B", "5B"], Suit.BING, None),
        (["1W", "1W", "2W", "2W", "4T", "4T", "6T", "6T", "3B", "3B", "7B", "7B", "9B", "9B"], None, None),
        (["1W", "2W", "3W", "4T", "5T", "6T", "2B", "2B", "9B", "9B", "9B"], Suit.BING, ("pong", ["7W", "7W", "7W"])),
    ],
)
def test_v4_hand_and_candidate_features_match_direct_hand_analysis(tiles, void_suit, meld):
    state = _base_state()
    players = [dict(player) for player in state.facts.players.value]
    own = next(player for player in players if player["relative_position"] == 0)
    own["concealed_hand"] = ObservedValue.observed(tiles)
    own["void_suit"] = ObservedValue.observed(None if void_suit is None else void_suit.value)
    own["melds"] = ObservedValue.observed(
        [] if meld is None else [{"kind": meld[0], "tiles": meld[1], "exposed": True, "from_player": 1}]
    )
    state = replace(state, facts=replace(state.facts, players=ObservedValue.observed(players)))
    hand, void, expected_candidates = _expected_candidate_values(state)
    encoded = encode_state(state)
    remaining = state.statistics.remaining_tile_counts.value
    useful_remaining = sum(remaining[parse_tile(text).index] for text in useful_tiles(hand, void))

    assert _section_values(encoded, "own_hand_shanten") == pytest.approx(
        (max(0.0, shanten(hand, void) / 8.0), min(1.0, useful_remaining / 40.0), 0.0, 1.0)
    )
    candidates = _section_values(encoded, "candidate_discard_features")
    assert candidates[:81] == pytest.approx(expected_candidates)
    assert candidates[81:] == (0.0, 1.0)


def test_v4_remaining_counts_unknown_uses_theoretical_fallback_and_lower_confidence():
    state = _base_state()
    degraded = replace(
        state,
        statistics=replace(state.statistics, remaining_tile_counts=ObservedValue.unknown()),
    )
    encoded = encode_state(degraded)
    own = next(player for player in state.facts.players.value if player["relative_position"] == 0)
    hand = Hand.from_strings(own["concealed_hand"].value)
    void = Suit(own["void_suit"].value)
    theoretical = tuple(4.0 - count for count in hand.counts)
    useful_remaining = sum(theoretical[parse_tile(text).index] for text in useful_tiles(hand, void))

    assert _section_values(encoded, "own_hand_shanten") == pytest.approx(
        (shanten(hand, void) / 8.0, min(1.0, useful_remaining / 40.0), 0.0, 0.5)
    )
    assert _section_values(encoded, "candidate_discard_features")[-2:] == (0.0, 0.5)


def test_v4_missing_own_hand_marks_both_sections_unknown():
    state = _base_state()
    players = [dict(player) for player in state.facts.players.value]
    players[0]["concealed_hand"] = ObservedValue.unknown()
    masked = replace(state, facts=replace(state.facts, players=ObservedValue.observed(players)))
    encoded = encode_state(masked)

    assert _section_values(encoded, "own_hand_shanten") == (0.0, 0.0, 1.0, 0.0)
    candidates = _section_values(encoded, "candidate_discard_features")
    assert candidates[:-2] == (0.0,) * 81
    assert candidates[-2:] == (1.0, 0.0)


def test_v4_features_survive_midgame_and_vision_degradation_and_ignore_opponent_hands():
    state = _base_state()
    baseline = encode_state(state)
    degraded_states = [MidGameSnapshot(k=1).apply(state), VisionNoise(miss_rate=0.2, seed=7).apply(state)]
    for degraded in degraded_states:
        encoded = encode_state(degraded)
        own_values = _section_values(encoded, "own_hand_shanten")
        candidate_values = _section_values(encoded, "candidate_discard_features")
        baseline_candidates = _section_values(baseline, "candidate_discard_features")
        assert own_values[0] == _section_values(baseline, "own_hand_shanten")[0]
        assert own_values[2] == 0.0
        assert candidate_values[-2] == 0.0
        assert candidate_values[0:81:3] == baseline_candidates[0:81:3]
        assert candidate_values[1:81:3] == baseline_candidates[1:81:3]


def test_v4_prefix_through_dimension_805_is_unchanged():
    encoded = encode_state(_state_with_public_reading_features())
    digest = hashlib.sha256(struct.pack("<806d", *encoded.values[:806])).hexdigest()
    assert digest == "596a38c63a4bc1a18729d9d2a3d93b4768c4e8075b5d897bb4f673a26a973236"


def test_v4_cached_feature_overhead_stays_within_v3_budget(monkeypatch):
    import state.encoder as encoder_module

    state = _state_with_public_reading_features()
    encode_state(state)
    iterations = 1000
    started = time.perf_counter()
    for _ in range(iterations):
        encode_state(state)
    v4_elapsed = time.perf_counter() - started

    monkeypatch.setattr(encoder_module, "_own_hand_context", lambda state: None)
    monkeypatch.setattr(encoder_module, "_own_hand_shanten", lambda context: [0.0] * 4)
    monkeypatch.setattr(encoder_module, "_candidate_discard_features", lambda context: [0.0] * 83)
    started = time.perf_counter()
    for _ in range(iterations):
        encode_state(state)
    v3_equivalent_elapsed = time.perf_counter() - started

    assert v4_elapsed <= v3_equivalent_elapsed * 2.0


def test_complete_masked_and_midgame_states_keep_belief_and_legality_chain_available():
    torch = pytest.importorskip("torch")
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from state.action_space import action_space_size, legal_mask
    from state.tile_belief import LearnedBelief, PriorBelief

    complete = _state_with_public_reading_features()
    players = [dict(player) for player in complete.facts.players.value]
    for player in players:
        if player["relative_position"] != 0:
            player["rivers"] = ObservedValue.unknown()
            player["melds"] = ObservedValue.unknown()
    masked = replace(complete, facts=replace(complete.facts, players=ObservedValue.observed(players)))
    midgame = MidGameSnapshot(k=1).apply(complete)
    states = [complete, masked, midgame]
    model = BeliefNet(
        BeliefNetConfig(input_size=encode_state(complete).size, hidden_size=16, residual_blocks=1)
    )
    learned = LearnedBelief(model=model)

    for state in states:
        encoded = encode_state(state)
        assert encoded.size == 893
        assert len(legal_mask(state)) == action_space_size()
        assert PriorBelief().infer(state).source.value == "prior"
        assert learned.infer(state).source.value == "learned"
        assert torch.isfinite(torch.tensor(encoded.values)).all()

    masked_discards = _section_values(encode_state(masked), "opponent_discard_counts")
    masked_melds = _section_values(encode_state(masked), "opponent_meld_tiles")
    assert [masked_discards[offset + 27] for offset in (0, 29, 58)] == [1.0, 1.0, 1.0]
    assert [masked_melds[offset + 27] for offset in (0, 29, 58)] == [1.0, 1.0, 1.0]


