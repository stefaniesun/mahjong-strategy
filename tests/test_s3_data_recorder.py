import json

from selfplay.data_recorder import load_jsonl, run_recorded_selfplay_game, write_jsonl
from state.protocol import ObservationStatus, S2ProtocolState


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False





def test_recorded_selfplay_produces_protocol_decision_records_without_private_hand_leakage():
    result, records = run_recorded_selfplay_game(game_id="g-1", seed=1, max_steps=600)

    assert result.finished is True
    assert records
    assert all(record.game_id == "g-1" for record in records)
    assert all(record.final_scores == result.scores for record in records)
    assert result.steps == len(records)
    assert [record.step for record in records] == list(range(result.steps))

    for record in records:

        assert record.state["version"] == "s2.v4"
        assert record.state["perspective_player"] == record.player
        assert record.legal_actions
        assert record.action in record.legal_actions
        assert set(record.labels) == {"tile_locations", "opponent_tenpai", "discard_danger"}
        assert not _contains_key(
            record.state,
            {"winners", "winner_relatives", "pending_winners"},
        )

        protocol = S2ProtocolState.from_dict(record.state)

        assert protocol.facts.players.status is ObservationStatus.OBSERVED
        for player in protocol.facts.players.value:
            concealed = player["concealed_hand"]
            if player["relative_position"] == 0:
                assert concealed.status is ObservationStatus.OBSERVED
                assert concealed.value
            elif not player["won"].value:
                assert concealed.status is ObservationStatus.UNKNOWN
                assert concealed.value is None



def test_recorded_selfplay_is_reproducible_for_same_seed():
    first_result, first_records = run_recorded_selfplay_game(game_id="same", seed=7, max_steps=600)
    second_result, second_records = run_recorded_selfplay_game(game_id="same", seed=7, max_steps=600)

    assert first_result == second_result
    assert [record.to_dict() for record in first_records] == [record.to_dict() for record in second_records]


def test_jsonl_round_trip_loads_one_record_per_line(tmp_path):
    _, records = run_recorded_selfplay_game(game_id="jsonl", seed=3, max_steps=600)
    output_path = tmp_path / "s3_selfplay.jsonl"

    write_jsonl(records, output_path)
    loaded = load_jsonl(output_path)

    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert all(json.loads(line)["game_id"] == "jsonl" for line in lines)
