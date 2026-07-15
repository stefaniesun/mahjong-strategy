from learning.datasets.dataset_builder import DatasetBuildConfig, _clean_action, build_belief_sample, build_policy_sample, load_decision_records
from learning.datasets.splits import split_records_by_game
from selfplay.data_recorder import run_recorded_selfplay_game, write_jsonl
from state.action_space import action_to_index
from state.tile_belief import PriorBelief







def _records_for_games():
    records = []
    for seed in range(1, 5):
        _, game_records = run_recorded_selfplay_game(game_id=f"game-{seed}", seed=seed, max_steps=600)
        records.extend(game_records)
    return records


def test_split_records_by_game_is_reproducible_and_has_no_game_leakage():
    records = _records_for_games()

    first = split_records_by_game(records, seed=7, ratios=(0.5, 0.25, 0.25))
    second = split_records_by_game(records, seed=7, ratios=(0.5, 0.25, 0.25))

    assert first == second
    split_games = {name: {record.game_id for record in items} for name, items in first.items()}
    assert split_games["train"]
    assert split_games["val"]
    assert split_games["test"]
    assert split_games["train"].isdisjoint(split_games["val"])
    assert split_games["train"].isdisjoint(split_games["test"])
    assert split_games["val"].isdisjoint(split_games["test"])


def test_load_decision_records_reads_jsonl_and_returns_split_records(tmp_path):
    records = _records_for_games()
    path = tmp_path / "s3_training.jsonl"
    write_jsonl(records, path)

    loaded = load_decision_records(path, seed=11, ratios=(0.5, 0.25, 0.25))

    assert sum(len(items) for items in loaded.values()) == len(records)
    assert {record.to_dict()["game_id"] for record in loaded["train"]}


def test_build_belief_sample_masks_belief_section_and_keeps_oracle_labels_out_of_input():
    record = _records_for_games()[0]

    sample = build_belief_sample(record, DatasetBuildConfig(seed=3, degradation_profile="perfect"))

    encoded_without_beliefs = build_belief_sample(record, DatasetBuildConfig(seed=3, degradation_profile="perfect")).encoded

    assert sample.game_id == record.game_id

    assert sample.player == record.player
    assert sample.labels == record.labels
    assert sample.encoded == encoded_without_beliefs
    assert sample.encoded.section("tile_location_beliefs")
    assert record.labels["tile_locations"] != {}


def test_build_policy_sample_has_action_index_and_legal_mask_covering_s3_action():
    record = _records_for_games()[0]

    sample = build_policy_sample(record, DatasetBuildConfig(seed=5, degradation_profile="perfect"))

    assert sample.game_id == record.game_id
    assert sample.action_index == action_to_index(_clean_action(record.action))
    assert len(sample.legal_mask) > sample.action_index

    assert sample.legal_mask[sample.action_index] is True
    assert any(sample.legal_mask)
    assert sample.action_kind == _clean_action(record.action)["kind"]
    assert sample.legal_action_count == sum(sample.legal_mask)


def test_build_policy_sample_accepts_explicit_belief_provider():
    class TrackingBelief(PriorBelief):
        calls = 0

        def infer(self, state):
            self.calls += 1
            return super().infer(state)

    record = _records_for_games()[0]
    belief = TrackingBelief(source="explicit-prior")

    sample = build_policy_sample(
        record,
        DatasetBuildConfig(seed=5, degradation_profile="perfect"),
        belief=belief,
    )

    assert belief.calls == 1
    assert sample.encoded.section("tile_location_beliefs")


