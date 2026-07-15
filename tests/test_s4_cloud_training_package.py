import json
from pathlib import Path

import pytest


pytest.importorskip("torch")


def test_cloud_training_smoke_creates_data_checkpoints_and_reports(tmp_path):
    from tools.cloud_train_s4 import CloudTrainingConfig, run_cloud_training

    output_dir = tmp_path / "cloud-output"
    result = run_cloud_training(
        CloudTrainingConfig(
            output_dir=output_dir,
            games=3,
            max_steps=80,

            seed=123,
            batch_size=2,
            hidden_size=16,
            residual_blocks=1,
            learning_rate=0.01,
            sample_limit=None,
            max_epochs=2,
            patience=2,
            device="cpu",

        )
    )

    assert result.records > 0
    assert result.belief_checkpoint.exists()
    assert result.policy_checkpoint.exists()
    assert result.json_report.exists()
    assert result.markdown_report.exists()

    import torch

    from state.encoder import ENCODER_VERSION

    report = json.loads(result.json_report.read_text(encoding="utf-8"))
    belief_checkpoint = torch.load(result.belief_checkpoint, map_location="cpu")
    policy_checkpoint = torch.load(result.policy_checkpoint, map_location="cpu")

    assert belief_checkpoint["encoder_version"] == ENCODER_VERSION
    assert policy_checkpoint["encoder_version"] == ENCODER_VERSION
    assert report["data"]["records"] == result.records
    assert report["execution"]["device"] == "cpu"

    assert report["checkpoints"]["belief"].endswith("belief_s4.pt")
    assert report["checkpoints"]["policy"].endswith("policy_s4.pt")
    assert report["belief_metrics"]["samples"] > 0
    assert report["policy_metrics"]["samples"] > 0
    assert report["policy_metrics"]["forced_samples"] >= 0
    assert report["policy_train_metrics"]["epochs_trained"] == 2
    assert policy_checkpoint["belief_metadata"]["source"] == "learned"
    split_games = report["data"]["split_game_ids"]
    assert set(split_games["train"]).isdisjoint(split_games["val"])
    assert set(split_games["train"]).isdisjoint(split_games["test"])
    assert set(split_games["val"]).isdisjoint(split_games["test"])



def test_data_fingerprint_changes_when_source_changes(tmp_path):
    from tools.cloud_train_s4 import data_fingerprint

    source = tmp_path / "records.jsonl"
    source.write_text('{"a": 1}\n', encoding="utf-8")
    first = data_fingerprint(source)
    source.write_text('{"a": 2}\n', encoding="utf-8")
    second = data_fingerprint(source)

    assert first != second
    assert len(first) == 64
    assert len(second) == 64
