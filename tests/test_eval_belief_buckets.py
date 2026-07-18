from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import sys
import zipfile

import pytest


@dataclass(frozen=True)
class _Record:
    game_id: str
    step: int = 0


def _game_id_for_split(target: str, seed: int) -> str:
    from tools.cloud_train_s4_50k_cached import _split_name

    for number in range(10_000):
        game_id = f"bucket-test-{number}"
        if _split_name(game_id, seed) == target:
            return game_id
    raise AssertionError(f"no {target} id found")


def test_validation_selection_uses_cloud_training_game_split_semantics():
    from tools.eval_belief_buckets import select_validation_records

    seed = 20260716
    val_id = _game_id_for_split("val", seed)
    train_id = _game_id_for_split("train", seed)
    selected = select_validation_records([_Record(train_id), _Record(val_id)], seed=seed)

    assert [record.game_id for record in selected] == [val_id]


def test_wall_count_phase_buckets_have_required_inclusive_boundaries():
    from tools.eval_belief_buckets import phase_bucket_from_wall_count

    assert phase_bucket_from_wall_count(41) == "opening"
    assert phase_bucket_from_wall_count(40) == "midgame"
    assert phase_bucket_from_wall_count(20) == "midgame"
    assert phase_bucket_from_wall_count(19) == "endgame"


def _passing_metrics():
    return {
        profile: {
            "opening": {"model_tile_log_loss": 0.9, "prior_tile_log_loss": 1.1, "gain": 0.2},
            "midgame": {"model_tile_log_loss": 0.8, "prior_tile_log_loss": 1.1, "gain": 0.3},
            "endgame": {"model_tile_log_loss": 0.7, "prior_tile_log_loss": 1.1, "gain": 0.4},
        }
        for profile in ("perfect", "light_noise", "midgame", "heavy")
    }


@pytest.mark.parametrize("field, value", [
    ("model_tile_log_loss", float("nan")),
    ("prior_tile_log_loss", float("inf")),
    ("gain", float("-inf")),
    ("samples", float("nan")),
])
def test_acceptance_rejects_nonfinite_bucket_values(field, value):
    from tools.eval_belief_buckets import evaluate_acceptance

    metrics = _passing_metrics()
    metrics["perfect"]["opening"][field] = value

    result = evaluate_acceptance(metrics)

    assert not result.passed
    assert any("not finite" in reason for reason in result.reasons)


def test_acceptance_rejects_undersized_source():
    from tools.eval_belief_buckets import evaluate_acceptance

    result = evaluate_acceptance(_passing_metrics(), source_records=19_999, target_validation_records=20_000)

    assert not result.passed
    assert any("target" in reason for reason in result.reasons)


def test_rendering_never_marks_undersized_or_nonfinite_report_as_pass():
    from tools.eval_belief_buckets import render_report

    metrics = _passing_metrics()
    metrics["heavy"]["endgame"]["gain"] = float("nan")
    report = {
        "checkpoint": {"path": "belief_s4.pt", "sha256": "abc", "encoder_version": "s2.v4.encoder.v4"},
        "source": {"kind": "test", "records": 1, "game_id_range": "test"},
        "settings": {"seed": 7, "target_validation_records": 2},
        "metrics": metrics,
    }

    rendered = render_report(report)

    assert "**PASS**" not in rendered
    assert "**EXCEPTION**" in rendered


def test_load_checkpoint_rejects_a_previous_encoder_version(tmp_path):
    import torch
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from tools.eval_belief_buckets import _load_checkpoint

    model = BeliefNet(BeliefNetConfig(input_size=893))
    path = tmp_path / "old-encoder.pt"
    torch.save(
        {"model_config": model.config.__dict__, "state_dict": model.state_dict(), "encoder_version": "s2.v4.encoder.v3"},
        path,
    )

    with pytest.raises(ValueError, match="encoder"):
        _load_checkpoint(path)


def _write_package(path, shard_lines):
    compressed = gzip.compress("".join(json.dumps(line) + "\n" for line in shard_lines).encode("utf-8"))
    manifest = {"schema": "s2.v4", "shards": [{"data_file": "records.jsonl.gz", "bytes": len(compressed)}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/manifest.json", json.dumps(manifest))
        archive.writestr("data/records.jsonl.gz", compressed)


def test_package_reader_streams_only_validation_records_in_order_and_stops_at_target(tmp_path, monkeypatch):
    import tools.eval_belief_buckets as buckets

    package = tmp_path / "records.zip"
    _write_package(
        package,
        [
            {"game_id": "train-1", "step": 0},
            {"game_id": "val-1", "step": 1},
            {"game_id": "val-1", "step": 2},
            {"game_id": "val-2", "step": 3},
        ],
    )

    class FakeDecisionRecord:
        @staticmethod
        def from_dict(payload):
            return _Record(payload["game_id"], payload["step"])

    monkeypatch.setattr(buckets, "DecisionRecord", FakeDecisionRecord)
    monkeypatch.setattr(buckets, "_split_name", lambda game_id, _: "val" if game_id.startswith("val") else "train")

    records = buckets._records_from_package(package, target_validation_records=2, split_seed=11)

    assert records == [_Record("val-1", 1), _Record("val-1", 2)]


def test_cli_writes_exception_report_and_exits_nonzero_when_evaluation_fails(tmp_path, monkeypatch):
    import tools.eval_belief_buckets as buckets

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(buckets, "build_report", lambda **_: (_ for _ in ()).throw(RuntimeError("broken evaluation")))
    monkeypatch.setattr(sys, "argv", ["eval_belief_buckets.py", "--report", str(report_path)])

    with pytest.raises(SystemExit, match="1"):
        buckets.main()

    rendered = report_path.read_text(encoding="utf-8")
    assert "**EXCEPTION**" in rendered
    assert "broken evaluation" in rendered
    assert "**PASS**" not in rendered


def test_cli_writes_gate_failure_report_to_stderr_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    import tools.eval_belief_buckets as buckets

    report_path = tmp_path / "report.md"
    report = {
        "checkpoint": {"path": "belief_s4.pt", "sha256": "abc", "encoder_version": "s2.v4.encoder.v4"},
        "source": {"kind": "test", "records": 1, "game_id_range": "test"},
        "settings": {"seed": 7, "target_validation_records": 2},
        "metrics": _passing_metrics(),
    }
    monkeypatch.setattr(buckets, "build_report", lambda **_: report)
    monkeypatch.setattr(sys, "argv", ["eval_belief_buckets.py", "--report", str(report_path)])

    with pytest.raises(SystemExit, match="1"):
        buckets.main()

    rendered = report_path.read_text(encoding="utf-8")
    assert "**EXCEPTION**" in rendered
    assert "validation source records 1 are below target 2" in rendered
    assert capsys.readouterr().err == rendered


def test_cli_returns_zero_for_passing_normal_report(tmp_path, monkeypatch, capsys):
    import tools.eval_belief_buckets as buckets

    report_path = tmp_path / "report.md"
    report = {
        "checkpoint": {"path": "belief_s4.pt", "sha256": "abc", "encoder_version": "s2.v4.encoder.v4"},
        "source": {"kind": "test", "records": 2, "game_id_range": "test"},
        "settings": {"seed": 7, "target_validation_records": 2},
        "metrics": _passing_metrics(),
    }
    monkeypatch.setattr(buckets, "build_report", lambda **_: report)
    monkeypatch.setattr(sys, "argv", ["eval_belief_buckets.py", "--report", str(report_path)])

    buckets.main()

    rendered = report_path.read_text(encoding="utf-8")
    assert "**PASS**" in rendered
    assert capsys.readouterr().out == rendered


def test_acceptance_requires_endgame_gain_above_opening_and_all_profiles_beat_prior():
    from tools.eval_belief_buckets import evaluate_acceptance

    passing = _passing_metrics()
    assert evaluate_acceptance(passing).passed is True

    failing = _passing_metrics()
    failing["heavy"]["endgame"]["gain"] = 0.1
    failing["light_noise"]["midgame"]["model_tile_log_loss"] = 1.2
    result = evaluate_acceptance(failing)

    assert result.passed is False
    assert any("endgame" in reason for reason in result.reasons)
    assert any("light_noise" in reason for reason in result.reasons)


def test_report_rendering_is_deterministic_and_records_required_provenance():
    from tools.eval_belief_buckets import render_report

    report = {
        "checkpoint": {"path": "belief_s4.pt", "sha256": "abc", "encoder_version": "s2.v4.encoder.v4"},
        "source": {"kind": "deterministic_regeneration", "records": 12, "game_id_range": "s4eval-1..s4eval-4"},
        "settings": {"seed": 7, "target_validation_records": 12},
        "metrics": _passing_metrics(),
    }

    first = render_report(report)
    second = render_report(report)

    assert first == second
    assert "s2.v4.encoder.v4" in first
    assert "deterministic_regeneration" in first
    assert "PASS" in first
    assert "endgame" in first
    assert first.isascii()


def test_regeneration_reports_deterministic_periodic_progress(monkeypatch):
    import tools.eval_belief_buckets as buckets

    record = _Record("generated")
    monkeypatch.setattr(buckets, "run_recorded_selfplay_game", lambda **_: (None, [record]))
    progress: list[tuple[int, int]] = []

    records = buckets._regenerate_records(
        seed=11, games=5, progress_every=2, on_progress=lambda completed, total: progress.append((completed, total))
    )

    assert records == [record] * 5
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_streaming_regeneration_retains_only_val_records_in_cloud_split_order(monkeypatch):
    import tools.eval_belief_buckets as buckets

    seen: list[str] = []

    def fake_game(**kwargs):
        game_id = kwargs["game_id"]
        seen.append(game_id)
        return None, [_Record(game_id), _Record(game_id, step=1)]

    monkeypatch.setattr(buckets, "run_recorded_selfplay_game", fake_game)
    monkeypatch.setattr(buckets, "_split_name", lambda game_id, _: "val" if game_id.endswith("-12") else "train")

    retained = buckets._regenerate_validation_records(seed=10, games=5, split_seed=12)

    assert seen == ["belief-bucket-10", "belief-bucket-11", "belief-bucket-12", "belief-bucket-13", "belief-bucket-14"]
    assert retained == [_Record("belief-bucket-12"), _Record("belief-bucket-12", step=1)]


def test_preselected_val_seeds_match_streaming_all_candidate_selection_and_order(monkeypatch):
    import tools.eval_belief_buckets as buckets

    def fake_game(**kwargs):
        game_id = kwargs["game_id"]
        return None, [_Record(game_id), _Record(game_id, step=1)]

    monkeypatch.setattr(buckets, "run_recorded_selfplay_game", fake_game)
    monkeypatch.setattr(buckets, "_split_name", lambda game_id, _: "val" if game_id.endswith(("-12", "-14")) else "train")

    full_stream = buckets._regenerate_validation_records(seed=10, games=5, split_seed=12)
    selected_seeds = buckets._candidate_val_seeds(start_seed=10, candidate_games=5, split_seed=12)
    preselected = buckets._regenerate_selected_validation_records(selected_seeds)

    assert selected_seeds == (12, 14)
    assert preselected == full_stream


def test_cached_prior_evaluation_matches_legacy_record_evaluation():
    from learning.datasets.dataset_builder import DatasetBuildConfig, build_belief_sample
    from learning.eval.eval_belief import evaluate_prior_belief_records, evaluate_prior_belief_samples
    from selfplay.data_recorder import run_recorded_selfplay_game

    _, records = run_recorded_selfplay_game(game_id="cached-prior-equivalence", seed=81, max_steps=240)
    records = records[:4]
    config = DatasetBuildConfig(seed=17, degradation_profile="heavy")
    samples = [build_belief_sample(record, config) for record in records]

    assert evaluate_prior_belief_samples(records, samples, config) == evaluate_prior_belief_records(records, config)
