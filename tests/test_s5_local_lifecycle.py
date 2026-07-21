from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
from rl.league import DualArenaMetrics, OpponentLeague, SnapshotMetadata
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from rl.train_rl import S5TrainingConfig, S5TrainingDependencies, run_s5_training
from rl.types import TrajectoryStep
from tools.cloud_train_s5 import S5CloudRunConfig, parse_args


def _files(tmp_path: Path) -> tuple[Path, Path]:
    belief, policy = tmp_path / "belief.pt", tmp_path / "policy.pt"
    belief.write_bytes(b"belief")
    policy.write_bytes(b"policy")
    return belief, policy


def _dependencies(indices: list[int], stop_file: Path | None = None) -> S5TrainingDependencies:
    def rollout(_model, _config, update, _seed, _runtime):
        indices.append(update)
        if stop_file is not None:
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.touch()
        return (
            TrajectoryStep((0.1, 0.2, 0.3), (True, True, False, False), 0, -0.69, 0.0, 0.0, False, "s4"),
            TrajectoryStep((0.3, 0.2, 0.1), (True, False, True, False), 2, -0.69, 0.0, 0.5, True, "s4"),
        )

    return S5TrainingDependencies(
        model_factory=lambda _config: PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0)),
        rollout_factory=rollout,
        arena_evaluator=lambda *_args: DualArenaMetrics(0.7, 0.6, 1000, 1000, 0, 0),
        league_factory=lambda: OpponentLeague(current_policy=SnapshotMetadata("cold-start", "v0", "current.pt", 0)),
        curriculum_factory=lambda: ObservationCurriculum(CurriculumConfig((CurriculumStage("only", DegradationProfile.perfect(), 1.0, 1.0),))),
    )


def _config(tmp_path: Path, **overrides) -> S5TrainingConfig:
    belief, policy = _files(tmp_path)
    values = dict(
        output_dir=tmp_path / "out",
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
        updates=3,
        device="cpu",
        episodes_per_update=4,
    )
    values.update(overrides)
    return S5TrainingConfig(**values)


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stop_is_observed_after_checkpoint_and_metric_publication(tmp_path) -> None:
    stop_file = tmp_path / "control" / "stop.flag"
    metrics_file = tmp_path / "out" / "metrics.jsonl"
    indices: list[int] = []

    result = run_s5_training(
        _config(tmp_path, stop_file=stop_file, metrics_file=metrics_file),
        dependencies=_dependencies(indices, stop_file),
    )

    rows = _rows(metrics_file)
    assert result.global_step == 1
    assert result.total_episodes == 4
    assert result.stopped is True
    assert result.checkpoint_path.is_file()
    assert indices == [0]
    assert rows[0]["checkpoint"] == str(result.checkpoint_path)
    assert rows[0]["global_step"] == 1
    assert rows[0]["total_episodes"] == 4
    assert rows[0]["curriculum_stage"] == "only"
    assert rows[0]["league_size"] == 5
    assert rows[0]["illegal_actions"] == 0
    assert rows[0]["zero_sum_failures"] == 0
    assert {"timestamp", "duration_seconds", "episodes_per_minute", "policy_loss", "value_loss", "entropy", "kl"} <= rows[0].keys()


def test_resume_appends_continuous_update_and_episode_counts(tmp_path) -> None:
    metrics_file = tmp_path / "out" / "metrics.jsonl"
    indices: list[int] = []
    first = run_s5_training(_config(tmp_path, updates=1, metrics_file=metrics_file), dependencies=_dependencies(indices))
    resumed = run_s5_training(
        _config(tmp_path, updates=1, metrics_file=metrics_file, resume_checkpoint=first.checkpoint_path),
        dependencies=_dependencies(indices),
    )

    rows = _rows(metrics_file)
    assert [row["update"] for row in rows] == [0, 1]
    assert [row["global_step"] for row in rows] == [1, 2]
    assert [row["total_episodes"] for row in rows] == [4, 8]
    assert indices == [0, 1]
    assert resumed.total_episodes == 8


def test_cloud_cli_exposes_lifecycle_paths_and_manifest_config_is_json_safe() -> None:
    args = parse_args(["--resume", "old.pt", "--stop-file", "stop.flag", "--metrics-file", "metrics.jsonl"])
    assert args.resume == Path("old.pt")
    assert args.stop_file == Path("stop.flag")
    assert args.metrics_file == Path("metrics.jsonl")

    config = S5CloudRunConfig(Path("out"), resume_checkpoint=Path("old.pt"), stop_file=Path("stop.flag"), metrics_file=Path("metrics.jsonl"))
    json.dumps(config.serializable_dict())


def test_diagnostic_checkpoint_is_not_resumable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    indices: list[int] = []
    config = _config(tmp_path, updates=1)

    from rl.train_rl import save_checkpoint as real_save_checkpoint

    def fail_save(*args, **kwargs):
        path = Path(args[0])
        if path.name == "latest.pt":
            raise OSError("disk failure")
        return real_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr("rl.train_rl.save_checkpoint", fail_save)
    with pytest.raises(OSError, match="disk failure"):
        run_s5_training(config, dependencies=_dependencies(indices))

    diagnostic = config.output_dir / "checkpoints" / "diagnostic.pt"
    renamed = diagnostic.with_name("renamed.pt")
    diagnostic.replace(renamed)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="diagnostic checkpoint"):
        run_s5_training(_config(tmp_path, updates=1, resume_checkpoint=renamed), dependencies=_dependencies([]))


def test_immutable_snapshot_restores_total_episodes(tmp_path) -> None:
    indices: list[int] = []
    first = run_s5_training(_config(tmp_path, updates=1), dependencies=_dependencies(indices))
    snapshot = first.checkpoint_path.parent / "snapshots" / "s5-step-1.pt"

    resumed = run_s5_training(
        _config(tmp_path, updates=1, resume_checkpoint=snapshot),
        dependencies=_dependencies(indices),
    )

    assert resumed.total_episodes == 8
    assert indices == [0, 1]
