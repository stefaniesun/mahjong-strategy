from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
from rl.league import DualArenaMetrics, OpponentLeague, SnapshotMetadata
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from rl.ppo_trainer import PPOHealth
from rl.train_rl import HealthThresholds, S5TrainingConfig, S5TrainingDependencies, run_s5_training
from rl.types import TrajectoryStep


class _UnsafeS4Artifact:
    pass


def _steps() -> tuple[TrajectoryStep, ...]:
    return (
        TrajectoryStep((0.1, 0.2, 0.3), (True, True, False, False), 0, -0.69, 0.0, 0.0, False, "s4"),
        TrajectoryStep((0.3, 0.2, 0.1), (True, False, True, False), 2, -0.69, 0.0, 0.5, True, "s4"),
    )


def _league_factory() -> OpponentLeague:
    return OpponentLeague(current_policy=SnapshotMetadata("cold-start", "v0", "current.pt", 0))


def _curriculum_factory() -> ObservationCurriculum:
    return ObservationCurriculum(CurriculumConfig((
        CurriculumStage("perfect", DegradationProfile.perfect(), 0.0, 0.0),
        CurriculumStage("noise", DegradationProfile("noise", 0.1), 0.0, 0.0),
    )))


def _files(tmp_path):
    belief, policy = tmp_path / "belief.pt", tmp_path / "policy.pt"
    belief.write_bytes(b"frozen-belief")
    policy.write_bytes(b"frozen-policy")
    return belief, policy


def _model_factory(_config: S5TrainingConfig) -> PolicyValueNet:
    return PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))


def test_training_config_rejects_non_json_s4_provenance_before_rollout(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    with pytest.raises(TypeError, match="JSON serializable"):
        S5TrainingConfig(
            output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy,
            frozen_s4_provenance={"bad": object()},
        )


def test_default_model_rejects_unsafe_s4_policy_artifact(tmp_path) -> None:
    from rl.train_rl import _default_model

    belief, policy = _files(tmp_path)
    torch.save(_UnsafeS4Artifact(), policy)
    config = S5TrainingConfig(
        output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
    )
    with pytest.raises(pickle.UnpicklingError, match="weights_only"):
        _default_model(config)


def test_default_model_infers_the_archived_s4_policy_architecture() -> None:
    """The production S4 archive must boot S5 without stale placeholder sizes."""
    from rl.train_rl import _default_model

    root = Path(__file__).resolve().parents[1]
    archive = root / "training_artifacts" / "S4" / "v1_20260711_repaired_cuda"
    belief = archive / "checkpoints" / "belief_s4.pt"
    policy = archive / "checkpoints" / "policy_s4.pt"
    if not belief.is_file() or not policy.is_file():
        pytest.skip("local S4 v1 archive is unavailable")

    model = _default_model(S5TrainingConfig(
        output_dir=archive / "_test_output",
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
    ))
    payload = torch.load(policy, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    assert model.config == PolicyValueNetConfig(input_size=263, action_size=637, hidden_size=128, residual_blocks=2, dropout=0.0)
    assert torch.equal(model.trunk[0].weight, state["trunk.0.weight"])
    assert torch.equal(model.action_head.weight, state["action_head.weight"])


def test_default_model_rejects_explicit_dimensions_that_conflict_with_s4_policy(tmp_path) -> None:
    """An override must fail clearly instead of producing a partially loaded S5 model."""
    from rl.train_rl import _default_model
    from learning.models.policy_net import PolicyNet, PolicyNetConfig

    belief = tmp_path / "belief.pt"
    policy = tmp_path / "policy.pt"
    belief.write_bytes(b"frozen-belief")
    source = PolicyNet(PolicyNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    torch.save({
        "model_config": {"input_size": 3, "action_size": 4, "hidden_size": 8, "residual_blocks": 0, "dropout": 0.0},
        "state_dict": source.state_dict(),
    }, policy)
    config = S5TrainingConfig(
        output_dir=tmp_path / "out",
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
        action_size=5,
    )
    with pytest.raises(ValueError, match=r"action_size=5 does not match frozen S4 policy action_size=4"):
        _default_model(config)


def _per_entry_evaluator(_model, opponent, _runtime) -> DualArenaMetrics:
    assert opponent.key in {"current", "cold-start", "s3", "greedy", "random", "s5-1", "s5-2"}
    return DualArenaMetrics(0.7, 0.6, 1000, 1000, 0, 0)


def test_two_update_cpu_smoke_writes_immutable_per_entry_arena_and_health_artifacts(tmp_path) -> None:
    belief = tmp_path / "belief.pt"
    policy = tmp_path / "policy.pt"
    belief.write_bytes(b"frozen-belief")
    policy.write_bytes(b"frozen-policy")
    model_config = PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0)

    def rollout_factory(_model, _config, update, next_seed, _runtime):
        assert next_seed == 100 + update
        return _steps()

    config = S5TrainingConfig(
        output_dir=tmp_path / "out",
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
        updates=2,
        rollout_seed_start=100,
        snapshot_interval=1,
        device="cpu",
    )
    result = run_s5_training(
        config,
        dependencies=S5TrainingDependencies(
            model_factory=_model_factory,
            rollout_factory=rollout_factory,
            arena_evaluator=_per_entry_evaluator,
            league_factory=_league_factory,
            curriculum_factory=_curriculum_factory,
        ),
    )
    assert result.global_step == 2
    assert result.report_path.exists() and result.markdown_path.exists()
    report = result.report
    assert set(report["arena"]) == {"by_opponent", "s3_comparison"}
    assert report["arena"]["s3_comparison"] == {
        "perfect_win_rate": 0.7,
        "degraded_win_rate": 0.6,
        "perfect_games": 1000,
        "degraded_games": 1000,
        "illegal_actions": 0,
        "zero_sum_failures": 0,
    }
    assert set(report["arena"]["by_opponent"]) == {"current", "cold-start", "s3", "greedy", "random"}
    assert len(report["health"]) == 2
    assert (config.output_dir / "checkpoints" / "latest.pt").exists()
    assert result.curriculum.stage_index == 1
    snapshots = sorted((config.output_dir / "checkpoints" / "snapshots").glob("*.pt"))
    assert [path.name for path in snapshots] == ["s5-step-1.pt", "s5-step-2.pt"]
    assert len({hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots}) == 2
    history = result.league.historical_entries
    assert [entry.key for entry in history] == ["cold-start", "s5-1"]
    assert [entry.snapshot.snapshot_id for entry in history] == ["cold-start", "s5-1"]
    assert [entry.snapshot.checkpoint_path for entry in history] == ["current.pt", str(snapshots[0])]
    assert result.league.current_entry.snapshot.snapshot_id == "s5-2"
    assert result.league.current_entry.snapshot.checkpoint_path == str(snapshots[1])
    parsed = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert parsed["arena"] == report["arena"]
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "S3 comparison" in markdown and "perfect_win_rate" in markdown


def test_resumed_run_uses_checkpointed_next_rollout_seed(tmp_path) -> None:
    belief, policy = tmp_path / "belief.pt", tmp_path / "policy.pt"
    belief.write_bytes(b"belief")
    policy.write_bytes(b"policy")
    observed_seeds: list[int] = []
    observed_runtime_snapshots: list[str] = []
    observed_runtime_generations: list[str] = []
    observed_runtime_checksums: list[str] = []

    def model_factory(_config):
        return PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))

    def rollout_factory(_model, _config, _update, seed, runtime):
        observed_seeds.append(seed)
        observed_runtime_snapshots.append(runtime.league.current_entry.snapshot.snapshot_id)
        observed_runtime_generations.append(runtime.policy_generation)
        observed_runtime_checksums.append(runtime.policy_checksum)
        assert runtime.curriculum is not None
        return _steps()

    def evaluator(_model, _opponent, runtime):
        assert runtime.league.current_entry.snapshot.snapshot_id in {"cold-start", "s5-1"}
        assert runtime.curriculum is not None
        return DualArenaMetrics(0.6, 0.6, 1000, 1000, 0, 0)

    deps = S5TrainingDependencies(model_factory=model_factory, rollout_factory=rollout_factory, arena_evaluator=evaluator, league_factory=_league_factory, curriculum_factory=lambda: ObservationCurriculum(CurriculumConfig((CurriculumStage("only", DegradationProfile.perfect(), 1.0, 1.0),))))
    base = dict(output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"}, updates=1, rollout_seed_start=41, device="cpu")
    first = run_s5_training(S5TrainingConfig(**base), dependencies=deps)
    resumed = run_s5_training(S5TrainingConfig(**base, resume_checkpoint=first.checkpoint_path), dependencies=deps)
    assert observed_seeds == [41, 42]
    assert observed_runtime_snapshots == ["cold-start", "s5-1"]
    assert observed_runtime_generations == ["s5-step-0", "s5-step-1"]
    assert len(observed_runtime_checksums) == 2
    assert all(len(checksum) == 64 for checksum in observed_runtime_checksums)
    assert observed_runtime_checksums[0] != observed_runtime_checksums[1]
    assert resumed.next_rollout_seed == 43


@pytest.mark.parametrize(
    ("health", "thresholds", "reason"),
    [
        (PPOHealth(0, 0, 0.1, 0.0, 0.0, 0, 0), HealthThresholds(min_entropy=0.1), "entropy_collapse"),
        (PPOHealth(0, 0, 0.1, 1.0, 2.0, 0, 0), HealthThresholds(max_kl=1.0), "kl_explosion"),
        (PPOHealth(0, 0, 2.0, 1.0, 0.0, 0, 0), HealthThresholds(max_value_loss=1.0), "value_loss_divergence"),
        (PPOHealth(float("nan"), 0, 0.1, 1.0, 0.0, 0, 0), HealthThresholds(), "nonfinite_health"),
    ],
)
def test_health_alerts_write_diagnostic_checkpoint(tmp_path, monkeypatch, health, thresholds, reason) -> None:
    import rl.train_rl as train_rl
    belief, policy = _files(tmp_path)
    monkeypatch.setattr(train_rl, "ppo_update", lambda *_args: health)
    config = S5TrainingConfig(output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"}, health_thresholds=thresholds)
    with pytest.raises(RuntimeError, match=reason):
        run_s5_training(config, dependencies=S5TrainingDependencies(model_factory=_model_factory, rollout_factory=lambda *_: _steps(), arena_evaluator=_per_entry_evaluator, league_factory=_league_factory, curriculum_factory=_curriculum_factory))
    diagnostic = torch.load(config.output_dir / "checkpoints" / "diagnostic.pt", weights_only=False)
    assert diagnostic["metrics"]["alert"] == reason


def test_s3_regression_writes_diagnostic_checkpoint(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    config = S5TrainingConfig(output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"}, health_thresholds=HealthThresholds(min_s3_perfect_win_rate=0.6))
    def evaluator(_model, opponent, _runtime):
        return DualArenaMetrics(0.5, 0.7, 1000, 1000, 0, 0) if opponent.key == "s3" else DualArenaMetrics(0.7, 0.7, 1000, 1000, 0, 0)
    with pytest.raises(RuntimeError, match="s3_arena_regression"):
        run_s5_training(config, dependencies=S5TrainingDependencies(model_factory=_model_factory, rollout_factory=lambda *_: _steps(), arena_evaluator=evaluator, league_factory=_league_factory, curriculum_factory=_curriculum_factory))
    diagnostic = torch.load(config.output_dir / "checkpoints" / "diagnostic.pt", weights_only=False)
    assert diagnostic["metrics"]["alert"] == "s3_arena_regression"


def test_curriculum_does_not_advance_on_incomplete_or_invalid_arena(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    curriculum = _curriculum_factory()

    def invalid_evaluator(_model, opponent, _runtime):
        return DualArenaMetrics(0.9, 0.9, 999, 1000, 1 if opponent.key == "s3" else 0, 0)

    result = run_s5_training(
        S5TrainingConfig(
            output_dir=tmp_path / "out",
            frozen_s4_belief_path=belief,
            frozen_s4_policy_path=policy,
            frozen_s4_provenance={"release": "S4-v1"},
        ),
        dependencies=S5TrainingDependencies(
            model_factory=_model_factory,
            rollout_factory=lambda *_: _steps(),
            arena_evaluator=invalid_evaluator,
            league_factory=_league_factory,
            curriculum_factory=lambda: curriculum,
        ),
    )

    assert result.curriculum.stage_index == 0


def test_snapshot_admission_rejects_candidate_that_loses_to_history(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    def evaluator(_model, opponent, _runtime):
        if opponent.key == "cold-start":
            return DualArenaMetrics(0.49, 0.8, 1000, 1000, 0, 0)
        return DualArenaMetrics(0.8, 0.8, 1000, 1000, 0, 0)
    result = run_s5_training(
        S5TrainingConfig(output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"}, updates=2),
        dependencies=S5TrainingDependencies(model_factory=_model_factory, rollout_factory=lambda *_: _steps(), arena_evaluator=evaluator, league_factory=_league_factory, curriculum_factory=_curriculum_factory),
    )
    assert [entry.key for entry in result.league.historical_entries] == ["cold-start"]
    assert result.league.current_entry.snapshot.snapshot_id == "s5-1"


def test_immutable_snapshot_refuses_overwrite(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    config = S5TrainingConfig(output_dir=tmp_path / "out", frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"})
    deps = S5TrainingDependencies(model_factory=_model_factory, rollout_factory=lambda *_: _steps(), arena_evaluator=_per_entry_evaluator, league_factory=_league_factory, curriculum_factory=_curriculum_factory)
    run_s5_training(config, dependencies=deps)
    snapshot = config.output_dir / "checkpoints" / "snapshots" / "s5-step-1.pt"
    original = snapshot.read_bytes()
    with pytest.raises(FileExistsError, match="immutable snapshot"):
        run_s5_training(config, dependencies=deps)
    assert snapshot.read_bytes() == original


def test_resumed_kl_schedule_uses_restored_global_progress(tmp_path, monkeypatch) -> None:
    import rl.train_rl as train_rl
    belief, policy = _files(tmp_path)
    coefficients: list[float] = []
    original = train_rl.ppo_update
    def recording_update(model, batch, optimizer, ppo):
        coefficients.append(ppo.kl_coef)
        return original(model, batch, optimizer, ppo)
    monkeypatch.setattr(train_rl, "ppo_update", recording_update)
    common = dict(frozen_s4_belief_path=belief, frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"}, kl_start_coef=0.0, kl_end_coef=1.0, kl_schedule_total_updates=2)
    deps = S5TrainingDependencies(model_factory=_model_factory, rollout_factory=lambda *_: _steps(), arena_evaluator=_per_entry_evaluator, league_factory=_league_factory, curriculum_factory=_curriculum_factory)
    uninterrupted = run_s5_training(S5TrainingConfig(**common, output_dir=tmp_path / "uninterrupted", updates=2), dependencies=deps)
    expected = coefficients.copy()
    coefficients.clear()
    first = run_s5_training(S5TrainingConfig(**common, output_dir=tmp_path / "resumed", updates=1), dependencies=deps)
    run_s5_training(S5TrainingConfig(**common, output_dir=tmp_path / "resumed", updates=1, resume_checkpoint=first.checkpoint_path), dependencies=deps)
    assert uninterrupted.global_step == 2
    assert coefficients == expected == [0.0, 1.0]


def test_default_kl_schedule_horizon_is_preserved_across_split_resume(tmp_path, monkeypatch) -> None:
    """A one-update invocation must not restart the default KL decay on resume."""
    import rl.train_rl as train_rl

    belief, policy = _files(tmp_path)
    coefficients: list[float] = []
    original = train_rl.ppo_update

    def recording_update(model, batch, optimizer, ppo):
        coefficients.append(ppo.kl_coef)
        return original(model, batch, optimizer, ppo)

    monkeypatch.setattr(train_rl, "ppo_update", recording_update)
    common = dict(
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
        kl_start_coef=0.0,
        kl_end_coef=1.0,
    )
    deps = S5TrainingDependencies(
        model_factory=_model_factory,
        rollout_factory=lambda *_: _steps(),
        arena_evaluator=_per_entry_evaluator,
        league_factory=_league_factory,
        curriculum_factory=_curriculum_factory,
    )
    run_s5_training(S5TrainingConfig(**common, output_dir=tmp_path / "uninterrupted", updates=2), dependencies=deps)
    expected = coefficients.copy()
    coefficients.clear()
    first = run_s5_training(S5TrainingConfig(**common, output_dir=tmp_path / "split", updates=1), dependencies=deps)
    payload = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["config"]["resolved_kl_schedule_total_updates"] == 2
    run_s5_training(
        S5TrainingConfig(**common, output_dir=tmp_path / "split", updates=1, resume_checkpoint=first.checkpoint_path),
        dependencies=deps,
    )
    assert expected == coefficients == [0.0, 1.0]


def test_resume_rejects_explicit_kl_horizon_that_differs_from_checkpoint(tmp_path) -> None:
    belief, policy = _files(tmp_path)
    deps = S5TrainingDependencies(
        model_factory=_model_factory,
        rollout_factory=lambda *_: _steps(),
        arena_evaluator=_per_entry_evaluator,
        league_factory=_league_factory,
        curriculum_factory=_curriculum_factory,
    )
    common = dict(
        output_dir=tmp_path / "out",
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v1"},
        updates=1,
        kl_schedule_total_updates=2,
    )
    first = run_s5_training(S5TrainingConfig(**common), dependencies=deps)
    with pytest.raises(ValueError, match="KL schedule horizon"):
        run_s5_training(
            S5TrainingConfig(**(common | {"kl_schedule_total_updates": 3}), resume_checkpoint=first.checkpoint_path),
            dependencies=deps,
        )


def test_resumed_report_and_latest_checkpoint_retain_prior_health_rows(tmp_path) -> None:
    """A resumed run must append health evidence instead of hiding the first run."""
    belief, policy = _files(tmp_path)
    deps = S5TrainingDependencies(
        model_factory=_model_factory,
        rollout_factory=lambda *_: _steps(),
        arena_evaluator=_per_entry_evaluator,
        league_factory=_league_factory,
        curriculum_factory=_curriculum_factory,
    )
    common = dict(
        output_dir=tmp_path / "out", frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"},
    )
    first = run_s5_training(S5TrainingConfig(**common, updates=2), dependencies=deps)
    resumed = run_s5_training(
        S5TrainingConfig(**common, updates=1, resume_checkpoint=first.checkpoint_path),
        dependencies=deps,
    )
    assert len(resumed.report["health"]) == 3
    latest_payload = torch.load(resumed.checkpoint_path, weights_only=False)
    assert len(latest_payload["metrics"]["health"]) == 3


def _fail_latest_checkpoint(path, *args, **kwargs):
    if Path(path).name == "latest.pt":
        raise RuntimeError("latest_checkpoint boom")
    from rl.checkpoints import save_checkpoint
    return save_checkpoint(path, *args, **kwargs)


@pytest.mark.parametrize(
    ("stage", "inject_failure"),
    [
        ("rollout", lambda monkeypatch: None),
        ("batch", lambda monkeypatch: monkeypatch.setattr("rl.train_rl._batch_from_steps", lambda *_args: (_ for _ in ()).throw(RuntimeError("batch boom")))),
        ("evaluation", lambda monkeypatch: monkeypatch.setattr("rl.train_rl._evaluate", lambda *_args: (_ for _ in ()).throw(RuntimeError("evaluation boom")))),
        ("snapshot", lambda monkeypatch: monkeypatch.setattr("rl.train_rl._write_immutable_snapshot", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("snapshot boom")))),
        ("league", lambda monkeypatch: monkeypatch.setattr(OpponentLeague, "promote_candidate", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("league boom")))),
        ("curriculum", lambda monkeypatch: monkeypatch.setattr(ObservationCurriculum, "advance", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("curriculum boom")))),
        ("latest_checkpoint", lambda monkeypatch: monkeypatch.setattr("rl.train_rl.save_checkpoint", _fail_latest_checkpoint)),
        ("report", lambda monkeypatch: monkeypatch.setattr("rl.train_rl._write_report", lambda *_args: (_ for _ in ()).throw(RuntimeError("report boom")))),
    ],
)
def test_every_training_stage_failure_writes_diagnostic_checkpoint(tmp_path, monkeypatch, stage, inject_failure) -> None:
    """Each orchestration boundary leaves a resume/debug payload with its stage."""
    belief, policy = _files(tmp_path)
    config = S5TrainingConfig(
        output_dir=tmp_path / stage, frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy, frozen_s4_provenance={"release": "S4-v1"},
    )
    inject_failure(monkeypatch)

    def rollout(*_args):
        if stage == "rollout":
            raise RuntimeError("rollout boom")
        return _steps()

    with pytest.raises(RuntimeError, match=f"{stage} boom"):
        run_s5_training(
            config,
            dependencies=S5TrainingDependencies(
                model_factory=_model_factory, rollout_factory=rollout,
                arena_evaluator=_per_entry_evaluator, league_factory=_league_factory,
                curriculum_factory=_curriculum_factory,
            ),
        )
    payload = torch.load(config.output_dir / "checkpoints" / "diagnostic.pt", weights_only=False)
    assert payload["metrics"]["stage"] == stage
    assert payload["metrics"]["reason"] == f"{stage}_exception"


def test_report_publication_manifest_rejects_mismatched_pair_and_fsyncs_parent(tmp_path, monkeypatch) -> None:
    """Readers accept a report pair only after its final manifest is durable."""
    import rl.train_rl as train_rl

    parent_syncs: list[Path] = []
    file_syncs: list[int] = []
    monkeypatch.setattr(train_rl, "_fsync_parent", lambda path: parent_syncs.append(Path(path)))
    monkeypatch.setattr(train_rl.os, "fsync", lambda descriptor: file_syncs.append(descriptor))
    report = {
        "global_step": 1, "device": "cpu", "health": [],
        "arena": {"by_opponent": {}, "s3_comparison": {}},
    }
    json_path, markdown_path = tmp_path / "report.json", tmp_path / "report.md"
    manifest_path = train_rl._write_report(json_path, markdown_path, report)
    assert parent_syncs == [tmp_path, tmp_path, tmp_path]
    assert len(file_syncs) == 3
    assert train_rl.load_published_s5_report(json_path) == report
    markdown_path.write_text("mismatched", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        train_rl.load_published_s5_report(json_path)
    assert manifest_path.exists()
