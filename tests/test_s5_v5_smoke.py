from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from rl.checkpoints import load_checkpoint
from tools.cloud_train_s5 import run_local_v5_prep_smoke


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory):
    return run_local_v5_prep_smoke(
        output_dir=tmp_path_factory.mktemp("s5-v5-prep") / "prep",
        games=50,
        updates=10,
        seed=20260718,
        project_root=ROOT,
    )


def test_v5_local_smoke_has_fifty_completed_zero_illegal_rollouts_and_ten_finite_updates(result) -> None:
    assert result.completed_games == 50
    assert result.illegal_actions == 0
    assert result.zero_sum_failures == 0
    assert result.trajectory_steps > 0
    assert result.ppo_updates == 10
    assert all(math.isfinite(value) for value in result.losses)
    assert all(math.isfinite(value) for value in result.entropies)
    assert all(math.isfinite(value) for value in result.kls)


def test_v5_local_smoke_resumes_and_keeps_league_and_observation_isolation(result) -> None:
    assert result.resume_updates == 5
    assert result.resume_curve_matches
    assert result.v5_snapshot_in_league
    assert result.effective_sampling_weights
    assert result.opponents_used_perfect_observation
    assert result.learner_used_curriculum_degradation
    assert result.benchmark_path.is_file()
    assert result.evidence_path.is_file()
    assert all("decision" not in path.name.lower() for path in result.artifact_paths)


def test_v5_local_smoke_evidence_proves_stateful_resume_equivalence(result) -> None:
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))

    comparison = evidence["resume_equivalence"]
    assert comparison["uninterrupted_updates"] == 15
    assert comparison["split_updates_before_checkpoint"] == 10
    assert comparison["split_updates_after_resume"] == 5
    assert comparison["same_initial_model_state"]
    assert comparison["same_reference_policy_state"]
    assert comparison["same_update_inputs"]
    assert comparison["model_state_matches"]
    assert comparison["optimizer_state_matches"]
    assert comparison["strict_tolerance"] == 0.0
    assert evidence["resume_curve_matches"] == all(
        comparison[name]
        for name in (
            "same_initial_model_state",
            "same_reference_policy_state",
            "same_update_inputs",
            "model_state_matches",
            "optimizer_state_matches",
        )
    )


def test_v5_local_smoke_checkpoint_rejects_tampered_payload(result, tmp_path: Path) -> None:
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    del payload["optimizer_state_dict"]
    tampered = tmp_path / "tampered-v5-prep.pt"
    torch.save(payload, tampered)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="checkpoint missing field: optimizer_state_dict"):
        load_checkpoint(tampered, model=model, optimizer=optimizer)
