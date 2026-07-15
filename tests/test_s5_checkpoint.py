from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from rl.checkpoints import load_checkpoint, save_checkpoint
from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
from rl.league import OpponentLeague, SnapshotMetadata
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig


class _UnsafePayload:
    pass


def _league() -> OpponentLeague:
    return OpponentLeague(current_policy=SnapshotMetadata("cold-start", "v0", "current.pt", 0))


def _curriculum() -> ObservationCurriculum:
    return ObservationCurriculum(CurriculumConfig((
        CurriculumStage("perfect", DegradationProfile.perfect(), 0.0, 0.0),
        CurriculumStage("noise", DegradationProfile("noise", 0.1), 0.0, 0.0),
    )))


def test_checkpoint_restores_model_optimizer_rng_and_orchestration_state(tmp_path) -> None:
    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(torch.ones(1, 3), torch.ones(1, 4, dtype=torch.bool)).values.sum().backward()
    optimizer.step()
    random.seed(17)
    torch.manual_seed(23)
    path = tmp_path / "state.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        global_step=7,
        next_rollout_seed=1234,
        league=_league(),
        curriculum=_curriculum(),
        config={"updates": 2},
        metrics={"loss": 1.0},
        frozen_s4_provenance={"belief": {"path": "belief.pt", "sha256": "abc"}},
    )
    expected_python = random.random()
    expected_torch = torch.rand(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    restored = load_checkpoint(path, model=model, optimizer=optimizer, restore_rng=True)
    assert restored.global_step == 7
    assert restored.next_rollout_seed == 1234
    assert restored.league.to_json() == _league().to_json()
    assert restored.curriculum.to_json() == _curriculum().to_json()
    assert random.random() == expected_python
    assert torch.equal(torch.rand(1), expected_torch)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_rejects_empty_frozen_s4_provenance_on_save_and_load(tmp_path) -> None:
    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    common = dict(
        model=model, optimizer=optimizer, global_step=0, next_rollout_seed=0,
        league=_league(), curriculum=_curriculum(), config={}, metrics={},
    )
    with pytest.raises(ValueError, match="frozen_s4_provenance"):
        save_checkpoint(tmp_path / "empty.pt", frozen_s4_provenance={}, **common)

    path = tmp_path / "valid.pt"
    save_checkpoint(path, frozen_s4_provenance={"release": "S4-v1"}, **common)
    payload = torch.load(path, weights_only=False)
    payload["frozen_s4_provenance"] = {}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="frozen_s4_provenance"):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_rejects_unsafe_pickle_before_restoring_model(tmp_path) -> None:
    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "unsafe.pt"
    torch.save(_UnsafePayload(), path)
    with pytest.raises(ValueError, match="safe S5 checkpoint"):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_rejects_non_json_s4_provenance_before_writing(tmp_path) -> None:
    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(TypeError, match="JSON serializable"):
        save_checkpoint(
            tmp_path / "bad.pt", model=model, optimizer=optimizer, global_step=0, next_rollout_seed=0,
            league=_league(), curriculum=_curriculum(), config={}, metrics={},
            frozen_s4_provenance={"not_json": object()},
        )


def test_checkpoint_replace_failure_preserves_existing_canonical_file(tmp_path, monkeypatch) -> None:
    import rl.checkpoints as checkpoints

    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "state.pt"
    common = dict(model=model, optimizer=optimizer, global_step=0, next_rollout_seed=0, league=_league(), curriculum=_curriculum(), config={}, metrics={}, frozen_s4_provenance={"release": "S4-v1"})
    save_checkpoint(path, **common)
    before = path.read_bytes()
    monkeypatch.setattr(checkpoints.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(OSError, match="rename failed"):
        save_checkpoint(path, **common)
    assert path.read_bytes() == before


def test_checkpoint_fsyncs_temp_file_and_parent_before_returning(tmp_path, monkeypatch) -> None:
    """The payload and its directory are both made durable around replacement."""
    import rl.checkpoints as checkpoints

    model = PolicyValueNet(PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    file_syncs: list[int] = []
    parent_syncs: list[Path] = []
    monkeypatch.setattr(checkpoints.os, "fsync", lambda descriptor: file_syncs.append(descriptor))
    monkeypatch.setattr(checkpoints, "_fsync_parent", lambda directory: parent_syncs.append(Path(directory)))
    save_checkpoint(
        tmp_path / "durable.pt", model=model, optimizer=optimizer, global_step=0, next_rollout_seed=0,
        league=_league(), curriculum=_curriculum(), config={}, metrics={}, frozen_s4_provenance={"release": "S4-v1"},
    )
    assert len(file_syncs) == 1
    assert parent_syncs == [tmp_path]
