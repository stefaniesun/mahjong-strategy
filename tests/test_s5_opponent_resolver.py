from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from policies.opponent_pool import GreedyPolicy, RandomPolicy
from policies.rule_policy import RulePolicy
from rl.league import OpponentEntry, OpponentKind, SnapshotMetadata
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from state.encoder import encoding_table


MODEL_CONFIG = PolicyValueNetConfig(
    input_size=encoding_table()[-1]["end"],
    action_size=637,
    hidden_size=8,
    residual_blocks=0,
)


class IdentityBeliefProvider:
    def apply(self, state):
        return state


def _model_factory() -> PolicyValueNet:
    return PolicyValueNet(MODEL_CONFIG)


def _checkpoint(tmp_path: Path, *, fill: float = 0.25) -> tuple[Path, str]:
    model = _model_factory()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
    path = tmp_path / f"policy-{fill}.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _model_entry(kind: OpponentKind, path: Path, checksum: str | None) -> OpponentEntry:
    snapshot = SnapshotMetadata(
        snapshot_id=f"{kind.value}-policy",
        policy_version="s5-test",
        checkpoint_path=str(path),
        training_step=7,
        checksum=checksum,
    )
    key = "current" if kind is OpponentKind.CURRENT else snapshot.snapshot_id
    return OpponentEntry(key, kind, snapshot)


def _resolver():
    from rl.opponent_resolver import OpponentResolver

    return OpponentResolver(
        model_factory=_model_factory,
        belief_provider=IdentityBeliefProvider(),
    )


@pytest.mark.parametrize(
    ("entry", "expected_type"),
    (
        (OpponentEntry("s3", OpponentKind.S3), RulePolicy),
        (OpponentEntry("greedy", OpponentKind.GREEDY), GreedyPolicy),
        (OpponentEntry("random", OpponentKind.RANDOM), RandomPolicy),
    ),
)
def test_resolver_constructs_fixed_league_opponents(entry, expected_type) -> None:
    assert isinstance(_resolver().resolve(entry, seed=17), expected_type)


@pytest.mark.parametrize("kind", (OpponentKind.CURRENT, OpponentKind.SNAPSHOT))
def test_resolver_loads_model_entries_from_metadata_checkpoint_and_freezes_them(
    tmp_path: Path,
    kind: OpponentKind,
) -> None:
    from rl.opponent_resolver import ModelPolicy

    path, checksum = _checkpoint(tmp_path, fill=0.375)

    policy = _resolver().resolve(_model_entry(kind, path, checksum), seed=19)

    assert isinstance(policy, ModelPolicy)
    assert policy.model.training is False
    assert {parameter.device.type for parameter in policy.model.parameters()} == {"cpu"}
    assert all(parameter.requires_grad is False for parameter in policy.model.parameters())
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.375))
        for parameter in policy.model.parameters()
    )


def test_model_policy_restores_cpu_eval_frozen_invariants_during_inference(tmp_path: Path) -> None:
    from engine.game import Game
    from state.action_space import legal_mask
    from state.adapters.from_engine import from_engine

    path, checksum = _checkpoint(tmp_path)
    policy = _resolver().resolve(
        _model_entry(OpponentKind.CURRENT, path, checksum),
        seed=3,
    )
    policy.model.train()
    for parameter in policy.model.parameters():
        parameter.requires_grad_(True)
    state = from_engine(Game(seed=5).reset(), 0)
    mask = legal_mask(state)

    action = policy.choose_action(state, mask)

    assert policy.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in policy.model.parameters())
    assert all(parameter.requires_grad is False for parameter in policy.model.parameters())
    assert action is not None


def test_current_resolution_does_not_reference_an_in_memory_candidate(tmp_path: Path) -> None:

    path, checksum = _checkpoint(tmp_path, fill=0.125)
    candidate = _model_factory()
    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.fill_(0.875)

    policy = _resolver().resolve(_model_entry(OpponentKind.CURRENT, path, checksum), seed=3)

    assert policy.model is not candidate
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.125))
        for parameter in policy.model.parameters()
    )


def test_resolver_rejects_missing_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="league snapshot is unavailable"):
        _resolver().resolve(
            _model_entry(OpponentKind.CURRENT, missing, None),
            seed=0,
        )


def test_resolver_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path, _ = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        _resolver().resolve(
            _model_entry(OpponentKind.SNAPSHOT, path, "0" * 64),
            seed=0,
        )


def test_resolver_requires_checksum_for_model_opponents(tmp_path: Path) -> None:
    path, _ = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="checksum is required"):
        _resolver().resolve(
            _model_entry(OpponentKind.CURRENT, path, None),
            seed=0,
        )


def test_resolver_loads_frozen_s4_policy_state_without_value_head(tmp_path: Path) -> None:

    source = _model_factory()
    state = {
        key: value
        for key, value in source.state_dict().items()
        if key.startswith("trunk.") or key.startswith("action_head.")
    }
    path = tmp_path / "s4-policy.pt"
    torch.save({"state_dict": state}, path)

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = _resolver().resolve(
        _model_entry(OpponentKind.CURRENT, path, checksum),
        seed=0,
    )


    assert all(
        torch.equal(policy.model.state_dict()[key], value)
        for key, value in state.items()
    )


def test_resolver_rejects_checkpoint_without_model_state(tmp_path: Path) -> None:
    path = tmp_path / "missing-state.pt"
    torch.save({"format_version": 1}, path)

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no safe model state"):
        _resolver().resolve(
            _model_entry(OpponentKind.CURRENT, path, checksum),
            seed=0,
        )



def test_random_resolution_is_reproducible_for_seed() -> None:

    entry = OpponentEntry("random", OpponentKind.RANDOM)
    first = _resolver().resolve(entry, seed=23)
    second = _resolver().resolve(entry, seed=23)
    legal_mask = [False] * 637
    legal_mask[0] = True
    legal_mask[1] = True

    first_actions = [first.choose_action(None, legal_mask) for _ in range(8)]
    second_actions = [second.choose_action(None, legal_mask) for _ in range(8)]

    assert first_actions == second_actions
