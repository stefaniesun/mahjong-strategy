from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl.models.value_net import PolicyValueNetConfig
from rl.rollout import FrozenBeliefProvider
from rl.train_rl import S5TrainingConfig, _default_model, _frozen_reference_policy, _scheduled_ppo
from state.encoder import ENCODER_VERSION, encoding_size, encoding_table


ROOT = Path(__file__).resolve().parents[1]
V5_CHECKPOINTS = ROOT / "training_artifacts" / "S4" / "v5_20260718_encoder_v4" / "checkpoints"
V5_POLICY = V5_CHECKPOINTS / "policy_s4.pt"
V5_BELIEF = V5_CHECKPOINTS / "belief_s4.pt"


def _v5_config(tmp_path: Path, *, policy: Path = V5_POLICY) -> S5TrainingConfig:
    return S5TrainingConfig(
        output_dir=tmp_path / "output",
        frozen_s4_belief_path=V5_BELIEF,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4-v5", "encoder_version": ENCODER_VERSION},
    )


def test_current_encoder_contract_is_structural_and_v5_is_893_dimensional() -> None:
    assert encoding_size() == sum(section["size"] for section in encoding_table())
    assert encoding_size() == 893


def test_default_s5_policy_loads_v5_weights_for_current_encoder(tmp_path: Path) -> None:
    model = _default_model(_v5_config(tmp_path))

    assert model.config == PolicyValueNetConfig(
        input_size=encoding_size(),
        action_size=637,
        hidden_size=256,
        residual_blocks=3,
        dropout=0.05,
    )
    payload = torch.load(V5_POLICY, map_location="cpu", weights_only=True)
    assert torch.equal(model.trunk[0].weight, payload["state_dict"]["trunk.0.weight"])


def test_default_s5_policy_rejects_checkpoint_with_stale_encoder_version(tmp_path: Path) -> None:
    payload = torch.load(V5_POLICY, map_location="cpu", weights_only=True)
    stale = deepcopy(payload)
    stale["encoder_version"] = "s2.v4.encoder.v3"
    policy = tmp_path / "stale-policy.pt"
    torch.save(stale, policy)

    with pytest.raises(ValueError, match="encoder version"):
        _default_model(_v5_config(tmp_path, policy=policy))


def test_default_s5_policy_rejects_input_size_that_disagrees_with_current_encoder(tmp_path: Path) -> None:
    payload = torch.load(V5_POLICY, map_location="cpu", weights_only=True)
    incompatible = deepcopy(payload)
    incompatible["model_config"] = dict(incompatible["model_config"], input_size=encoding_size() + 1)
    policy = tmp_path / "wrong-width-policy.pt"
    torch.save(incompatible, policy)

    with pytest.raises(ValueError, match="input_size.*current encoder"):
        _default_model(_v5_config(tmp_path, policy=policy))


def test_v5_learned_belief_provider_is_frozen(tmp_path: Path) -> None:
    provider = FrozenBeliefProvider.from_checkpoint(str(V5_BELIEF))

    assert provider.belief.model.config.input_size == encoding_size() == 893
    assert provider.belief.model.training is False
    assert all(not parameter.requires_grad for parameter in provider.belief.model.parameters())


def test_frozen_belief_provider_rejects_current_version_checkpoint_with_wrong_input_size(tmp_path: Path) -> None:
    payload = torch.load(V5_BELIEF, map_location="cpu", weights_only=True)
    incompatible = deepcopy(payload)
    incompatible["model_config"] = dict(incompatible["model_config"], input_size=encoding_size() + 1)
    checkpoint = tmp_path / "wrong-width-belief.pt"
    torch.save(incompatible, checkpoint)

    with pytest.raises(ValueError, match="belief checkpoint input_size=.*current encoder"):
        FrozenBeliefProvider.from_checkpoint(str(checkpoint))


def test_v5_kl_reference_is_frozen_copy_and_its_coefficient_decays(tmp_path: Path) -> None:
    config = replace(
        _v5_config(tmp_path),
        updates=3,
        kl_start_coef=0.6,
        kl_end_coef=0.0,
    )
    learner = _default_model(config)
    reference = _frozen_reference_policy(learner)

    assert reference is not learner
    assert reference.training is False
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert all(
        torch.equal(reference.state_dict()[name], learner.state_dict()[name])
        for name in learner.state_dict()
    )
    learner.trunk[0].weight.data.add_(1.0)
    assert not torch.equal(reference.trunk[0].weight, learner.trunk[0].weight)
    assert [
        _scheduled_ppo(config, step, schedule_total_updates=3).kl_coef
        for step in range(3)
    ] == pytest.approx([0.6, 0.3, 0.0])
