import pytest
import torch

from learning.models.policy_net import PolicyNet, PolicyNetConfig
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig


def _config() -> PolicyValueNetConfig:
    return PolicyValueNetConfig(input_size=6, action_size=4, hidden_size=8, residual_blocks=1)


def test_forward_returns_action_logits_and_batch_values() -> None:
    model = PolicyValueNet(_config())

    output = model(torch.randn(3, 6))

    assert output.action_logits.shape == (3, 4)
    assert output.values.shape == (3,)


def test_forward_preserves_float64_model_dtype() -> None:
    model = PolicyValueNet(_config()).double()
    features = torch.randn(3, 6, dtype=torch.float64)

    output = model(features)

    assert output.action_logits.dtype is torch.float64
    assert output.values.dtype is torch.float64


def test_forward_masks_illegal_actions_from_probabilities() -> None:
    model = PolicyValueNet(_config())
    features = torch.randn(2, 6)
    legal_mask = torch.tensor([[True, False, True, False], [False, True, True, False]])

    output = model(features, legal_mask)

    assert torch.equal(
        output.action_logits[~legal_mask],
        torch.full_like(output.action_logits[~legal_mask], torch.finfo(output.action_logits.dtype).min),
    )
    assert torch.equal(output.action_probs[~legal_mask], torch.zeros_like(output.action_probs[~legal_mask]))


def test_forward_rejects_invalid_legal_masks() -> None:
    model = PolicyValueNet(_config())
    features = torch.randn(2, 6)

    with pytest.raises(ValueError, match="shape must match"):
        model(features, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one legal action"):
        model(features, torch.tensor([[True, False, False, False], [False, False, False, False]]))


def test_load_s4_policy_state_dict_preserves_value_head() -> None:
    config = _config()
    s4_policy = PolicyNet(
        PolicyNetConfig(
            input_size=config.input_size,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            residual_blocks=config.residual_blocks,
            dropout=config.dropout,
        )
    )
    model = PolicyValueNet(config)
    original_value_head = {key: value.detach().clone() for key, value in model.value_head.state_dict().items()}

    model.load_s4_policy_state_dict(s4_policy.state_dict())

    assert model.trunk.state_dict().keys() == s4_policy.trunk.state_dict().keys()
    assert model.action_head.state_dict().keys() == s4_policy.action_head.state_dict().keys()
    for key, value in s4_policy.trunk.state_dict().items():
        assert torch.equal(model.trunk.state_dict()[key], value)
    for key, value in s4_policy.action_head.state_dict().items():
        assert torch.equal(model.action_head.state_dict()[key], value)
    for key, value in original_value_head.items():
        assert torch.equal(model.value_head.state_dict()[key], value)


def test_load_s4_policy_state_dict_rejects_missing_or_mismatched_weights() -> None:
    config = _config()
    model = PolicyValueNet(config)
    source = PolicyNet(PolicyNetConfig(input_size=6, action_size=4, hidden_size=8, residual_blocks=1))
    missing = source.state_dict()
    missing.pop("action_head.bias")

    with pytest.raises(ValueError, match="missing"):
        model.load_s4_policy_state_dict(missing)

    mismatched = source.state_dict()
    mismatched["action_head.weight"] = torch.randn(5, 8)
    with pytest.raises(ValueError, match="shape"):
        model.load_s4_policy_state_dict(mismatched)
