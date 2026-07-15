import math

import pytest
import torch

import rl.ppo_trainer as ppo_trainer
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from rl.ppo_trainer import PPOBatch, PPOConfig, PPOHealth, compute_gae, ppo_update


def _model() -> PolicyValueNet:
    torch.manual_seed(7)
    return PolicyValueNet(
        PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=1)
    )


def _batch() -> PPOBatch:
    return PPOBatch(
        features=torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]]),
        legal_mask=torch.tensor([[True, True, False, False], [False, True, True, False]]),
        actions=torch.tensor([0, 2]),
        old_log_probs=torch.tensor([-0.7, -0.6]),
        old_values=torch.tensor([0.1, -0.2]),
        rewards=torch.tensor([1.0, -0.5]),
        dones=torch.tensor([False, True]),
        reference_logits=torch.tensor([[0.3, -0.1, 4.0, -3.0], [7.0, 0.2, 0.4, -2.0]]),
    )


def test_compute_gae_matches_known_three_step_example() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        values=torch.tensor([0.5, 1.0, 1.5]),
        dones=torch.tensor([False, False, True]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert torch.allclose(advantages, torch.tensor([3.8696, 3.4300, 1.5000]), atol=1e-6)
    assert torch.allclose(returns, torch.tensor([4.3696, 4.4300, 3.0000]), atol=1e-6)


def test_compute_gae_stops_bootstrap_at_terminal_boundary() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 10.0, 1.0]),
        values=torch.tensor([0.0, 2.0, 0.0]),
        dones=torch.tensor([True, False, True]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([1.0, 8.9, 1.0]))
    assert torch.allclose(returns, torch.tensor([1.0, 10.9, 1.0]))


def test_ppo_config_rejects_nonfinite_hyperparameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        PPOConfig(gamma=float("nan"))


def test_ppo_update_rejects_rows_without_legal_actions() -> None:
    batch = _batch()
    batch.legal_mask[1] = False

    with pytest.raises(ValueError, match="at least one legal action"):
        ppo_update(_model(), batch, torch.optim.Adam(_model().parameters()), PPOConfig())


def test_ppo_update_rejects_nonfinite_numeric_legal_mask() -> None:
    batch = _batch()
    batch.legal_mask = batch.legal_mask.to(torch.float32)
    batch.legal_mask[0, 0] = torch.nan
    model = _model()

    with pytest.raises(ValueError, match="legal_mask.*finite"):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())


def test_ppo_update_rejects_nonbinary_numeric_legal_mask() -> None:
    batch = _batch()
    batch.legal_mask = batch.legal_mask.to(torch.float32)
    batch.legal_mask[0, 0] = 2.0
    model = _model()

    with pytest.raises(ValueError, match="legal_mask.*zero or one"):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())


def test_ppo_update_rejects_in_range_action_marked_illegal() -> None:
    batch = _batch()
    batch.legal_mask[0, 0] = False
    model = _model()

    with pytest.raises(ValueError, match="actions must be legal"):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())


@pytest.mark.parametrize("field", ["old_log_probs", "old_values", "rewards", "dones"])
def test_ppo_update_rejects_trajectory_tensors_with_model_batch_mismatch(field: str) -> None:
    batch = _batch()
    value = getattr(batch, field)
    setattr(batch, field, torch.cat((value, value[-1:])))
    model = _model()

    with pytest.raises(ValueError, match=rf"{field} must have shape \[batch\]"):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("legal_mask", "legal_mask batch size must match features"),
        ("actions", r"actions must have shape \[batch\]"),
        ("reference_logits", "reference_logits shape must match legal_mask"),
    ],
)
def test_ppo_update_rejects_other_model_batch_mismatches(field: str, message: str) -> None:
    batch = _batch()
    value = getattr(batch, field)
    setattr(batch, field, torch.cat((value, value[:1])))
    model = _model()

    with pytest.raises(ValueError, match=message):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())


def test_ppo_update_does_not_backpropagate_into_historical_rollout_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    batch.features = batch.features.detach().requires_grad_(True)
    batch.old_log_probs = batch.old_log_probs.detach().requires_grad_(True)
    batch.old_values = batch.old_values.detach().requires_grad_(True)
    batch.rewards = batch.rewards.detach().requires_grad_(True)
    batch.reference_logits = batch.reference_logits.detach().requires_grad_(True)
    model = _model()
    received_by_gae: list[torch.Tensor] = []
    original_compute_gae = ppo_trainer.compute_gae

    def spy_compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        received_by_gae.extend((rewards, values, dones))
        return original_compute_gae(rewards, values, dones, gamma, gae_lambda)

    monkeypatch.setattr(ppo_trainer, "compute_gae", spy_compute_gae)

    ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())

    assert len(received_by_gae) == 3
    assert all(not tensor.requires_grad for tensor in received_by_gae)
    assert batch.features.grad is None
    assert batch.old_log_probs.grad is None
    assert batch.old_values.grad is None
    assert batch.rewards.grad is None
    assert batch.reference_logits.grad is None


def test_ppo_update_accepts_uint8_action_indices() -> None:
    batch = _batch()
    batch.actions = batch.actions.to(torch.uint8)
    model = _model()

    health = ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())

    assert isinstance(health, PPOHealth)


def test_ppo_update_uses_clipped_ratio_for_selected_action_log_probs() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_head.bias.copy_(torch.tensor([math.log(0.2), math.log(0.8), 0.0, 0.0]))
    batch = PPOBatch(
        features=torch.zeros(2, 3),
        legal_mask=torch.tensor([[True, True, False, False], [True, True, False, False]]),
        actions=torch.tensor([0, 1]),
        old_log_probs=torch.tensor([math.log(0.5), math.log(0.5)]),
        old_values=torch.zeros(2),
        rewards=torch.tensor([0.0, 1.0]),
        dones=torch.tensor([True, True]),
        reference_logits=torch.zeros(2, 4),
    )

    metrics = ppo_update(model, batch, torch.optim.SGD(model.parameters(), lr=0.0), PPOConfig())

    assert metrics.policy_loss == pytest.approx(-0.2, abs=1e-6)
    assert metrics.clip_fraction == pytest.approx(1.0)


def test_ppo_update_ignores_illegal_reference_logits_in_kl() -> None:
    model = _model()
    first_batch = _batch()
    second_batch = _batch()
    first_batch.reference_logits[~first_batch.legal_mask] = torch.tensor([100.0, -100.0, 50.0, -50.0])
    second_batch.reference_logits[~second_batch.legal_mask] = torch.tensor([-50.0, 50.0, -100.0, 100.0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    first_metrics = ppo_update(model, first_batch, optimizer, PPOConfig())
    second_metrics = ppo_update(model, second_batch, optimizer, PPOConfig())

    assert first_metrics.kl == second_metrics.kl


def test_ppo_update_runs_one_optimizer_step_with_finite_metrics() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_weight = model.action_head.weight.detach().clone()

    metrics = ppo_update(model, _batch(), optimizer, PPOConfig())

    assert isinstance(metrics, PPOHealth)
    assert all(
        math.isfinite(value)
        for value in (
            metrics.total_loss,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.kl,
            metrics.clip_fraction,
            metrics.grad_norm,
        )
    )
    assert not torch.equal(model.action_head.weight.detach(), initial_weight)


def test_ppo_update_masks_illegal_logits_from_policy_probability() -> None:
    model = _model()
    batch = _batch()

    ppo_update(model, batch, torch.optim.SGD(model.parameters(), lr=0.0), PPOConfig())
    probabilities = model(batch.features, batch.legal_mask).action_probs

    assert torch.equal(probabilities[~batch.legal_mask], torch.zeros_like(probabilities[~batch.legal_mask]))


def test_ppo_update_rejects_nan_rollout_values() -> None:
    batch = _batch()
    batch.rewards[0] = torch.nan
    model = _model()

    with pytest.raises(ValueError, match="finite"):
        ppo_update(model, batch, torch.optim.Adam(model.parameters()), PPOConfig())
