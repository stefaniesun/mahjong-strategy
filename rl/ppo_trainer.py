from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from rl.models.value_net import PolicyValueNet


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "gamma",
            "gae_lambda",
            "clip_ratio",
            "value_coef",
            "entropy_coef",
            "kl_coef",
            "max_grad_norm",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be between 0 and 1")
        for name in ("value_coef", "entropy_coef", "kl_coef"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(frozen=True)
class PPOHealth:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    clip_fraction: float
    grad_norm: float


@dataclass
class PPOBatch:
    features: torch.Tensor
    legal_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    reference_logits: torch.Tensor


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite floating-point values")


def _validate_legal_mask(legal_mask: torch.Tensor) -> None:
    if legal_mask.dtype == torch.bool:
        return
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if legal_mask.dtype not in integer_dtypes and not torch.is_floating_point(legal_mask):
        raise ValueError("legal_mask must be boolean or finite zero-one numeric values")
    if torch.is_floating_point(legal_mask) and not torch.isfinite(legal_mask).all():
        raise ValueError("legal_mask numeric values must be finite")
    if not torch.all((legal_mask == 0) | (legal_mask == 1)):
        raise ValueError("legal_mask numeric values must be zero or one")


def _validate_trajectory_inputs(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
) -> None:
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be one-dimensional")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching shapes")
    if rewards.numel() == 0:
        raise ValueError("trajectory tensors must not be empty")
    _validate_finite("rewards", rewards)
    _validate_finite("values", values)
    if dones.dtype != torch.bool:
        if not torch.is_floating_point(dones) or not torch.isfinite(dones).all():
            raise ValueError("dones must be boolean or finite zero-one values")
        if not torch.all((dones == 0) | (dones == 1)):
            raise ValueError("dones must be boolean or finite zero-one values")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantages with a zero bootstrap after the final step."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be between 0 and 1")
    _validate_trajectory_inputs(rewards, values, dones)

    rewards = rewards.to(dtype=values.dtype, device=values.device)
    done_flags = dones.to(dtype=torch.bool, device=values.device)
    advantages = torch.zeros_like(values)
    next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    terminal_value = torch.zeros((), dtype=values.dtype, device=values.device)

    for step in range(values.numel() - 1, -1, -1):
        next_value = values[step + 1] if step + 1 < values.numel() else terminal_value
        not_done = (~done_flags[step]).to(dtype=values.dtype)
        delta = rewards[step] + gamma * next_value * not_done - values[step]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[step] = next_advantage
    return advantages, advantages + values


def _validate_batch(batch: PPOBatch) -> None:
    if batch.features.ndim != 2:
        raise ValueError("features must have shape [batch, feature]")
    if batch.legal_mask.ndim != 2:
        raise ValueError("legal_mask must have shape [batch, action]")
    _validate_legal_mask(batch.legal_mask)
    batch_size = batch.features.shape[0]
    if batch_size == 0:
        raise ValueError("batch must not be empty")
    if batch.legal_mask.shape[0] != batch_size:
        raise ValueError("legal_mask batch size must match features")
    if not batch.legal_mask.bool().any(dim=-1).all():
        raise ValueError("each sample must have at least one legal action")
    if batch.actions.ndim != 1 or batch.actions.shape[0] != batch_size:
        raise ValueError("actions must have shape [batch]")
    if batch.actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("actions must be integer indices")
    action_indices = batch.actions.to(dtype=torch.long)
    if (action_indices < 0).any() or (action_indices >= batch.legal_mask.shape[1]).any():
        raise ValueError("actions must be valid action indices")
    action_rows = torch.arange(batch_size, device=batch.actions.device)
    if not batch.legal_mask.to(device=batch.actions.device, dtype=torch.bool)[action_rows, action_indices].all():
        raise ValueError("actions must be legal under legal_mask")

    for name in ("features", "old_log_probs", "old_values", "rewards", "reference_logits"):
        _validate_finite(name, getattr(batch, name))
    if batch.reference_logits.shape != batch.legal_mask.shape:
        raise ValueError("reference_logits shape must match legal_mask")
    for name in ("old_log_probs", "old_values", "rewards", "dones"):
        value = getattr(batch, name)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch]")
    _validate_trajectory_inputs(batch.rewards, batch.old_values, batch.dones)


def ppo_update(
    model: PolicyValueNet,
    batch: PPOBatch,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
) -> PPOHealth:
    """Perform one PPO minibatch update against detached S4 reference logits."""
    _validate_batch(batch)

    device = next(model.parameters()).device
    features = batch.features.to(device=device).detach()
    legal_mask = batch.legal_mask.to(device=device, dtype=torch.bool)
    actions = batch.actions.to(device=device, dtype=torch.long)
    output = model(features, legal_mask)
    logits, values = output.action_logits, output.values
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise ValueError("model outputs must be finite")

    dtype = values.dtype
    old_log_probs = batch.old_log_probs.to(device=device, dtype=dtype).detach()
    old_values = batch.old_values.to(device=device, dtype=dtype).detach()
    rewards = batch.rewards.to(device=device, dtype=dtype).detach()
    dones = batch.dones.to(device=device).detach()
    with torch.no_grad():
        advantages, returns = compute_gae(rewards, old_values, dones, config.gamma, config.gae_lambda)
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + torch.finfo(dtype).eps
        )

    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = torch.exp(selected_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    policy_loss = -torch.minimum(ratio * normalized_advantages, clipped_ratio * normalized_advantages).mean()
    value_loss = F.mse_loss(values, returns)

    policy_probs = torch.softmax(logits, dim=-1)
    legal_terms = legal_mask.to(dtype=dtype)
    entropy = -(policy_probs * log_probs * legal_terms).sum(dim=-1).mean()

    reference_logits = batch.reference_logits.to(device=device, dtype=dtype).detach()
    reference_logits = reference_logits.masked_fill(~legal_mask, torch.finfo(dtype).min)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    kl = (policy_probs * (log_probs - reference_log_probs) * legal_terms).sum(dim=-1).mean()

    total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy + config.kl_coef * kl
    if not torch.isfinite(total_loss):
        raise ValueError("PPO loss must be finite")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    return PPOHealth(
        total_loss=float(total_loss.detach()),
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        entropy=float(entropy.detach()),
        kl=float(kl.detach()),
        clip_fraction=float((ratio.sub(1.0).abs() > config.clip_ratio).float().mean().detach()),
        grad_norm=float(grad_norm.detach()),
    )
