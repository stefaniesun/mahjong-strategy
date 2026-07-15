from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Sequence


import torch
import torch.nn.functional as F

from learning.datasets.dataset_builder import PolicySample
from learning.device import resolve_device
from learning.models.belief_net import set_torch_seed
from learning.models.policy_net import PolicyNet, PolicyNetConfig
from state.encoder import ENCODER_VERSION



@dataclass(frozen=True)
class TrainPolicyConfig:
    model: PolicyNetConfig
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "auto"
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 1e-4
    forced_action_weight: float = 0.1
    discard_weight: float = 1.5
    swap_three_weight: float = 2.0
    declare_missing_suit_weight: float = 1.5
    pong_pass_weight: float = 2.0



@dataclass(frozen=True)
class PolicyBatch:
    features: torch.Tensor
    action_targets: torch.Tensor
    legal_mask: torch.Tensor
    sample_weights: torch.Tensor

    def to(self, device: torch.device) -> "PolicyBatch":
        return PolicyBatch(
            features=self.features.to(device),
            action_targets=self.action_targets.to(device),
            legal_mask=self.legal_mask.to(device),
            sample_weights=self.sample_weights.to(device),
        )


def policy_batch_from_samples(
    samples: Sequence[PolicySample],
    config: TrainPolicyConfig | None = None,
) -> PolicyBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    cfg = config
    weights = [_sample_weight(sample, cfg) if cfg is not None else 1.0 for sample in samples]
    return PolicyBatch(
        features=torch.tensor([sample.encoded.values for sample in samples], dtype=torch.float32),
        action_targets=torch.tensor([sample.action_index for sample in samples], dtype=torch.long),
        legal_mask=torch.tensor([sample.legal_mask for sample in samples], dtype=torch.bool),
        sample_weights=torch.tensor(weights, dtype=torch.float32),
    )


def _sample_weight(sample: PolicySample, config: TrainPolicyConfig) -> float:
    weight = config.forced_action_weight if sample.legal_action_count == 1 else 1.0
    kind_weights = {
        "discard": config.discard_weight,
        "swap_three": config.swap_three_weight,
        "declare_void": config.declare_missing_suit_weight,
    }
    weight *= kind_weights.get(sample.action_kind, 1.0)
    if sample.is_pong_pass_decision:
        weight *= config.pong_pass_weight
    return weight



def train_policy_step(model: PolicyNet, batch: PolicyBatch, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    output = model(batch.features, legal_mask=batch.legal_mask)
    weight_sum = batch.sample_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        raise ValueError("sample weight sum must be positive")
    per_sample_loss = F.cross_entropy(output.logits, batch.action_targets, reduction="none")
    loss = (per_sample_loss * batch.sample_weights).sum() / weight_sum
    loss.backward()

    optimizer.step()
    with torch.no_grad():
        predictions = output.logits.argmax(dim=-1)
        accuracy = (predictions == batch.action_targets).float().mean()
    return {"loss": float(loss.detach().cpu()), "top1_accuracy": float(accuracy.detach().cpu())}


def train_policy_epoch(samples: Sequence[PolicySample], config: TrainPolicyConfig) -> tuple[PolicyNet, dict[str, float | str]]:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        raise ValueError("samples must not be empty")
    set_torch_seed(config.seed)
    device = resolve_device(config.device)
    model = PolicyNet(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shuffled = list(samples)
    random.Random(config.seed).shuffle(shuffled)
    totals = {"loss": 0.0, "top1_accuracy": 0.0}
    batches = 0
    for start in range(0, len(shuffled), config.batch_size):
        batch = policy_batch_from_samples(shuffled[start : start + config.batch_size], config).to(device)

        metrics = train_policy_step(model, batch, optimizer)
        for key in totals:
            totals[key] += metrics[key]
        batches += 1
    averaged = {key: value / batches for key, value in totals.items()}
    averaged["batches"] = float(batches)
    averaged["samples"] = float(len(samples))
    averaged["device"] = device.type
    return model, averaged


def train_policy(
    train_samples: Sequence[PolicySample],
    validation_samples: Sequence[PolicySample],
    config: TrainPolicyConfig,
) -> tuple[PolicyNet, dict[str, Any]]:
    _validate_training_inputs(train_samples, validation_samples, config)
    set_torch_seed(config.seed)
    device = resolve_device(config.device)
    model = PolicyNet(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_accuracy = float("-inf")
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = _run_training_epoch(model, optimizer, train_samples, config, device, epoch)
        validation_metrics = _validation_metrics(model, validation_samples, config, device)
        candidate_accuracy = validation_metrics["non_forced_accuracy"]
        comparable_accuracy = -1.0 if candidate_accuracy is None else float(candidate_accuracy)
        candidate_loss = float(validation_metrics["loss"])
        improved = comparable_accuracy > best_accuracy + config.min_delta or (
            abs(comparable_accuracy - best_accuracy) <= config.min_delta and candidate_loss < best_loss
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if improved:
            best_state = deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            best_epoch = epoch
            best_accuracy = comparable_accuracy
            best_loss = candidate_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a best model")
    model.load_state_dict(best_state)
    model.to(device)
    best_validation_metrics = history[best_epoch - 1]["validation"]
    metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "early_stopped": len(history) < config.max_epochs,
        "history": history,
        "best_validation_metrics": best_validation_metrics,
        "device": device.type,
        "samples": len(train_samples),
    }
    return model, metrics


def _validate_training_inputs(
    train_samples: Sequence[PolicySample],
    validation_samples: Sequence[PolicySample],
    config: TrainPolicyConfig,
) -> None:
    if not train_samples:
        raise ValueError("train_samples must not be empty")
    if not validation_samples:
        raise ValueError("validation_samples must not be empty")
    if config.batch_size <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    weights = (
        config.forced_action_weight,
        config.discard_weight,
        config.swap_three_weight,
        config.declare_missing_suit_weight,
        config.pong_pass_weight,
    )
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("sample weights must be positive")


def _run_training_epoch(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[PolicySample],
    config: TrainPolicyConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    shuffled = list(samples)
    random.Random(config.seed + epoch - 1).shuffle(shuffled)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0
    for start in range(0, len(shuffled), config.batch_size):
        items = shuffled[start : start + config.batch_size]
        batch = policy_batch_from_samples(items, config).to(device)
        metrics = train_policy_step(model, batch, optimizer)
        total_loss += metrics["loss"] * len(items)
        total_correct += round(metrics["top1_accuracy"] * len(items))
        total_samples += len(items)
        batches += 1
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": total_correct / total_samples,
        "batches": float(batches),
        "samples": float(total_samples),
    }


def _validation_metrics(
    model: PolicyNet,
    samples: Sequence[PolicySample],
    config: TrainPolicyConfig,
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    batch = policy_batch_from_samples(samples, config).to(device)
    with torch.no_grad():
        output = model(batch.features, legal_mask=batch.legal_mask)
        per_sample = F.cross_entropy(output.logits, batch.action_targets, reduction="none")
        loss = (per_sample * batch.sample_weights).sum() / batch.sample_weights.sum()
        correct = output.logits.argmax(dim=-1) == batch.action_targets
        non_forced = batch.legal_mask.sum(dim=-1) > 1
    non_forced_count = int(non_forced.sum().detach().cpu())
    non_forced_accuracy = None
    if non_forced_count:
        non_forced_accuracy = float(correct[non_forced].float().mean().detach().cpu())
    return {
        "loss": float(loss.detach().cpu()),
        "top1_accuracy": float(correct.float().mean().detach().cpu()),
        "samples": len(samples),
        "non_forced_samples": non_forced_count,
        "non_forced_accuracy": non_forced_accuracy,
    }


def save_policy_checkpoint(

    path: str | Path,
    model: PolicyNet,
    config: TrainPolicyConfig,
    metrics: dict[str, float | str],
    *,
    data_fingerprint: str,
    belief_metadata: dict[str, Any] | None = None,
    split_summary: dict[str, Any] | None = None,
) -> None:

    payload = {
        "model_config": asdict(model.config),
        "encoder_version": ENCODER_VERSION,
        "training_config": asdict(config),

        "state_dict": model.state_dict(),
        "metrics": dict(metrics),
        "data_fingerprint": data_fingerprint,
        "execution_device": next(model.parameters()).device.type,
        "belief_metadata": dict(belief_metadata or {"source": "prior"}),
        "split_summary": dict(split_summary or {}),

    }
    torch.save(payload, path)
