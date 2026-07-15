from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from learning.datasets.dataset_builder import PolicySample
from learning.models.policy_net import PolicyNet
from learning.training.train_policy import policy_batch_from_samples


@dataclass(frozen=True)
class PolicySliceMetrics:
    samples: int
    accuracy: float | None


@dataclass(frozen=True)
class PolicyEvalReport:
    samples: int
    top1_accuracy: float
    illegal_argmax_count: int
    illegal_probability_mass: float
    forced_samples: int
    forced_rate: float
    non_forced_samples: int
    non_forced_accuracy: float | None
    by_action_kind: dict[str, PolicySliceMetrics]
    by_phase: dict[str, PolicySliceMetrics]
    pong_pass_response: PolicySliceMetrics



def evaluate_policy_samples(model: PolicyNet, samples: Sequence[PolicySample]) -> PolicyEvalReport:
    if not samples:
        raise ValueError("samples must not be empty")
    device = next(model.parameters()).device
    batch = policy_batch_from_samples(samples).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch.features, legal_mask=batch.legal_mask)
        predictions = output.logits.argmax(dim=-1)
        correct = (predictions == batch.action_targets).detach().cpu().tolist()
        accuracy = (predictions == batch.action_targets).float().mean()
        illegal_argmax = (~batch.legal_mask.gather(1, predictions.unsqueeze(1)).squeeze(1)).sum()
        illegal_mass = output.probs.masked_fill(batch.legal_mask, 0.0).sum(dim=-1).mean()
    forced = [sample.legal_action_count == 1 for sample in samples]
    non_forced_correct = [is_correct for is_correct, is_forced in zip(correct, forced) if not is_forced]
    by_action_kind = _group_metrics(samples, correct, lambda sample: sample.action_kind)
    by_phase = _group_metrics(samples, correct, lambda sample: sample.phase)
    response_correct = [is_correct for sample, is_correct in zip(samples, correct) if sample.is_pong_pass_decision]
    forced_samples = sum(forced)
    return PolicyEvalReport(
        samples=len(samples),
        top1_accuracy=float(accuracy.detach().cpu()),
        illegal_argmax_count=int(illegal_argmax.detach().cpu()),
        illegal_probability_mass=float(illegal_mass.detach().cpu()),
        forced_samples=forced_samples,
        forced_rate=forced_samples / len(samples),
        non_forced_samples=len(non_forced_correct),
        non_forced_accuracy=_accuracy(non_forced_correct),
        by_action_kind=by_action_kind,
        by_phase=by_phase,
        pong_pass_response=PolicySliceMetrics(len(response_correct), _accuracy(response_correct)),
    )


def _accuracy(correct: Sequence[bool]) -> float | None:
    return sum(correct) / len(correct) if correct else None


def _group_metrics(
    samples: Sequence[PolicySample],
    correct: Sequence[bool],
    key,
) -> dict[str, PolicySliceMetrics]:
    grouped: dict[str, list[bool]] = {}
    for sample, is_correct in zip(samples, correct):
        grouped.setdefault(str(key(sample)), []).append(is_correct)
    return {name: PolicySliceMetrics(len(items), _accuracy(items)) for name, items in sorted(grouped.items())}

