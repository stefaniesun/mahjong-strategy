from __future__ import annotations

import math
from typing import Iterable, Sequence


def multiclass_log_loss(predictions: Sequence[Sequence[float]], targets: Sequence[int], *, eps: float = 1e-12) -> float:
    _check_same_length(predictions, targets)
    if not predictions:
        return 0.0
    total = 0.0
    for row, target in zip(predictions, targets):
        if target < 0 or target >= len(row):
            raise ValueError("target index out of range")
        total -= math.log(_clamp(float(row[target]), eps, 1.0))
    return total / len(predictions)


def multiclass_brier_score(predictions: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _check_same_length(predictions, targets)
    if not predictions:
        return 0.0
    total = 0.0
    for row, target in zip(predictions, targets):
        if target < 0 or target >= len(row):
            raise ValueError("target index out of range")
        total += sum((float(probability) - (1.0 if index == target else 0.0)) ** 2 for index, probability in enumerate(row))
    return total / len(predictions)


def soft_multiclass_log_loss(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
) -> float:
    _check_soft_multiclass_inputs(predictions, targets)
    if not predictions:
        return 0.0
    return sum(
        -sum(float(target) * math.log(_clamp(float(probability), eps, 1.0)) for probability, target in zip(row, target_row))
        for row, target_row in zip(predictions, targets)
    ) / len(predictions)


def soft_multiclass_brier_score(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> float:
    _check_soft_multiclass_inputs(predictions, targets)
    if not predictions:
        return 0.0
    return sum(
        sum((float(probability) - float(target)) ** 2 for probability, target in zip(row, target_row))
        for row, target_row in zip(predictions, targets)
    ) / len(predictions)


def binary_brier_score(probabilities: Sequence[float], targets: Sequence[int | bool]) -> float:
    _check_same_length(probabilities, targets)
    if not probabilities:
        return 0.0
    return sum((float(probability) - float(target)) ** 2 for probability, target in zip(probabilities, targets)) / len(probabilities)


def binary_ece(probabilities: Sequence[float], targets: Sequence[int | bool], *, bins: int = 10) -> float:
    _check_same_length(probabilities, targets)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not probabilities:
        return 0.0

    total = len(probabilities)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        in_bin = []
        for probability, target in zip(probabilities, targets):
            probability = float(probability)
            if (bin_index == bins - 1 and lower <= probability <= upper) or (bin_index < bins - 1 and lower <= probability < upper):
                in_bin.append((probability, float(target)))

        if not in_bin:
            continue
        confidence = sum(probability for probability, _ in in_bin) / len(in_bin)
        accuracy = sum(target for _, target in in_bin) / len(in_bin)
        ece += len(in_bin) / total * abs(confidence - accuracy)
    return ece


def _check_same_length(left: Iterable[object], right: Iterable[object]) -> None:
    if len(left) != len(right):  # type: ignore[arg-type]
        raise ValueError("inputs must have the same length")


def _check_soft_multiclass_inputs(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> None:
    _check_same_length(predictions, targets)
    for row, target_row in zip(predictions, targets):
        if len(row) != len(target_row):
            raise ValueError("prediction and target rows must have the same length")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in target_row):
            raise ValueError("soft targets must be finite and non-negative")
        if not math.isclose(sum(float(value) for value in target_row), 1.0, abs_tol=1e-6):
            raise ValueError("soft target rows must sum to one")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
