from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from engine.tiles import Tile, parse_tile, tile_to_str
from learning.datasets.dataset_builder import BeliefSample, DatasetBuildConfig, _degrade, _record_state, build_belief_sample

from learning.models.belief_net import BeliefNet
from learning.training.train_belief import _tile_location_labels
from learning.training.metrics import (
    binary_brier_score,
    binary_ece,
    soft_multiclass_brier_score,
    soft_multiclass_log_loss,
)
from selfplay.data_recorder import DecisionRecord
from state.tile_belief import PriorBelief



_LOCATION_TO_INDEX = {"wall": 0, "1": 1, "2": 2, "3": 3}
_OPPONENT_TO_INDEX = {"1": 0, "2": 1, "3": 2}


@dataclass(frozen=True)
class BeliefEvalReport:
    samples: int
    tile_count: int
    tile_log_loss: float
    tile_brier: float
    opponent_tenpai_count: int
    opponent_tenpai_brier: float
    opponent_tenpai_ece: float
    discard_danger_count: int
    discard_danger_brier: float
    discard_danger_ece: float


def evaluate_belief_model(model: BeliefNet, samples: Sequence[BeliefSample]) -> BeliefEvalReport:
    if not samples:
        raise ValueError("samples must not be empty")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        features = torch.tensor([sample.encoded.values for sample in samples], dtype=torch.float32, device=device)
        output = model(features)
        tile_probs = output.tile_location_probs.detach().cpu().tolist()
        tenpai_probs = output.opponent_tenpai_probs.detach().cpu().tolist()
        danger_probs = output.discard_danger_probs.detach().cpu().tolist()
    return _report_from_predictions(samples, tile_probs, tenpai_probs, danger_probs)



def evaluate_belief_model_by_phase(model: BeliefNet, samples: Sequence[BeliefSample]) -> dict[str, BeliefEvalReport]:
    grouped: dict[str, list[BeliefSample]] = {}
    for sample in samples:
        grouped.setdefault(_phase_bucket(sample.phase, sample.step), []).append(sample)
    return {phase: evaluate_belief_model(model, phase_samples) for phase, phase_samples in grouped.items()}


def evaluate_prior_belief_records(records: Sequence[DecisionRecord], config: DatasetBuildConfig) -> BeliefEvalReport:
    samples = [build_belief_sample(record, config) for record in records]
    if not samples:
        raise ValueError("records must not be empty")
    prior_predictions: list[list[list[float]]] = []
    tenpai_predictions: list[list[float]] = []
    danger_predictions: list[list[list[float]]] = []
    prior = PriorBelief()
    for record in records:
        beliefs = prior.infer(_sample_state(record, config))

        tile_locations = beliefs.tile_location_beliefs.value or {}
        tenpai = beliefs.opponent_tenpai_beliefs.value or {}
        danger = beliefs.discard_danger.value or {}
        prior_predictions.append([
            [float(tile_locations.get(tile, {}).get(location, 0.0)) for location in ("wall", "1", "2", "3")]
            for tile in _tile_texts()
        ])
        tenpai_predictions.append([float(tenpai.get(relative, 0.0)) for relative in ("1", "2", "3")])
        danger_predictions.append([
            [float(danger.get(tile, {}).get(relative, 0.0)) for relative in ("1", "2", "3")]
            for tile in _tile_texts()
        ])
    return _report_from_predictions(samples, prior_predictions, tenpai_predictions, danger_predictions)


def evaluate_prior_belief_records_by_profile(records: Sequence[DecisionRecord], configs: Sequence[DatasetBuildConfig]) -> dict[str, BeliefEvalReport]:
    return {config.degradation_profile: evaluate_prior_belief_records(records, config) for config in configs}


def _report_from_predictions(
    samples: Sequence[BeliefSample],
    tile_probs: Sequence[Sequence[Sequence[float]]],
    tenpai_probs: Sequence[Sequence[float]],
    danger_probs: Sequence[Sequence[Sequence[float]]],
) -> BeliefEvalReport:
    tile_predictions: list[list[float]] = []
    tile_targets: list[list[float]] = []

    tenpai_predictions: list[float] = []
    tenpai_targets: list[bool] = []
    danger_predictions: list[float] = []
    danger_targets: list[bool] = []

    for row, sample in enumerate(samples):
        distribution, mask = _tile_location_labels(sample.labels.get("tile_locations"))
        for tile_index, include in enumerate(mask):
            if not include:
                continue
            tile_predictions.append(list(tile_probs[row][tile_index]))
            tile_targets.append(distribution[tile_index])
        for relative, value in sample.labels.get("opponent_tenpai", {}).items():

            if str(relative) not in _OPPONENT_TO_INDEX:
                continue
            tenpai_predictions.append(float(tenpai_probs[row][_OPPONENT_TO_INDEX[str(relative)]]))
            tenpai_targets.append(bool(value))
        for tile_text, per_opponent in sample.labels.get("discard_danger", {}).items():
            tile_index = parse_tile(tile_text).index
            for relative, value in per_opponent.items():
                if str(relative) not in _OPPONENT_TO_INDEX:
                    continue
                danger_predictions.append(float(danger_probs[row][tile_index][_OPPONENT_TO_INDEX[str(relative)]]))
                danger_targets.append(bool(value))

    return BeliefEvalReport(
        samples=len(samples),
        tile_count=len(tile_targets),
        tile_log_loss=soft_multiclass_log_loss(tile_predictions, tile_targets),
        tile_brier=soft_multiclass_brier_score(tile_predictions, tile_targets),
        opponent_tenpai_count=len(tenpai_targets),
        opponent_tenpai_brier=binary_brier_score(tenpai_predictions, tenpai_targets),
        opponent_tenpai_ece=binary_ece(tenpai_predictions, tenpai_targets),
        discard_danger_count=len(danger_targets),
        discard_danger_brier=binary_brier_score(danger_predictions, danger_targets),
        discard_danger_ece=binary_ece(danger_predictions, danger_targets),
    )


def _sample_state(record: DecisionRecord, config: DatasetBuildConfig):
    return _degrade(_record_state(record), config, record)


def _tile_texts() -> list[str]:
    return [tile_to_str(Tile.from_index(index)) for index in range(27)]


def _phase_bucket(phase: str, step: int) -> str:
    normalized = phase.lower()
    if "opening" in normalized or "exchange" in normalized or "void" in normalized or step < 18:
        return "opening"
    if "late" in normalized or step >= 72:
        return "late"
    return "middle"

