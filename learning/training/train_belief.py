from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from engine.tiles import parse_tile
from learning.datasets.dataset_builder import BeliefSample
from learning.device import resolve_device
from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed


_OPPONENT_TO_INDEX = {"1": 0, "2": 1, "3": 2}


@dataclass(frozen=True)
class TrainBeliefConfig:
    model: BeliefNetConfig
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class BeliefBatch:

    features: torch.Tensor
    tile_location_targets: torch.Tensor
    tile_location_mask: torch.Tensor
    opponent_tenpai_targets: torch.Tensor
    opponent_tenpai_mask: torch.Tensor
    discard_danger_targets: torch.Tensor
    discard_danger_mask: torch.Tensor

    def to(self, device: torch.device) -> "BeliefBatch":
        return BeliefBatch(
            features=self.features.to(device),
            tile_location_targets=self.tile_location_targets.to(device),
            tile_location_mask=self.tile_location_mask.to(device),
            opponent_tenpai_targets=self.opponent_tenpai_targets.to(device),
            opponent_tenpai_mask=self.opponent_tenpai_mask.to(device),
            discard_danger_targets=self.discard_danger_targets.to(device),
            discard_danger_mask=self.discard_danger_mask.to(device),
        )


def belief_batch_from_samples(samples: Sequence[BeliefSample]) -> BeliefBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    features = torch.tensor([sample.encoded.values for sample in samples], dtype=torch.float32)
    tile_targets = torch.zeros((len(samples), 27, 4), dtype=torch.float32)
    tile_mask = torch.zeros((len(samples), 27), dtype=torch.bool)

    tenpai_targets = torch.zeros((len(samples), 3), dtype=torch.float32)
    tenpai_mask = torch.zeros((len(samples), 3), dtype=torch.bool)
    danger_targets = torch.zeros((len(samples), 27, 3), dtype=torch.float32)
    danger_mask = torch.zeros((len(samples), 27, 3), dtype=torch.bool)

    for row, sample in enumerate(samples):
        distribution, mask = _tile_location_labels(sample.labels.get("tile_locations"))
        tile_targets[row] = torch.tensor(distribution, dtype=torch.float32)
        tile_mask[row] = torch.tensor(mask, dtype=torch.bool)
        for relative, value in sample.labels.get("opponent_tenpai", {}).items():

            if str(relative) in _OPPONENT_TO_INDEX:
                index = _OPPONENT_TO_INDEX[str(relative)]
                tenpai_targets[row, index] = float(bool(value))
                tenpai_mask[row, index] = True
        for tile_text, per_opponent in sample.labels.get("discard_danger", {}).items():
            tile_index = parse_tile(tile_text).index
            for relative, value in per_opponent.items():
                if str(relative) in _OPPONENT_TO_INDEX:
                    index = _OPPONENT_TO_INDEX[str(relative)]
                    danger_targets[row, tile_index, index] = float(bool(value))
                    danger_mask[row, tile_index, index] = True

    return BeliefBatch(
        features=features,
        tile_location_targets=tile_targets,
        tile_location_mask=tile_mask,
        opponent_tenpai_targets=tenpai_targets,
        opponent_tenpai_mask=tenpai_mask,
        discard_danger_targets=danger_targets,
        discard_danger_mask=danger_mask,
    )


def train_belief_step(model: BeliefNet, batch: BeliefBatch, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    output = model(batch.features)
    tile_loss = _masked_tile_location_loss(output.tile_location_logits, batch.tile_location_targets, batch.tile_location_mask)
    tenpai_loss = _masked_binary_loss(output.opponent_tenpai_logits, batch.opponent_tenpai_targets, batch.opponent_tenpai_mask)
    danger_loss = _masked_binary_loss(output.discard_danger_logits, batch.discard_danger_targets, batch.discard_danger_mask)
    loss = tile_loss + tenpai_loss + danger_loss
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "tile_location_loss": float(tile_loss.detach().cpu()),
        "opponent_tenpai_loss": float(tenpai_loss.detach().cpu()),
        "discard_danger_loss": float(danger_loss.detach().cpu()),
    }


def train_belief_epoch(samples: Sequence[BeliefSample], config: TrainBeliefConfig) -> tuple[BeliefNet, dict[str, float | str]]:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        raise ValueError("samples must not be empty")
    set_torch_seed(config.seed)
    device = resolve_device(config.device)
    model = BeliefNet(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shuffled = list(samples)
    random.Random(config.seed).shuffle(shuffled)
    totals: dict[str, float] = {"loss": 0.0, "tile_location_loss": 0.0, "opponent_tenpai_loss": 0.0, "discard_danger_loss": 0.0}
    batches = 0
    for start in range(0, len(shuffled), config.batch_size):
        batch = belief_batch_from_samples(shuffled[start : start + config.batch_size]).to(device)
        metrics = train_belief_step(model, batch, optimizer)
        for key in totals:
            totals[key] += metrics[key]
        batches += 1
    averaged = {key: value / batches for key, value in totals.items()}
    averaged["batches"] = float(batches)
    averaged["samples"] = float(len(samples))
    averaged["device"] = device.type
    return model, averaged





def _tile_location_labels(labels: object) -> tuple[list[list[float]], list[bool]]:
    if not isinstance(labels, dict) or set(labels) != {"counts", "distribution", "mask"}:
        raise ValueError("legacy or incompatible tile-location labels; regenerate the dataset")
    distribution = labels["distribution"]
    mask = labels["mask"]
    if not isinstance(distribution, list) or len(distribution) != 27 or not isinstance(mask, list) or len(mask) != 27:
        raise ValueError("tile-location distribution must be 27x4 and mask must have length 27")
    normalized: list[list[float]] = []
    for index, row in enumerate(distribution):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("tile-location distribution must be 27x4")
        values = [float(value) for value in row]
        if bool(mask[index]) and (any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6):
            raise ValueError("masked tile-location distributions must be finite, non-negative, and sum to one")
        normalized.append(values)
    return normalized, [bool(value) for value in mask]


def _masked_tile_location_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    per_tile = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    return per_tile[mask].mean()


def _masked_binary_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
