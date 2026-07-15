from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BeliefNetConfig:
    input_size: int
    hidden_size: int = 128
    residual_blocks: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class BeliefNetOutput:
    tile_location_logits: torch.Tensor
    opponent_tenpai_logits: torch.Tensor
    discard_danger_logits: torch.Tensor

    @property
    def tile_location_probs(self) -> torch.Tensor:
        return torch.softmax(self.tile_location_logits, dim=-1)

    @property
    def opponent_tenpai_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.opponent_tenpai_logits)

    @property
    def discard_danger_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.discard_danger_logits)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class BeliefNet(nn.Module):
    def __init__(self, config: BeliefNetConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            *[ResidualBlock(config.hidden_size, config.dropout) for _ in range(config.residual_blocks)],
        )
        self.tile_location_head = nn.Linear(config.hidden_size, 27 * 4)
        self.opponent_tenpai_head = nn.Linear(config.hidden_size, 3)
        self.discard_danger_head = nn.Linear(config.hidden_size, 27 * 3)

    def forward(self, features: torch.Tensor) -> BeliefNetOutput:
        hidden = self.trunk(features.float())
        batch_size = hidden.shape[0]
        return BeliefNetOutput(
            tile_location_logits=self.tile_location_head(hidden).reshape(batch_size, 27, 4),
            opponent_tenpai_logits=self.opponent_tenpai_head(hidden),
            discard_danger_logits=self.discard_danger_head(hidden).reshape(batch_size, 27, 3),
        )


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
