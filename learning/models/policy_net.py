from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from learning.models.belief_net import ResidualBlock


@dataclass(frozen=True)
class PolicyNetConfig:
    input_size: int
    action_size: int
    hidden_size: int = 128
    residual_blocks: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyNetOutput:
    logits: torch.Tensor

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)


class PolicyNet(nn.Module):
    def __init__(self, config: PolicyNetConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            *[ResidualBlock(config.hidden_size, config.dropout) for _ in range(config.residual_blocks)],
        )
        self.action_head = nn.Linear(config.hidden_size, config.action_size)

    def forward(self, features: torch.Tensor, legal_mask: torch.Tensor | None = None) -> PolicyNetOutput:
        logits = self.action_head(self.trunk(features.float()))
        if legal_mask is not None:
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask shape must match logits shape")
            if not legal_mask.any(dim=-1).all():
                raise ValueError("each sample must have at least one legal action")
            logits = logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)
        return PolicyNetOutput(logits=logits)
