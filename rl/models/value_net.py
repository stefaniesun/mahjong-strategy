from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from learning.models.belief_net import ResidualBlock


@dataclass(frozen=True)
class PolicyValueNetConfig:
    input_size: int
    action_size: int
    hidden_size: int = 128
    residual_blocks: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyValueNetOutput:
    action_logits: torch.Tensor
    values: torch.Tensor

    @property
    def action_probs(self) -> torch.Tensor:
        return torch.softmax(self.action_logits, dim=-1)


class PolicyValueNet(nn.Module):
    def __init__(self, config: PolicyValueNetConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            *[ResidualBlock(config.hidden_size, config.dropout) for _ in range(config.residual_blocks)],
        )
        self.action_head = nn.Linear(config.hidden_size, config.action_size)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self, features: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> PolicyValueNetOutput:
        trunk_output = self.trunk(features.to(dtype=self.trunk[0].weight.dtype))
        action_logits = self.action_head(trunk_output)
        if legal_mask is not None:
            if legal_mask.shape != action_logits.shape:
                raise ValueError("legal_mask shape must match logits shape")
            if not legal_mask.any(dim=-1).all():
                raise ValueError("each sample must have at least one legal action")
            action_logits = action_logits.masked_fill(
                ~legal_mask.bool(), torch.finfo(action_logits.dtype).min
            )
        return PolicyValueNetOutput(
            action_logits=action_logits,
            values=self.value_head(trunk_output).squeeze(-1),
        )

    def load_s4_policy_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        transfer_keys = [
            key
            for key in self.state_dict()
            if key.startswith("trunk.") or key.startswith("action_head.")
        ]
        missing_keys = [key for key in transfer_keys if key not in state_dict]
        if missing_keys:
            raise ValueError(f"missing S4 policy weights: {', '.join(missing_keys)}")

        mismatched_keys = [
            key
            for key in transfer_keys
            if state_dict[key].shape != self.state_dict()[key].shape
        ]
        if mismatched_keys:
            raise ValueError(f"S4 policy weight shape mismatch: {', '.join(mismatched_keys)}")

        self.load_state_dict({key: state_dict[key] for key in transfer_keys}, strict=False)
