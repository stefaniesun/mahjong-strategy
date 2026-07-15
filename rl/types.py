"""Immutable data structures shared by S5 rollout and PPO code."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One learner decision recorded during a completed game rollout."""

    feature_values: Sequence[float]
    legal_mask: Sequence[bool]
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_values", tuple(self.feature_values))
        object.__setattr__(self, "legal_mask", tuple(self.legal_mask))
