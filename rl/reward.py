"""Terminal-only reward helpers for S5 rollouts."""

from dataclasses import replace
from typing import Sequence

from rl.types import TrajectoryStep


def normalize_terminal_score(final_score: int | float, score_scale: float) -> float:
    """Scale a final settlement score into the reward used by PPO."""
    if not score_scale > 0:
        raise ValueError("score_scale must be positive")
    return float(final_score) / score_scale


def assign_terminal_reward(
    steps: Sequence[TrajectoryStep], final_score: int | float, score_scale: float
) -> tuple[TrajectoryStep, ...]:
    """Return a trajectory with reward and episode completion only at its end."""
    if not steps:
        raise ValueError("cannot assign a terminal reward to an empty trajectory")

    terminal_reward = normalize_terminal_score(final_score, score_scale)
    final_index = len(steps) - 1
    return tuple(
        replace(step, reward=terminal_reward if index == final_index else 0.0, done=index == final_index)
        for index, step in enumerate(steps)
    )
