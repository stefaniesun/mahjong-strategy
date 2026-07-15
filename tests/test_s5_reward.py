from dataclasses import FrozenInstanceError

import pytest

from rl.reward import assign_terminal_reward, normalize_terminal_score
from rl.types import TrajectoryStep


def transition() -> TrajectoryStep:
    return TrajectoryStep(
        feature_values=[0.25, 0.75],
        legal_mask=[True, False, True],
        action=2,
        old_log_prob=-0.4,
        value=0.1,
        reward=0.0,
        done=False,
        policy_version="s5-policy-7",
    )


def test_terminal_reward_is_only_on_final_transition() -> None:
    transitions = [transition(), transition(), transition()]

    rewarded = assign_terminal_reward(transitions, final_score=24, score_scale=48.0)

    assert [item.reward for item in rewarded] == [0.0, 0.0, 0.5]
    assert [item.done for item in rewarded] == [False, False, True]
    assert isinstance(rewarded, tuple)


def test_normalize_terminal_score_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="score_scale"):
        normalize_terminal_score(12, 0.0)


def test_normalize_terminal_score_rejects_nan_scale() -> None:
    with pytest.raises(ValueError, match="score_scale"):
        normalize_terminal_score(12, float("nan"))


def test_assign_terminal_reward_rejects_empty_trajectories() -> None:
    with pytest.raises(ValueError, match="empty"):
        assign_terminal_reward([], final_score=12, score_scale=48.0)


def test_trajectory_step_is_immutable_and_tuple_safe() -> None:
    step = transition()

    assert step.feature_values == (0.25, 0.75)
    assert step.legal_mask == (True, False, True)
    with pytest.raises(FrozenInstanceError):
        step.reward = 1.0  # type: ignore[misc]
