import json

import pytest

from rl.curriculum import (
    CurriculumConfig,
    CurriculumStage,
    DegradationProfile,
    ObservationCurriculum,
)
from rl.league import DualArenaMetrics


def _config() -> CurriculumConfig:
    return CurriculumConfig(
        stages=(
            CurriculumStage("light", DegradationProfile("light", vision_miss_rate=0.02, mid_game_ratio=0.0), 0.5, 0.5),
            CurriculumStage("heavy", DegradationProfile("heavy", vision_miss_rate=0.20, mid_game_ratio=0.5), 0.6, 0.55),
        )
    )


def test_only_learner_observation_uses_curriculum_degradation():
    curriculum = ObservationCurriculum(_config())

    assert curriculum.learner_profile.name == "light"
    assert curriculum.opponent_profile == DegradationProfile.perfect()
    assert curriculum.opponent_profile.is_perfect
    assert curriculum.opponent_profile != curriculum.learner_profile


def test_stage_advances_deterministically_only_when_dual_arena_criterion_is_met():
    curriculum = ObservationCurriculum(_config())
    failing = DualArenaMetrics(perfect_win_rate=0.8, degraded_win_rate=0.49, perfect_games=1000, degraded_games=1000, illegal_actions=0, zero_sum_failures=0)
    passing = DualArenaMetrics(perfect_win_rate=0.5, degraded_win_rate=0.5, perfect_games=1000, degraded_games=1000, illegal_actions=0, zero_sum_failures=0)

    assert curriculum.advance(failing) is False
    assert curriculum.stage_index == 0
    assert curriculum.advance(passing) is True
    assert curriculum.stage_index == 1
    assert curriculum.advance(DualArenaMetrics(1.0, 1.0, 1000, 1000, 0, 0)) is False
    assert curriculum.stage_index == 1


def test_curriculum_json_roundtrip_preserves_stage_and_never_serializes_opponent_noise():
    curriculum = ObservationCurriculum(_config())
    curriculum.advance(DualArenaMetrics(0.8, 0.8, 1000, 1000, 0, 0))

    restored = ObservationCurriculum.from_json(curriculum.to_json())

    assert restored.to_dict() == curriculum.to_dict()
    assert restored.stage_index == 1
    assert restored.opponent_profile == DegradationProfile.perfect()


@pytest.mark.parametrize("profile", [
    {"name": 5, "vision_miss_rate": 0.1, "mid_game_ratio": 0.2},
    {"name": "bad", "vision_miss_rate": "0.1", "mid_game_ratio": 0.2},
    {"name": "bad", "vision_miss_rate": float("nan"), "mid_game_ratio": 0.2},
])
def test_degradation_profile_rejects_invalid_runtime_types(profile):
    with pytest.raises((TypeError, ValueError)):
        DegradationProfile(**profile)


def test_curriculum_checkpoint_rejects_coercible_values():
    state = ObservationCurriculum(_config()).to_dict()
    state["stage_index"] = True
    with pytest.raises((TypeError, ValueError)):
        ObservationCurriculum.from_dict(state)

    state = ObservationCurriculum(_config()).to_dict()
    stages = state["config"]["stages"]
    stages[0]["min_perfect_win_rate"] = "0.5"
    with pytest.raises((TypeError, ValueError)):
        ObservationCurriculum.from_dict(state)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda state: state.__setitem__("stage_index", True), "stage_index"),
        (
            lambda state: state["config"]["stages"][0].__setitem__("min_perfect_win_rate", "0.5"),  # type: ignore[index]
            "perfect_win_rate",
        ),
    ],
)
def test_curriculum_json_roundtrip_rejects_strict_scalar_types(mutate, error):
    """Strict decoding must reject one invalid scalar in an otherwise valid JSON state."""
    payload = ObservationCurriculum(_config()).to_json()
    state = json.loads(payload)
    mutate(state)

    with pytest.raises((TypeError, ValueError), match=error):
        ObservationCurriculum.from_json(json.dumps(state))


def test_curriculum_public_constructors_reject_untyped_fields():
    with pytest.raises(TypeError):
        CurriculumStage("bad", "not-a-profile", 0.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CurriculumConfig([_config().stages[0]])  # type: ignore[arg-type]


def test_curriculum_checkpoint_requires_all_serialized_stage_fields():
    state = ObservationCurriculum(_config()).to_dict()
    del state["config"]["stages"][0]["learner_profile"]["mid_game_ratio"]

    with pytest.raises(ValueError, match="missing"):
        ObservationCurriculum.from_dict(state)
