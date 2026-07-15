"""Deterministic learner-only observation-degradation curriculum for S5."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

from rl.league import DualArenaMetrics


@dataclass(frozen=True, slots=True)
class DegradationProfile:
    """Portable learner observation profile; zero values mean perfect visibility."""

    name: str
    vision_miss_rate: float = 0.0
    mid_game_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("degradation profile name must be a string")
        if not self.name:
            raise ValueError("degradation profile name must be non-empty")
        for name, value in (("vision_miss_rate", self.vision_miss_rate), ("mid_game_ratio", self.mid_game_ratio)):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def is_perfect(self) -> bool:
        return self.vision_miss_rate == 0.0 and self.mid_game_ratio == 0.0

    @classmethod
    def perfect(cls) -> "DegradationProfile":
        return cls("perfect", vision_miss_rate=0.0, mid_game_ratio=0.0)


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One ordered learner profile and its explicit dual-arena promotion rule."""

    name: str
    learner_profile: DegradationProfile
    min_perfect_win_rate: float
    min_degraded_win_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("curriculum stage name must be a string")
        if not self.name:
            raise ValueError("curriculum stage name must be non-empty")
        if not isinstance(self.learner_profile, DegradationProfile):
            raise TypeError("learner_profile must be DegradationProfile")
        for name, value in (
            ("min_perfect_win_rate", self.min_perfect_win_rate),
            ("min_degraded_win_rate", self.min_degraded_win_rate),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """An explicitly ordered, serializable sequence of learner-only stages."""

    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            raise TypeError("curriculum stages must be a tuple")
        if not self.stages:
            raise ValueError("curriculum requires at least one stage")
        if any(not isinstance(stage, CurriculumStage) for stage in self.stages):
            raise TypeError("curriculum stages must be CurriculumStage values")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("curriculum stage names must be unique and ordered")


class ObservationCurriculum:
    """Stateful stage selector; opponents are permanently perfect-observation."""

    def __init__(self, config: CurriculumConfig, *, stage_index: int = 0) -> None:
        if not isinstance(config, CurriculumConfig):
            raise TypeError("config must be CurriculumConfig")
        if not isinstance(stage_index, int) or isinstance(stage_index, bool) or not 0 <= stage_index < len(config.stages):
            raise ValueError("stage_index is outside configured stages")
        self.config = config
        self.stage_index = stage_index

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self.stage_index]

    @property
    def learner_profile(self) -> DegradationProfile:
        return self.current_stage.learner_profile

    @property
    def opponent_profile(self) -> DegradationProfile:
        """The isolation boundary: no curriculum stage may degrade opponents."""
        return DegradationProfile.perfect()

    def advance(self, metrics: DualArenaMetrics) -> bool:
        """Advance exactly one stage only after the current explicit criterion passes."""
        if not isinstance(metrics, DualArenaMetrics):
            raise TypeError("metrics must be DualArenaMetrics")
        if self.stage_index == len(self.config.stages) - 1:
            return False
        if (
            metrics.perfect_games <= 0
            or metrics.degraded_games <= 0
            or metrics.perfect_games != metrics.degraded_games
            or metrics.illegal_actions != 0
            or metrics.zero_sum_failures != 0
        ):
            return False
        stage = self.current_stage
        if (
            metrics.perfect_win_rate < stage.min_perfect_win_rate
            or metrics.degraded_win_rate < stage.min_degraded_win_rate
        ):
            return False
        self.stage_index += 1
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "stages": [
                    {
                        "name": stage.name,
                        "learner_profile": asdict(stage.learner_profile),
                        "min_perfect_win_rate": stage.min_perfect_win_rate,
                        "min_degraded_win_rate": stage.min_degraded_win_rate,
                    }
                    for stage in self.config.stages
                ]
            },
            "stage_index": self.stage_index,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ObservationCurriculum":
        if not isinstance(data, Mapping):
            raise TypeError("curriculum state must be a mapping")
        state = _exact_mapping(data, "curriculum state", {"config", "stage_index"})
        config_data = _exact_mapping(state["config"], "curriculum config", {"stages"})
        raw_stages = config_data["stages"]
        if not isinstance(raw_stages, list):
            raise TypeError("curriculum stages must be a list")
        stages: list[CurriculumStage] = []
        for raw_stage in raw_stages:
            stage_data = _exact_mapping(raw_stage, "curriculum stage", _STAGE_FIELDS)
            profile = _exact_mapping(stage_data["learner_profile"], "learner_profile", _PROFILE_FIELDS)
            stages.append(
                CurriculumStage(
                    name=stage_data["name"],
                    learner_profile=DegradationProfile(**profile),
                    min_perfect_win_rate=stage_data["min_perfect_win_rate"],
                    min_degraded_win_rate=stage_data["min_degraded_win_rate"],
                )
            )
        return cls(CurriculumConfig(tuple(stages)), stage_index=state["stage_index"])

    @classmethod
    def from_json(cls, payload: str) -> "ObservationCurriculum":
        if not isinstance(payload, str):
            raise TypeError("curriculum JSON must be a string")
        return cls.from_dict(json.loads(payload))


_STAGE_FIELDS = frozenset({"name", "learner_profile", "min_perfect_win_rate", "min_degraded_win_rate"})
_PROFILE_FIELDS = frozenset({"name", "vision_miss_rate", "mid_game_ratio"})


def _exact_mapping(value: object, name: str, expected_fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    data = dict(value)
    missing = expected_fields.difference(data)
    if missing:
        raise ValueError(f"{name} missing required field: {sorted(missing)[0]}")
    unexpected = set(data).difference(expected_fields)
    if unexpected:
        raise ValueError(f"{name} has unexpected field: {sorted(unexpected)[0]}")
    return data
