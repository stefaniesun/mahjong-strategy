from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SELF_DRAW_MODES = {"add_di", "add_fan"}


@dataclass(frozen=True)
class RuleConfig:
    base_score: int = 1
    max_fan: int = 4
    self_draw_mode: str = "add_di"
    enable_heavenly_hand: bool = False
    enable_earthly_hand: bool = False
    enable_eighteen_arhats: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RuleConfig":
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.base_score, int) or self.base_score <= 0:
            raise ValueError("base_score must be a positive integer")
        if not isinstance(self.max_fan, int) or self.max_fan < 0:
            raise ValueError("max_fan must be a non-negative integer")
        if self.self_draw_mode not in SELF_DRAW_MODES:
            raise ValueError("self_draw_mode must be one of: add_di, add_fan")
        for field_name in (
            "enable_heavenly_hand",
            "enable_earthly_hand",
            "enable_eighteen_arhats",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")


def load_rule_config(path: str | Path) -> RuleConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("rules config must be a mapping")
    return RuleConfig.from_mapping(data)
