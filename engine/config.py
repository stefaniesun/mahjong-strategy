from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


SELF_DRAW_MODES = {"add_di", "add_fan"}
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "rules.yaml"
RULES_ENV = "MAHJONG_RULES"


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


@lru_cache(maxsize=1)
def active_rule_config() -> RuleConfig:
    """当前生效的规则:环境变量 MAHJONG_RULES 指向的文件,否则是 configs/rules.yaml。

    换规则重新训练只需要改这份配置;没有显式传规则的地方都读这里,所以各层不会各算各的。
    """
    return load_rule_config(os.environ.get(RULES_ENV) or DEFAULT_RULES_PATH)
