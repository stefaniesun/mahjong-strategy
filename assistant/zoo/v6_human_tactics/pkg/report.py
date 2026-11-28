"""统一输出结构:每条结论都带档位、依据,第③档必须带假设。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TIER_CERTAIN = 1
TIER_COUNT = 2
TIER_PROBABILITY = 3
TIER_NAMES = {TIER_CERTAIN: "确定", TIER_COUNT: "计数", TIER_PROBABILITY: "概率"}

ASSUMPTION_RANDOM = "按未知牌随机分布估算:假设所有未出现的牌随机分布在牌墙和各家暗手牌中,未考虑对手留牌的选择。"


@dataclass(frozen=True)
class Finding:
    tier: int
    category: str
    title: str
    basis: tuple[str, ...] = ()
    assumption: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in TIER_NAMES:
            raise ValueError(f"unknown tier: {self.tier}")
        if self.tier == TIER_PROBABILITY and not self.assumption:
            raise ValueError("tier-3 findings must state their assumption")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_name": TIER_NAMES[self.tier],
            "category": self.category,
            "title": self.title,
            "basis": list(self.basis),
            "assumption": self.assumption,
            "data": self.data,
        }


def certain(category: str, title: str, basis=(), **data: Any) -> Finding:
    return Finding(TIER_CERTAIN, category, title, tuple(basis), None, data)


def counted(category: str, title: str, basis=(), **data: Any) -> Finding:
    return Finding(TIER_COUNT, category, title, tuple(basis), None, data)


def probable(category: str, title: str, basis=(), assumption: str = ASSUMPTION_RANDOM, **data: Any) -> Finding:
    return Finding(TIER_PROBABILITY, category, title, tuple(basis), assumption, data)
