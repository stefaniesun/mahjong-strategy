"""冻结策略版本: v6_human_tactics (git 481df17)。"""
from __future__ import annotations

from .pkg.bot import AssistantPolicy


def build_policy(
    *,
    stochastic: bool = False,
    seed: int | None = None,
    exact_swap_live: bool = False,
) -> AssistantPolicy:
    return AssistantPolicy(
        stochastic=stochastic,
        seed=seed,
        exact_swap_live=exact_swap_live,
    )


__all__ = ["AssistantPolicy", "build_policy"]
