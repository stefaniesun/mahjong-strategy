"""冻结策略版本: v2_ev_base (git 8cb0cfc)。"""
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
