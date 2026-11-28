"""对手 AI 的加载:S4 监督模型、S5 强化学习存档,或 S3 规则 AI(调试用,启动最快)。"""
from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from policies.base_policy import BasePolicy
from policies.rule_policy import RulePolicy

ROOT = Path(__file__).resolve().parents[1]
S4_POLICY = ROOT / "training_artifacts" / "S4" / "v5_20260718_encoder_v4" / "checkpoints" / "policy_s4.pt"

# 仅保留万局实测头部强策略(完全剔除 s3_rule / s3_plus / s4_v5 / v2 等弱策略)
TOP_MIX_ROSTER: tuple[tuple[str, bool, str], ...] = (
    ("v6_human_tactics", False, "v6实战精调"),
    ("v6_human_tactics", True, "v6精调·随"),
    ("v5_unified_ev", False, "v5统一EV"),
    ("v5_unified_ev", True, "v5统一EV·随"),
    ("v4_waitq_tactics", False, "v4听口暗杠"),
    ("v3_goodwait_read", False, "v3快攻读牌"),
)


class TopMixOpponentPool(BasePolicy):
    """每局为下家/对家/上家随机分配不同头部强策略(v6/v5/v4/v3), 模拟风格各异的真人高手对局。"""

    def __init__(self) -> None:
        from assistant.zoo import load_version

        self._specs = list(TOP_MIX_ROSTER)
        self._instances: dict[tuple[str, bool], Any] = {
            (ver, stoch): load_version(ver, stochastic=stoch, exact_swap_live=False)
            for ver, stoch, _ in self._specs
        }
        self._seat_spec: dict[int, tuple[str, bool, str]] = {}
        self.assign_hand_seats(random.Random(0), seed=0)

    def assign_hand_seats(self, rng: random.Random, seed: int) -> None:
        # 尽量让三家风格各不相同(不重复抽样), 更贴近真实三家真人
        chosen = rng.sample(self._specs, k=3)
        for offset, seat in enumerate((1, 2, 3)):
            spec = chosen[offset]
            self._seat_spec[seat] = spec
            pol = self._instances[(spec[0], spec[1])]
            if hasattr(pol, "rng"):
                pol.rng = random.Random(seed * 31 + seat * 17 + offset)

    def seat_label(self, seat: int) -> str | None:
        spec = self._seat_spec.get(seat)
        return spec[2] if spec else None

    def seat_summary(self) -> dict[int, str]:
        return {s: spec[2] for s, spec in self._seat_spec.items()}

    def choose_action(self, protocol_state, legal_mask: Sequence[bool]):
        return self._instances[("v6_human_tactics", False)].choose_action(protocol_state, legal_mask)

    def choose_from_game(self, engine_state, recorder, actor: int, legal_actions):
        spec = self._seat_spec.get(actor, self._specs[0])
        pol = self._instances[(spec[0], spec[1])]
        return pol.choose_from_game(engine_state, recorder, actor, legal_actions)


def load_ai(kind: str, checkpoint: Path | None = None, s4_policy: Path = S4_POLICY) -> tuple[BasePolicy, str]:
    if kind == "top_mix":
        return TopMixOpponentPool(), "顶级混编 (v6/v5/v4/v3 每局随机)"
    if kind == "rule":
        return RulePolicy(), "S3 规则 AI"
    if kind == "rule_plus":
        from policies.rule_policy_plus import PlusConfig, RulePlusPolicy

        return RulePlusPolicy(PlusConfig.parse("all")), "S3+ 增强规则 AI"
    if kind == "assistant":
        from assistant.bot import AssistantPolicy

        return AssistantPolicy(stochastic=True), "辅助器全套 AI (最新)"
    if kind.startswith("zoo:"):
        from assistant.zoo import load_version, parse_zoo_spec

        parsed = parse_zoo_spec(kind)
        if parsed is None:
            raise ValueError(f"未知的策略版本库 AI:{kind}")
        ver, stoch = parsed
        mode = "随机版" if stoch else "确定版"
        return load_version(ver, stochastic=stoch, exact_swap_live=True), f"版本库 {ver} ({mode})"
    if kind == "s4":
        from policies.learned_policy import LearnedPolicy

        path = checkpoint or s4_policy
        belief = path.with_name("belief_s4.pt")
        return LearnedPolicy(path, belief_model_path=belief if belief.is_file() else None), "S4 v5" if path == s4_policy else f"S4 ({path.parent.parent.name})"
    if kind == "s5":
        if checkpoint is None:
            raise ValueError("S5 需要用 --checkpoint 指定存档路径")
        import torch

        from rl.checkpoints import load_checkpoint
        from rl.opponent_resolver import ModelPolicy
        from rl.rollout import FrozenBeliefProvider
        from rl.train_rl import S5TrainingConfig, _default_model

        # 与 tools/evaluate_s5_snapshots.py 相同:先按 S4 结构建模型,再载入 S5 存档的参数
        model = _default_model(
            S5TrainingConfig(
                output_dir=s4_policy.parent,
                frozen_s4_belief_path=s4_policy,
                frozen_s4_policy_path=s4_policy,
                frozen_s4_provenance={"evaluation": True},
            )
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        restored = load_checkpoint(checkpoint, model=model, optimizer=optimizer, restore_rng=False, map_location="cpu")
        model.eval()
        belief = FrozenBeliefProvider.from_checkpoint(str(s4_policy.with_name("belief_s4.pt")))
        return ModelPolicy(model, belief_provider=belief, seed=0), f"S5 step {restored.global_step}"
    raise ValueError(f"未知的 AI 类型:{kind}")
