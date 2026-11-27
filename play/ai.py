"""对手 AI 的加载:S4 监督模型、S5 强化学习存档,或 S3 规则 AI(调试用,启动最快)。"""
from __future__ import annotations

from pathlib import Path

from policies.base_policy import BasePolicy
from policies.rule_policy import RulePolicy

ROOT = Path(__file__).resolve().parents[1]
S4_POLICY = ROOT / "training_artifacts" / "S4" / "v5_20260718_encoder_v4" / "checkpoints" / "policy_s4.pt"


def load_ai(kind: str, checkpoint: Path | None = None, s4_policy: Path = S4_POLICY) -> tuple[BasePolicy, str]:
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
