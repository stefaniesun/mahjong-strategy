"""端到端测试 Zoo SL 数据生成器: 生成分片 -> 被 S4 训练管线直接加载 -> 构建 Belief/Policy 样本并编码。"""
from __future__ import annotations

from pathlib import Path

from learning.datasets.dataset_builder import DatasetBuildConfig, _clean_action, build_belief_sample
from learning.models.belief_net import BeliefNet, BeliefNetConfig
from state.action_space import action_to_index
from state.encoder import encoding_size
from state.tile_belief import LearnedBelief
from tools.cloud_train_s4_50k_cached import (
    TrainingConfig,
    _iter_records,
    _load_manifest,
    _policy_samples,
)
from tools.generate_zoo_sl_data import generate_dataset


def test_generate_zoo_sl_data_matches_s4_training_pipeline(tmp_path: Path):
    out_dir = tmp_path / "zoo_sl_smoke"
    manifest = generate_dataset(
        games=4,
        seed_start=990001,
        shard_games=2,
        out_dir=out_dir,
        workers=1,
        teachers=["v4_waitq_tactics_det", "v3_goodwait_read_det"],
        opponents=["v2_ev_base_det", "s3_plus", "s3_rule"],
        resume=False,
    )
    assert manifest["schema"] == "s2.v4"
    assert manifest["games"] == 4
    assert manifest["finished_games"] == 4
    assert manifest["decision_records"] >= 100
    assert len(manifest["shards"]) == 2

    # 1) 验证断点续跑直接复用全部分片
    manifest2 = generate_dataset(
        games=4,
        seed_start=990001,
        shard_games=2,
        out_dir=out_dir,
        workers=1,
        teachers=["v4_waitq_tactics_det", "v3_goodwait_read_det"],
        opponents=["v2_ev_base_det", "s3_plus", "s3_rule"],
        resume=True,
    )
    assert manifest2["dataset_fingerprint"] == manifest["dataset_fingerprint"]

    # 2) 验证 S4 缓存训练脚本能原样加载 manifest 并构建 Belief + Encoded 样本
    cfg = TrainingConfig(
        data_dir=out_dir / "shards",
        manifest=out_dir / "manifest.json",
        output_dir=tmp_path / "s4_out",
    )
    paths, loaded = _load_manifest(cfg)
    assert len(paths) == 2
    assert loaded["games"] == 4

    recs = list(_iter_records(paths))
    assert len(recs) == manifest["decision_records"]
    for rec in recs[:40]:
        sample = build_belief_sample(rec, DatasetBuildConfig(seed=42, degradation_profile="perfect"))
        assert len(sample.encoded.values) == encoding_size()
        idx = action_to_index(_clean_action(rec.action))
        assert idx >= 0

    # 3) 端到端验证 S4 的 _policy_samples() 可直接把本分片转成 PolicySample
    tiny_belief = BeliefNet(BeliefNetConfig(input_size=encoding_size(), hidden_size=32, residual_blocks=1))
    tiny_belief.eval()
    p_samples = _policy_samples(recs[:32], LearnedBelief(model=tiny_belief), seed=42, epoch=0)
    assert len(p_samples) == 32
    assert all(s.legal_mask[s.action_index] for s in p_samples)
