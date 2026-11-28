"""测试策略版本库 (assistant.zoo): 版本自动发现、模块完全隔离、跨版本混战与加载器接入。"""
from __future__ import annotations

from assistant.recorder import PublicRecorder
from assistant.zoo import (
    build_opponent_pool,
    clear_all_zoo_caches,
    get_manifest,
    list_versions,
    load_version,
    parse_zoo_spec,
)
from engine.game import Game
from play.ai import load_ai
from play.session import next_actor


def test_zoo_lists_and_isolates_v2_and_v3_snapshots():
    versions = list_versions()
    assert "v2_ev_base" in versions
    assert "v3_goodwait_read" in versions
    assert "v4_waitq_tactics" in versions
    assert "v5_unified_ev" in versions
    assert "v6_human_tactics" in versions

    m2 = get_manifest("v2_ev_base")
    m3 = get_manifest("v3_goodwait_read")
    m4 = get_manifest("v4_waitq_tactics")
    m5 = get_manifest("v5_unified_ev")
    m6 = get_manifest("v6_human_tactics")
    assert m2["git_commit"] == "8cb0cfc"
    assert m3["git_commit"] == "3f0bcd9"
    assert m4["git_commit"] == "66e601a"
    assert m5["git_commit"] == "0eaef06"
    assert m6["git_commit"] == "481df17"

    p2 = load_version("v2_ev_base", stochastic=False)
    p3 = load_version("v3_goodwait_read", stochastic=False)
    p4 = load_version("v4_waitq_tactics", stochastic=False)
    p5 = load_version("v5_unified_ev", stochastic=False)
    p6 = load_version("v6_human_tactics", stochastic=False)
    # 必须是各自独立冻结的模块类, 互不污染
    assert len({type(p2), type(p3), type(p4), type(p5), type(p6)}) == 5
    assert type(p2).__module__ == "assistant.zoo.v2_ev_base.pkg.bot"
    assert type(p3).__module__ == "assistant.zoo.v3_goodwait_read.pkg.bot"
    assert type(p4).__module__ == "assistant.zoo.v4_waitq_tactics.pkg.bot"
    assert type(p5).__module__ == "assistant.zoo.v5_unified_ev.pkg.bot"
    assert type(p6).__module__ == "assistant.zoo.v6_human_tactics.pkg.bot"


def test_zoo_cross_version_table_plays_legal_complete_games():
    pool = build_opponent_pool(["v4_waitq_tactics", "v5_unified_ev", "v6_human_tactics"], seed=7)
    lineup = [
        pool["v4_waitq_tactics_det"],
        pool["v5_unified_ev_det"],
        pool["v6_human_tactics_det"],
        pool["v6_human_tactics_rand"],
    ]
    for seed in (301, 302):
        game = Game(seed=seed, dealer=seed % 4)
        game.reset()
        rec = PublicRecorder(game)
        steps = 0
        while not game.state.finished and steps < 900:
            actor = next_actor(game)
            assert actor is not None
            legal = game.legal_actions(actor)
            act = lineup[actor].choose_from_game(game.state, rec, actor, legal)
            assert act in legal
            rec.step(actor, act)
            steps += 1
        assert game.state.finished
    clear_all_zoo_caches()


def test_zoo_spec_parsing_and_play_ai_loader():
    assert parse_zoo_spec("v2_ev_base_det") == ("v2_ev_base", False)
    assert parse_zoo_spec("v2_ev_base_rand") == ("v2_ev_base", True)
    assert parse_zoo_spec("zoo:v3_goodwait_read") == ("v3_goodwait_read", True)
    assert parse_zoo_spec("zoo:v3_goodwait_read:det") == ("v3_goodwait_read", False)
    assert parse_zoo_spec("nonexistent_model") is None

    pol, label = load_ai("zoo:v2_ev_base:det")
    assert "v2_ev_base" in label
    assert hasattr(pol, "choose_from_game")
