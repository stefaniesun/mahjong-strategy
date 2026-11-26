"""测试 AssistantPolicy 完整全流程闭环(换三张->定缺->碰杠->出牌->结算)及 play.session 接入。"""
from __future__ import annotations

from pathlib import Path

from assistant.bot import AssistantPolicy
from assistant.recorder import PublicRecorder
from engine.game import Game
from play.ai import load_ai
from play.session import GameSession, next_actor


def test_assistant_policy_plays_complete_games_legally_and_fast():
    bot = AssistantPolicy(stochastic=True, seed=123, exact_swap_live=False)
    for seed in (101, 102, 103):
        game = Game(seed=seed, dealer=seed % 4)
        game.reset()
        rec = PublicRecorder(game)
        steps = 0
        while not game.state.finished and steps < 900:
            actor = next_actor(game)
            assert actor is not None
            legal = game.legal_actions(actor)
            act = bot.choose_from_game(game.state, rec, actor, legal)
            assert act in legal
            rec.step(actor, act)
            steps += 1
        assert game.state.finished


def test_play_session_supports_assistant_ai(tmp_path: Path):
    ai, name = load_ai("assistant")
    assert "辅助器" in name
    session = GameSession(ai, ai_name=name, log_dir=tmp_path, rng_seed=7)
    # 推进直到需要人类玩家(换三张)做决策
    session.advance()
    v = session.view()
    assert v["decision"] is not None
    assert v["decision"]["swap"] is True
    assert v["report"] is not None
    assert v["report"]["opening_info"]["swap_options"]
