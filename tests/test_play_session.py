"""人机对局会话:能连续打完多局、牌局可原样复现、发给前端的视图不泄露别人的暗牌。"""
import json
import random

from engine.actions import ActionKind
from play.ai import load_ai
from play.session import GameSession, replay_hand
from policies.rule_policy import RulePolicy


def _human_move(session: GameSession, rng: random.Random) -> None:
    decision = session.view(include_report=False)["decision"]
    if decision["swap"]:
        action = rng.choice([a for a in session.human_legal() if a.kind is ActionKind.SWAP_THREE])
        session.act({"kind": "swap_three", "tiles": [f"{t.rank}{t.suit.value}" for t in action.tiles]})
    elif decision["void"]:
        session.act({"kind": "declare_void", "suit": rng.choice(decision["void"])})
    elif decision["buttons"] and rng.random() < 0.5:
        session.act({"index": rng.choice(decision["buttons"])["index"]})
    elif decision["discard"]:
        session.act({"kind": "discard", "tile": rng.choice(decision["discard"])})
    else:
        session.act({"index": decision["buttons"][-1]["index"]})


def test_session_plays_consecutive_hands_and_logs_replayable_records(tmp_path):
    ai, name = load_ai("rule")
    session = GameSession(ai, ai_name=name, log_dir=tmp_path, rng_seed=11)
    rng = random.Random(5)
    for _ in range(3):
        while not session.state.finished:
            session.advance()
            if session.waiting_for_human():
                _human_move(session, rng)
        view = session.view()
        assert sum(seat["score"] for seat in view["seats"]) == 0
        game, _ = replay_hand(session.seed, session.log.dealer, session.log.steps)
        assert game.state.to_dict() == session.state.to_dict()
        session.new_hand()
    assert session.hand_no == 4
    assert len(list(tmp_path.glob("hand*_seed*.json"))) == 3
    assert sum(session.totals) == 0


def test_view_never_contains_other_players_hidden_tiles_before_the_hand_ends(tmp_path):
    session = GameSession(RulePolicy(), ai_name="rule", log_dir=tmp_path, rng_seed=3)
    rng = random.Random(1)
    checked = 0
    while not session.state.finished:
        session.advance()
        view = session.view()
        if not view["finished"]:
            for seat in view["seats"][1:]:
                assert seat["hand"] is None
            checked += 1
        if session.waiting_for_human():
            _human_move(session, rng)
    assert checked > 20
    assert all(seat["hand"] is not None for seat in session.view()["seats"])


def test_mark_saves_note_with_replayable_position(tmp_path):
    session = GameSession(RulePolicy(), ai_name="rule", log_dir=tmp_path, rng_seed=8)
    session.advance()
    session.mark("这里应该提醒换牌")
    entry = json.loads((tmp_path / "marks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["note"] == "这里应该提醒换牌"
    game, _ = replay_hand(entry["replay"]["seed"], entry["replay"]["dealer"], entry["replay"]["steps"])
    assert game.state.phase == session.state.phase
    assert entry["report"] is not None
