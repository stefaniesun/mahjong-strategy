import json
from pathlib import Path

from state.action_space import action_space_size
from state.protocol import S2ProtocolState
from tools.export_s3_replay import export_s3_replay, write_replay_json



def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False






def test_export_s3_replay_contains_initial_steps_and_final_state():
    replay = export_s3_replay(seed=1, max_steps=600, game_id="replay-1")

    assert replay["schema"] == "s3.replay.v1"
    assert replay["meta"]["seed"] == 1
    assert replay["meta"]["game_id"] == "replay-1"
    assert replay["initial_state"]["phase"] == "swap_three"
    assert replay["final_state"]["finished"] is True
    assert replay["result"]["finished"] is True
    assert replay["steps"]
    assert replay["result"]["steps"] == len(replay["steps"])
    assert [step["step"] for step in replay["steps"]] == list(
        range(len(replay["steps"]))
    )

    first_step = replay["steps"][0]

    assert first_step["step"] == 0
    assert first_step["player"] in range(4)
    assert first_step["before"]["phase"] == "swap_three"
    assert first_step["after"]
    assert first_step["protocol_state"]["perspective_player"] == first_step["player"]
    assert first_step["legal_actions"]
    assert first_step["action"] in first_step["legal_actions"]
    for replay_step in replay["steps"]:
        protocol_state = S2ProtocolState.from_dict(
            replay_step["protocol_state"]
        )
        assert protocol_state.perspective_player == replay_step["player"]
        assert len(replay_step["legal_mask"]) == action_space_size()
        assert all(
            isinstance(item, bool)
            for item in replay_step["legal_mask"]
        )

        assert replay_step["action"] in replay_step["legal_actions"]
    assert not _contains_key(
        replay,
        {"winners", "winner_relatives", "pending_winners"},
    )




    for state in [replay["initial_state"], replay["final_state"], first_step["before"], first_step["after"]]:
        assert len(state["players"]) == 4
        assert all("hand" in player for player in state["players"])
        assert all("melds" in player for player in state["players"])
        assert all("river" in player for player in state["players"])
        assert "wall_count" in state
        assert len(state["dice"]) == 2
        assert all(1 <= value <= 6 for value in state["dice"])
        assert state["dice_sum"] == sum(state["dice"])



def test_export_s3_replay_respects_zero_and_one_step_limits():
    empty = export_s3_replay(seed=1, max_steps=0)
    single = export_s3_replay(seed=1, max_steps=1)

    assert empty["result"]["steps"] == 0
    assert empty["steps"] == []
    assert single["result"]["steps"] == 1
    assert [step["step"] for step in single["steps"]] == [0]


def test_export_s3_replay_rejects_invalid_step_limits():
    for invalid in [-1, 1.5, True, "1"]:
        try:
            export_s3_replay(seed=1, max_steps=invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"expected invalid max_steps to fail: {invalid!r}"
            )


def test_export_s3_replay_is_reproducible_for_same_seed():

    first = export_s3_replay(seed=7, max_steps=600, game_id="same")
    second = export_s3_replay(seed=7, max_steps=600, game_id="same")

    assert first == second


def test_write_replay_json_round_trips(tmp_path):
    replay = export_s3_replay(seed=3, max_steps=600, game_id="json")
    output_path = tmp_path / "s3_replay_seed_3.json"

    write_replay_json(replay, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == replay


def test_s3_replay_viewer_html_supports_replay_json_and_training_jsonl():
    viewer = Path("tools/s3_replay_viewer.html")

    html = viewer.read_text(encoding="utf-8")

    assert "S3 Mahjong Replay Viewer" in html
    assert "s3.replay.v1" in html
    assert "jsonl" in html.lower()
    assert "tools/mahjong_tiles" not in html
    assert "GitHub raw" not in html
    assert "图片从本地 mahjong_tiles/ 加载" in html
    assert "const TILE_BASE = 'mahjong_tiles/'" in html

    for suit in ["Man", "Sou", "Pin"]:
        for rank in range(1, 10):
            assert Path(f"tools/mahjong_tiles/{suit}{rank}.png").exists()
    assert Path("tools/mahjong_tiles/Back.png").exists()
    assert "nextStep" in html

    assert "prevStep" in html
    assert "togglePlay" in html
    assert "事件消息" in html
    assert "describeEvent" in html
    assert "牌池更新" in html
    assert "findDrawnTile" in html
    assert "inferDrawEvent" in html
    assert "wall_count" in html
    assert "findTurnDraw" in html
    assert "findLastDrawnTile" in html

    assert "findPlayerLastDraw" in html
    assert "getStepParts" in html
    assert "currentPart" in html
    assert "visibleStateForPart" in html
    assert "function getVisibleRiverUpdate" in html

    assert "displayHandTiles" in html

    assert "removeOneTile" in html


    assert "drawn-tile" in html
    assert "hand-with-draw" in html
    assert "part?.kind === 'draw'" in html
    assert "drawn-tile-gap" in html
    assert "action.kind === 'pass'" in html
    assert "return [{ kind:'draw', label:'摸牌', state:frame.after" in html


    assert "previousFrame" in html
    assert "摸牌" in html
    assert "grid-template-columns:minmax(0,1fr) 320px" in html


    assert "height:calc(100vh - 46px)" in html

    assert "eventTileHtml" in html
    assert "event-tile" in html
    assert "eventLog" in html
    assert "renderEventLog" in html
    assert "event-scroll" in html
    assert "event-entry" in html
    assert "formatDice" in html
    assert "开局" in html
    assert "骰子" in html
    assert "column-reverse" in html
    assert "event-line" in html
    assert "scrollTop = 0" in html

    assert "当前动作" not in html
    assert "合法动作" not in html
    assert "id=\"action\"" not in html
    assert "id=\"legal\"" not in html




