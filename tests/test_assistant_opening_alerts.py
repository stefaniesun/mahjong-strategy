from itertools import combinations

from engine.tiles import full_wall, tile_to_str

from assistant.analyzer import analyze
from assistant.counting import conservation_errors
from assistant.opening import analyze_opening
from assistant.report import TIER_CERTAIN
from assistant.state import AssistantState, PlayerView
from assistant_fixtures import WinRecord, make_state, pong


def test_void_options_sorted_by_stated_criterion():
    state = make_state(
        ["1W", "9W", "1T", "2T", "3T", "5T", "6T", "1B", "2B", "3B", "5B", "5B", "8B"],
        phase="declare_void",
    )
    info, findings = analyze_opening(state)
    rows = info["void_options"]
    assert [r["suit"] for r in rows][0] == "万"
    keys = [(r["shanten_without"], r["count"]) for r in rows]
    assert keys == sorted(keys)
    assert findings[0].title == "定缺:按标准排第一的是缺万"


def test_swap_options_list_every_legal_same_suit_triple():
    own = ["1W", "2W", "3W", "5W", "1T", "2T", "6T", "1B", "4B", "4B", "7B", "8B", "9B"]
    state = make_state(own, phase="swap_three")
    info, _ = analyze_opening(state)
    expected = 0
    for suit in "WTB":
        tiles = [t for t in own if t.endswith(suit)]
        expected += len(set(combinations(sorted(tiles), 3)))
    assert len(info["swap_options"]) == expected
    keys = [r["total_cost"] for r in info["swap_options"]]
    assert keys == sorted(keys)
    assert all("exp_shanten" in r and "gift_shape" in r and "shanten_rest" in r for r in info["swap_options"])


def test_alerts_void_tiles_pass_lock_and_self_win():
    state = make_state(
        ["1W", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "9T", "2B", "3B", "4B", "5B"],
        voids={0: "W"},
        passed_hu_lock=True,
        passed_fan=1,
    )
    report = analyze(state)
    titles = [f.title for f in report.sections["alerts"]]
    assert "手里还有 1 张缺门牌,打完之前不能胡" in titles
    assert any(t.startswith("过庄中") for t in titles)

    winning = make_state(["1T", "1T", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "9T", "9T", "5T"], voids={0: "W"})
    titles = [f.title for f in analyze(winning).sections["alerts"]]
    assert any(t.startswith("可以自摸") for t in titles)


def test_alerts_can_win_discard_and_pass_lock_blocks_small_win():
    own = ["2T", "3T", "4T", "5T", "6T", "7T", "1B", "2B", "3B", "7B", "7B", "9B", "9B"]
    state = make_state(own, voids={0: "W"}, current_player=2, pending_discard=(2, "9B"), rivers={2: ["9B"]})
    titles = [f.title for f in analyze(state).sections["alerts"]]
    assert any(t.startswith("可以胡9筒") for t in titles)
    locked = make_state(
        own, voids={0: "W"}, current_player=2, pending_discard=(2, "9B"), rivers={2: ["9B"]}, passed_hu_lock=True, passed_fan=3
    )
    titles = [f.title for f in analyze(locked).sections["alerts"]]
    assert any(t.startswith("过庄中不能胡这张") for t in titles)


def test_alerts_pong_and_haidi():
    own = ["5T", "5T", "6T", "7T", "1B", "2B", "3B", "4B", "5B", "6B", "8B", "8B", "9B"]
    state = make_state(own, voids={0: "W"}, current_player=2, pending_discard=(2, "5T"), rivers={2: ["5T"]})
    titles = [f.title for f in analyze(state).sections["alerts"]]
    assert any(t.startswith("碰5条") for t in titles)

    # 构造只剩 1 张牌墙的局面:除了 7 张以外全部现身,三家各扣 2 张暗手牌
    own_late = ["5T", "6T", "7T", "1B", "1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B", "9B"]
    remaining = [tile_to_str(t) for t in full_wall()]
    for tile in own_late:
        remaining.remove(tile)
    visible = remaining[7:]
    late = make_state(
        own_late,
        voids={0: "W"},
        rivers={1: visible[:30], 2: visible[30:60], 3: visible[60:]},
        hidden={1: 2, 2: 2, 3: 2},
        current_player=3,
    )
    assert conservation_errors(late) == []
    assert late.wall_count == 1
    titles = [f.title for f in analyze(late).sections["alerts"]]
    assert "牌墙只剩最后一张:这时不能杠" in titles


def test_conservation_violation_stops_all_conclusions():
    players = tuple(PlayerView(seat=i, hidden_count=0 if i == 0 else 13, river=("5W",) * 2 if i else ()) for i in range(4))
    bad = AssistantState(0, "play", ("5W", "5W", "5W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "9T"), players, wall_count=40)
    report = analyze(bad)
    assert report.errors
    assert list(report.sections) == ["alerts"]
    assert report.sections["alerts"][0].tier == TIER_CERTAIN
    assert "停止输出结论" in report.sections["alerts"][0].title


def test_concealed_kong_alert_counts_only_players_still_in_game():
    state = make_state(
        ["5W", "5W", "5W", "5W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "1B"],
        won={2: True},
        wins=(WinRecord(seat=2, tile=None, self_draw=True, from_seat=None),),
        hidden={2: 14},
    )
    titles = [f.title for f in analyze(state).sections["alerts"]]
    assert any(t.startswith("暗杠5万:仍在场的 2 家各付 2 分") for t in titles), titles
