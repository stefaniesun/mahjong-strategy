from engine.fan_calc import WinContext, calculate_fan
from engine.hand import Hand
from engine.tiles import parse_tile

from assistant.analyzer import analyze
from assistant.counting import compute_counts, conservation_errors
from assistant.danger import wait_shapes
from assistant.own_hand import analyze_own_hand, my_remaining_draws
from assistant.report import ASSUMPTION_RANDOM, TIER_CERTAIN, TIER_COUNT, TIER_PROBABILITY
from assistant.tiles import label, tidx
from assistant_fixtures import WinRecord, make_state, pong


def _shape_tiles(tile):
    return [(name, sorted(label(t) for t in needed)) for name, needed in wait_shapes(tidx(tile))]


def test_wait_shapes_cover_every_way_a_tile_can_complete_a_hand():
    assert _shape_tiles("1W") == [("单钓/对倒/七对", ["1万"]), ("两面", ["2万", "3万"])]
    assert _shape_tiles("3W") == [("单钓/对倒/七对", ["3万"]), ("边张", ["1万", "2万"]), ("嵌张", ["2万", "4万"]), ("两面", ["4万", "5万"])]
    assert _shape_tiles("7T") == [("单钓/对倒/七对", ["7条"]), ("两面", ["5条", "6条"]), ("嵌张", ["6条", "8条"]), ("边张", ["8条", "9条"])]
    assert _shape_tiles("9B") == [("单钓/对倒/七对", ["9筒"]), ("两面", ["7筒", "8筒"])]


def test_exhausted_neighbours_make_a_tile_100_percent_safe():
    # 2万四张全在牌河;1万牌河 3 张、第 4 张在自己手里 -> 没人能等 1万
    state = make_state(
        ["1W", "5T", "6T", "7T", "2B", "3B", "4B", "6B", "7B", "8B", "9B", "9B", "9T"],
        rivers={1: ["2W", "2W", "1W"], 2: ["2W", "1W"], 3: ["2W", "1W"]},
    )
    assert conservation_errors(state) == []
    report = analyze(state)
    row = next(r for r in report.danger_rows if r["tile"] == "1万")
    assert row["safe_all"]
    titles = [f.title for f in report.sections["danger"] if f.tier == TIER_CERTAIN]
    assert {"1万对下家 100% 安全", "1万对对家 100% 安全", "1万对上家 100% 安全"} <= set(titles)


def test_void_suit_tile_is_safe_against_that_player_only():
    state = make_state(
        ["3W", "5T", "6T", "7T", "2B", "3B", "4B", "6B", "7B", "8B", "9B", "9B", "9T"],
        voids={1: "W", 2: "T", 3: "B"},
    )
    report = analyze(state)
    row = next(r for r in report.danger_rows if r["tile"] == "3万")
    by_seat = {entry["seat"]: entry for entry in row["per_opponent"]}
    assert by_seat[1]["safe"] and by_seat[1]["reason"] == "void"
    assert not by_seat[2]["safe"] and not by_seat[3]["safe"]
    probable = [f for f in report.sections["danger"] if f.tier == TIER_PROBABILITY and f.data.get("tile") == "3万"]
    assert probable and all(f.assumption == ASSUMPTION_RANDOM for f in probable)
    assert all("不等于他已经听牌" in " ".join(f.basis) for f in probable)


def test_tiles_certainly_held_by_a_winner_are_removed_from_other_players_caps():
    # 下家碰三组胡 9条,9条只剩 1 张没出现,所以他手里必然是 9条单钓...
    # 构造:下家暗手牌只剩 1 张,胡 5万 -> 他扣着的那张必然是 5万;5万共现 2 张(+1 胡牌张单独亮出不在牌河)
    state = make_state(
        ["5W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "2B", "3B", "4B"],
        melds={1: [pong("1W"), pong("2W"), pong("3W"), pong("4W")]},
        voids={1: "B"},
        won={1: True},
        hidden={1: 1},
        wins=[WinRecord(1, "5W", True)],
        rivers={2: ["5W"]},
    )
    report = analyze(state)
    (inference,) = report.winners
    assert inference.complete and inference.certain[tidx("5W")] == 1
    # 5万:自己 1 + 牌河 1 + 亮出的胡牌张 1 = 已现 3;剩下 1 张必然在下家手里 -> 对其他人,5万 单钓不可能
    row = next(r for r in report.danger_rows if r["tile"] == "5万")
    for entry in row["per_opponent"]:
        single = next(s for s in entry["shapes"] if s["name"].startswith("单钓"))
        assert not single["possible"]


def test_discard_candidates_count_only_live_tiles():
    # 打 3条 后听 2条/5条;2条已绝 -> 实际只剩 5条的 4 张
    state = make_state(
        ["1B", "1B", "2B", "2B", "3B", "3B", "3T", "3T", "4T", "7B", "7B", "7B", "8B", "8B"],
        rivers={1: ["2T", "2T"], 2: ["2T", "2T"]},
    )
    info, _ = analyze_own_hand(state, compute_counts(state))
    cand = next(c for c in info["candidates"] if c["tile"] == "3条")
    assert cand["shanten_after"] == 0
    assert {w["tile"]: w["remaining"] for w in cand["waits"]} == {"2条": 0, "5条": 4}
    assert cand["live"] == 4


def test_speed_and_fan_criteria_conflict_is_flagged_with_both_accounts():
    # 打 3条:听 2/5条两面,剩 8 张,平胡;打 4条:可听龙七对,3 番,只剩 4 张
    state = make_state(["1B", "1B", "2B", "2B", "3B", "3B", "3T", "3T", "4T", "7B", "7B", "7B", "8B", "8B"])
    info, findings = analyze_own_hand(state, compute_counts(state))
    assert info["candidates"][0]["tile"] == "3条"
    assert info["order_by_fan"][0] == "4条"
    assert info["conflict"] == {"speed": "3条", "fan": "4条"}
    conflict = next(f for f in findings if f.title.startswith("两个标准结论不同"))
    assert conflict.tier == TIER_COUNT
    assert any("3 番" in line for line in conflict.basis)


def test_wait_table_fan_and_scores_match_engine():
    state = make_state(["1B", "1B", "2B", "2B", "3B", "3B", "3T", "3T", "7B", "7B", "7B", "8B", "8B"], current_player=2)
    info, _ = analyze_own_hand(state, compute_counts(state))
    assert info["mode"] == "wait" and info["shanten"] == 0
    for wait in info["waits"]:
        hand = Hand.from_strings(list(state.own_hand))
        tile = {"3条": "3T", "8筒": "8B", "7筒": "7B"}[wait["tile"]]
        hand.add(parse_tile(tile))
        assert wait["fan"] == calculate_fan(hand, WinContext()).fan
        assert wait["tsumo_each"] == calculate_fan(hand, WinContext(self_draw=True)).score_per_payer


def test_remaining_draws_follow_seat_order_and_skip_winners():
    base = dict(own=["1W"] * 0 + ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"])
    state = make_state(base["own"], current_player=3)
    # 牌墙 W 张,从 0 号(自己)开始轮流摸:自己摸第 1、5、9... 张
    expected = (state.wall_count + 3) // 4
    assert my_remaining_draws(state) == expected
    won_state = make_state(base["own"], current_player=3, won={1: True, 2: True}, wins=[WinRecord(1, None, True), WinRecord(2, None, True)], hidden={1: 13, 2: 13})
    assert my_remaining_draws(won_state) == (won_state.wall_count + 1) // 2
