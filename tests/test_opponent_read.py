"""测试对手读牌(听牌进度、清一色/金钩钓预警、现物/双筋点炮风险)、4副露确定性安全剪枝及已胡反推联动余牌。"""
from __future__ import annotations

from assistant.analyzer import analyze
from assistant.counting import compute_counts
from assistant.opponent_read import estimate_dealin_prob, read_opponent
from assistant.state import AssistantState, MeldView, PlayerView, SwapInfo, WinRecord
from assistant.tiles import tidx
from assistant_fixtures import make_state, pong


def test_opponent_read_detects_flush_threat_and_genbutsu_safety():
    # 下家(seat=1)缺万, 副露2组条子, 弃了3张筒子+0张条子 -> 清一色条警戒; 且刚打过 5B(近期现物)
    own = "1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 5B 7B 8B".split()
    state = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        melds={1: [pong("2T"), pong("8T")]},
        rivers={1: ["1W", "9W", "2B", "4B", "5B"]},
        hidden={1: 7, 2: 13, 3: 13},
    )
    rep = analyze(state)
    assert not rep.errors
    reads = {r["seat"]: r for r in rep.opponent_reads}
    r1 = reads[1]
    assert r1["flush_suit"] == "条"
    assert r1["flush_level"] == "strong"
    assert any("清一色警戒" in t for t in r1["threats"])

    # 对下家而言: 5B 是近3巡现物 -> 点炮风险极低; 5T 命中清一色条警戒门 -> 点炮风险远高于 5B
    by_tile = {row["tile"]: row for row in rep.danger_rows}
    e_5b = next(e for e in by_tile["5筒"]["per_opponent"] if e["seat"] == 1)
    e_5t = next(e for e in by_tile["5条"]["per_opponent"] if e["seat"] == 1)
    assert e_5b["dealin_prob"] < 0.01
    assert e_5t["dealin_prob"] > e_5b["dealin_prob"] * 10
    assert any("现物" in tag for tag in e_5b["risk_tags"])
    assert any("清一色警戒" in tag for tag in e_5t["risk_tags"])


def test_four_melds_hidden_one_prunes_impossible_two_tile_shapes():
    # 下家(seat=1)有4组副露只剩1张暗牌(金钩钓), 定缺万; 我方手握3张5条+牌河1张5条 -> 外池0张5条!
    # 下家只有1张暗牌, 不可能凑 34T/46T/67T 等2张顺子搭, 且外池0张5条 -> 5条对下家必须判定 100% 安全!
    own = "5T 5T 5T 1B 2B 3B 4B 5B 6B 7B 8B 9B 2B 2B".split()
    state = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        melds={1: [pong("1T"), pong("2T"), pong("8T"), pong("9T")]},
        rivers={1: ["5T", "1W"]},
        hidden={1: 1, 2: 13, 3: 13},
    )
    rep = analyze(state)
    assert not rep.errors
    reads = {r["seat"]: r for r in rep.opponent_reads}
    assert reads[1]["toitoi_alert"] == "金钩钓单吊"
    assert reads[1]["tenpai_prob"] >= 0.90

    row_5t = next(row for row in rep.danger_rows if row["tile"] == "5条")
    e1 = next(e for e in row_5t["per_opponent"] if e["seat"] == 1)
    assert e1["safe"] is True
    assert e1["dealin_prob"] == 0.0


def test_known_void_tile_from_swap_sets_tenpai_prob_zero():
    own = "1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 5B 7B 8B".split()
    swap = SwapInfo(given=("1W", "5W", "9W"), given_to=1, received=("2T", "5T", "8T"), received_from=3)
    state = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        rivers={1: ["1W", "5W"]},  # 还剩 9W 没打出
        swap=swap,
    )
    rep = analyze(state)
    assert not rep.errors
    r1 = next(r for r in rep.opponent_reads if r["seat"] == 1)
    assert r1["holds_known_void"] is True
    assert r1["tenpai_prob"] == 0.0


def test_own_hand_deducts_winner_certain_tiles_for_dead_wait():
    # 我方听单钓 9筒(牌面看到 3 张 9筒: 我手里0张? 不, 牌河有3张9筒 -> 外池未现 1 张9筒),
    # 已胡的下家(seat=1)副露4组(剩1张暗牌)且自摸胡 9筒 -> 下家必然锁死这最后 1 张9筒!
    # 联动 winner_inference 后, 我方听 9筒 的 remaining 必须为 0 且判定为实质死听!
    own = "1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 4B 9B".split()
    state = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "T"},
        melds={1: [pong("1B"), pong("2B"), pong("3B"), pong("5B")]},
        rivers={2: ["9B"]},  # 我手里1张9B + 牌河1张9B + 下家自摸亮出1张9B + 下家扣着1张9B凑对子 = 4张9B!
        won={1: True},
        wins=(WinRecord(seat=1, tile="9B", self_draw=True, from_seat=None),),
        hidden={1: 1, 2: 13, 3: 13},
    )
    rep = analyze(state)
    assert not rep.errors
    assert rep.own_info["winners_locked_applied"] is True
    w9b = next(w for w in rep.own_info["waits"] if w["tile"] == "9筒")
    assert w9b["raw_unseen"] == 1
    assert w9b["remaining"] == 0
    titles = [f.title for f in rep.sections["own_hand"]]
    assert any("死听" in t for t in titles)


def test_defense_conflict_surfaced_when_fastest_discard_is_high_risk():
    # 下家(seat=1)强清一色条且快听(2副露+多弃筒), 我方手牌打 5T(清一色高危) vs 打 5B(近3巡现物极安全) 同向听
    own = "1T 2T 3T 5T 1B 2B 3B 4B 5B 6B 7B 8B 9B 5B".split()
    state = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        melds={1: [pong("2T"), pong("8T")]},
        rivers={1: ["1W", "9W", "1B", "3B", "4B", "6B", "7B", "5B"]},
        hidden={1: 7, 2: 13, 3: 13},
    )
    rep = analyze(state)
    assert not rep.errors
    assert rep.own_info["mode"] == "discard"
    by_tile = {c["tile"]: c for c in rep.own_info["candidates"]}
    assert by_tile["5条"]["dealin_prob"] > by_tile["5筒"]["dealin_prob"]
