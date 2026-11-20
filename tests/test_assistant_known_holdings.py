"""换三张带来的确定信息:换给别家、还没出手的牌一定还在他手里。"""
import numpy as np

from assistant.analyzer import analyze
from assistant.counting import compute_counts
from assistant.known_holdings import swap_floors
from assistant.report import TIER_CERTAIN
from assistant.state import SwapInfo
from assistant.tiles import tidx
from assistant.winner_inference import enumerate_hidden_hands, infer_winners, STATUS_COMPLETE
from assistant_fixtures import WinRecord, make_state, pong

OWN = ["1T", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "2B", "3B", "4B", "6B"]


def test_floor_subtracts_discards_meld_tiles_from_hand_and_robbed_kong():
    state = make_state(
        OWN,
        rivers={1: ["5W"]},
        melds={1: [pong("7W", from_seat=2)]},
        hidden={1: 10},
        swap=SwapInfo(given=("5W", "5W", "7W"), given_to=1, received=("1B", "2B", "3B"), received_from=3),
    )
    floor = swap_floors(state)[1]
    # 5万:换过去 2 张,打出 1 张 -> 至少还有 1 张;7万:碰时从手里拿出 2 张 -> 下限 0
    assert floor[tidx("5W")] == 1
    assert floor[tidx("7W")] == 0
    assert sum(floor) == 1

    robbed = make_state(
        OWN,
        melds={1: [pong("5W", from_seat=2)]},
        hidden={1: 10},
        wins=[WinRecord(2, "5W", False, 1, robbing_kong=True)],
        won={2: True},
        swap=SwapInfo(given=("5W", "5W", "5W"), given_to=1),
    )
    # 碰用掉 2 张,补杠被抢走 1 张:3 张都已经出手
    assert swap_floors(robbed) == {}


def test_opponent_holding_his_void_suit_tile_cannot_win_so_every_tile_is_safe():
    state = make_state(
        OWN,
        voids={1: "W", 2: "W", 3: "W"},
        swap=SwapInfo(given=("5W", "6W", "9W"), given_to=1),
    )
    report = analyze(state)
    for row in report.danger_rows:
        entry = next(e for e in row["per_opponent"] if e["seat"] == 1)
        assert entry["safe"] and entry["reason"] == "holds_void", row["tile"]
    titles = [f.title for f in report.sections["opening"] if f.tier == TIER_CERTAIN]
    assert any(t.startswith("下家还拿着缺门牌 5万、6万、9万") for t in titles)

    # 他把缺门牌打完以后,这条结论就不再成立
    done = make_state(
        OWN,
        voids={1: "W", 2: "W", 3: "W"},
        rivers={1: ["5W", "6W", "9W"]},
        swap=SwapInfo(given=("5W", "6W", "9W"), given_to=1),
    )
    report = analyze(done)
    assert not any(e["reason"] == "holds_void" for row in report.danger_rows for e in row["per_opponent"])


def test_tiles_known_in_one_hand_block_shapes_for_the_others():
    # 1万:自己 1 张、牌河 2 张,最后 1 张是换给下家、他还没打出的;2万 4 张全在牌河
    # -> 对家和上家都不可能拿着 1万 或 2万,1万对他们形状全堵死
    own = ["1W"] + OWN[1:]
    with_swap = make_state(
        own,
        rivers={2: ["1W", "2W", "2W"], 3: ["1W", "2W", "2W"]},
        swap=SwapInfo(given=("1W", "8W", "9W"), given_to=1),
    )
    row = next(r for r in analyze(with_swap).danger_rows if r["tile"] == "1万")
    by_seat = {e["seat"]: e for e in row["per_opponent"]}
    assert by_seat[2]["safe"] and by_seat[2]["reason"] == "blocked"
    assert by_seat[3]["safe"] and by_seat[3]["reason"] == "blocked"
    single = next(s for s in by_seat[1]["shapes"] if s["name"].startswith("单钓"))
    assert single["possible"] and single.get("known") and single["hold_prob"] == 1.0

    without_swap = make_state(own, rivers={2: ["1W", "2W", "2W"], 3: ["1W", "2W", "2W"]})
    row = next(r for r in analyze(without_swap).danger_rows if r["tile"] == "1万")
    assert not any(e["safe"] for e in row["per_opponent"])


def test_winner_candidates_must_contain_the_tiles_given_to_him():
    unseen = [0] * 27
    for tile, n in {"1W": 2, "2W": 1, "3W": 2, "5W": 1, "9T": 2}.items():
        unseen[tidx(tile)] = n
    floor = [0] * 27
    floor[tidx("3W")] = 2
    status, hands = enumerate_hidden_hands(tidx("1W"), 3, "B", tuple(unseen), floor=tuple(floor))
    assert status == STATUS_COMPLETE
    # 4 种组合里只有"11万对倒 + 33万将"含两张 3万
    assert len(hands) == 1 and hands[0][tidx("3W")] == 2 and hands[0][tidx("1W")] == 2


def test_infer_winners_uses_swap_floor_as_certain_holding():
    state = make_state(
        ["4T", "5T", "6T", "7T", "8T", "9T", "4B", "5B", "6B", "7B", "8B", "9B", "2B"],
        melds={1: [pong("6W"), pong("7W"), pong("8W")]},
        voids={1: "B"},
        won={1: True},
        hidden={1: 4},
        wins=[WinRecord(1, "1W", False, 2)],
        rivers={2: ["1W", "1W", "4W", "4W", "9W", "9W", "9W", "2T", "2T", "2T", "2T", "3T", "3T", "3T", "1T", "1T", "1T", "1T"]},
        swap=SwapInfo(given=("2W", "3W", "5W"), given_to=1),
    )
    floors = swap_floors(state)
    (inference,) = infer_winners(state, compute_counts(state), floors=floors)
    assert inference.complete
    for tile in ("2W", "3W"):
        assert inference.certain[tidx(tile)] >= 1
    assert np.all(inference.candidates[:, tidx("5W")] >= 1)
