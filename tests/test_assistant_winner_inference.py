import itertools
import random

import numpy as np

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, Tile
from engine.win_check import can_win

from assistant.counting import compute_counts
from assistant.tiles import tidx
from assistant.winner_inference import (
    STATUS_COMPLETE,
    STATUS_NO_CANDIDATE,
    STATUS_TOO_MANY,
    enumerate_hidden_hands,
    infer_winners,
)
from assistant_fixtures import WinRecord, make_state, pong


def _as_set(arr):
    return {tuple(int(x) for x in row) for row in arr}


def _vec(**counts):
    vec = [0] * 27
    for text, count in counts.items():
        vec[tidx(text)] = count
    return tuple(vec)


def _unseen(mapping):
    vec = [0] * 27
    for text, count in mapping.items():
        vec[tidx(text)] = count
    return tuple(vec)


def _brute_force(win_index, fixed_melds, void, unseen):
    """暴力:枚举所有张数合法的暗手牌,逐个用引擎胡牌判定。"""
    n = 13 - 3 * fixed_melds
    allowed = [t for t in range(27) if unseen[t] > 0 and (void is None or "WTB"[t // 9] != void)]
    dummy = Meld(MeldKind.PONG, (Tile.from_index(26),) * 3, exposed=True)
    found = set()
    for combo in itertools.product(*[range(unseen[t] + 1) for t in allowed]):
        if sum(combo) != n:
            continue
        counts = [0] * 27
        for t, c in zip(allowed, combo):
            counts[t] = c
        hand = Hand(counts=list(counts), melds=[dummy] * fixed_melds)
        if hand.count(Tile.from_index(win_index)) >= 4:
            continue
        hand.add(Tile.from_index(win_index))
        if can_win(hand, Suit(void) if void else None):
            found.add(tuple(counts))
    return found


def test_user_example_win_on_1w_with_three_pongs():
    # 尾盘:碰了三组、缺筒、胡 1万;没出现的只剩 1万×2、2万、3万×2、5万、9条×2
    unseen = _unseen({"1W": 2, "2W": 1, "3W": 2, "5W": 1, "9T": 2})
    status, hands = enumerate_hidden_hands(tidx("1W"), 3, "B", unseen)
    assert status == STATUS_COMPLETE
    expected = {
        _vec(**{"2W": 1, "3W": 1, "9T": 2}),  # 23万等1万(边张)+ 99条将
        _vec(**{"1W": 2, "9T": 2}),  # 11万对倒
        _vec(**{"1W": 2, "3W": 2}),  # 11万对倒,33万将
        _vec(**{"1W": 2, "2W": 1, "3W": 1}),  # 11万将 + 123万
    }
    assert _as_set(hands) == expected


def test_matches_brute_force_on_random_small_positions():
    rng = random.Random(20261103)
    checked = 0
    while checked < 40:
        fixed = rng.choice([0, 1, 2, 3, 4])
        void = rng.choice([None, "W", "T", "B"])
        unseen = [0] * 27
        types = rng.sample(range(27), rng.randint(5, 9))
        for t in types:
            unseen[t] = rng.randint(1, 3)
        allowed = [t for t in types if void is None or "WTB"[t // 9] != void]
        if not allowed or sum(unseen[t] for t in allowed) < 13 - 3 * fixed:
            continue
        size = 1
        for t in allowed:
            size *= unseen[t] + 1
        if size > 150_000:
            continue
        win_index = rng.choice(allowed)
        status, hands = enumerate_hidden_hands(win_index, fixed, void, tuple(unseen), limit=10**7)
        truth = _brute_force(win_index, fixed, void, tuple(unseen))
        assert _as_set(hands) == truth, (fixed, void, unseen, win_index)
        assert (status == STATUS_COMPLETE) == bool(truth)
        checked += 1


def test_candidates_never_contain_void_tiles_and_respect_unseen_counts():
    unseen = _unseen({"1W": 2, "2W": 2, "3W": 2, "4W": 2, "5T": 3, "6T": 2, "7T": 2, "9B": 4})
    status, hands = enumerate_hidden_hands(tidx("5T"), 2, "B", unseen, limit=10**6)
    assert status == STATUS_COMPLETE
    assert not hands[:, 18:].any()
    assert (hands <= np.asarray(unseen)).all()


def test_full_pool_early_game_is_reported_too_many_quickly():
    status, hands = enumerate_hidden_hands(tidx("5T"), 0, "W", tuple([3] * 27))
    assert status == STATUS_TOO_MANY
    assert len(hands) == 0


def test_winning_tile_in_void_suit_has_no_candidate():
    status, _ = enumerate_hidden_hands(tidx("5W"), 0, "W", tuple([2] * 27))
    assert status == STATUS_NO_CANDIDATE


def test_infer_winners_certain_tiles_and_probabilities():
    # 下家碰了三组、缺筒、点炮胡 1万(放炮者是对家)
    state = make_state(
        ["4T", "5T", "6T", "7T", "8T", "9T", "4B", "5B", "6B", "7B", "8B", "9B", "2B"],
        melds={1: [pong("6W"), pong("7W"), pong("8W")]},
        voids={1: "B"},
        won={1: True},
        hidden={1: 4},
        wins=[WinRecord(1, "1W", False, 2)],
        # 放炮的那张 1万 也留在对家牌河里;牌河共 3 张 1万,所以 1万 只剩 1 张没出现
        rivers={2: ["1W", "1W", "1W", "4W", "4W", "9W", "9W", "9W", "2T", "2T", "2T", "2T", "3T", "3T", "3T", "1T", "1T", "1T", "1T"]},
    )
    counts = compute_counts(state)
    (inference,) = infer_winners(state, counts)
    assert inference.complete
    # 1万只剩 1 张没出现:任何组合里 1万最多 1 张
    assert inference.candidates[:, tidx("1W")].max() <= 1
    assert abs(sum(inference.weights) - 1.0) < 1e-9
    for t in range(27):
        assert inference.certain[t] <= inference.expected[t] + 1e-9


def test_joint_filter_keeps_exactly_the_candidates_with_a_compatible_partner():
    # 两名胡牌者都碰了三组;他们的可能组合会争抢稀缺的牌,联合约束要剔除"合起来超量"的组合
    state = make_state(
        ["7B", "7B", "8B", "8B", "9B", "9B", "1B", "1B", "4B", "4B", "5B", "5B", "6B"],
        melds={1: [pong("4T"), pong("5T"), pong("6T")], 3: [pong("7T"), pong("8T"), pong("9T")]},
        voids={1: "B", 3: "B"},
        won={1: True, 3: True},
        hidden={1: 4, 3: 4},
        wins=[WinRecord(1, "2W", False, 2), WinRecord(3, "3W", False, 2)],
        rivers={2: ["2W", "3W", "5W", "5W", "6W", "6W", "7W", "7W", "8W", "8W", "9W", "9W", "1T", "1T", "2T", "2T", "3T", "3T"]},
    )
    counts = compute_counts(state)
    a, b = infer_winners(state, counts)
    assert a.complete and b.complete and a.joint and b.joint
    unseen = np.asarray(counts.unseen)
    _, raw_a = enumerate_hidden_hands(tidx("2W"), 3, "B", counts.unseen)
    _, raw_b = enumerate_hidden_hands(tidx("3W"), 3, "B", counts.unseen)

    def has_partner(x, others):
        return any(((x + y) <= unseen).all() for y in others)

    assert _as_set(a.candidates) == {tuple(int(v) for v in x) for x in raw_a if has_partner(x, raw_b)}
    assert _as_set(b.candidates) == {tuple(int(v) for v in y) for y in raw_b if has_partner(y, raw_a)}
