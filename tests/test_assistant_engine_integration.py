"""用真实引擎对局检验:适配器不偷看、张数守恒、胡牌张记录正确,以及上帝视角对账。"""
from dataclasses import replace

from engine.hand import Hand
from engine.tiles import Tile, parse_tile, tile_to_str
from engine.ting_check import ting_tiles
from engine.win_check import can_win

from assistant.adapters.from_engine import build_state
from assistant.analyzer import analyze
from assistant.counting import compute_counts, conservation_errors
from assistant.driver import run_game
from assistant.known_holdings import swap_floors
from assistant.tiles import from_label, tidx

GAMES = range(101, 113)


def _god_hidden_counts(game_state, recorder, perspective):
    """上帝视角:每种牌真实落在"视角玩家看不见的地方"(牌墙 + 别人暗手牌,已胡者不含亮出的胡牌张)的张数。"""
    counts = [0] * 27
    for tile in game_state.wall:
        counts[tile.index] += 1
    wins = {record.seat: record for record in recorder.wins}
    for seat in range(4):
        if seat == perspective:
            continue
        hand = Hand(counts=list(game_state.hands[seat].counts))
        record = wins.get(seat)
        if record is not None and record.self_draw and record.tile is not None:
            hand.remove(parse_tile(record.tile))
        for index, c in enumerate(hand.counts):
            counts[index] += c
    return counts


def _true_hidden(game_state, recorder, seat):
    hand = Hand(counts=list(game_state.hands[seat].counts))
    record = next(r for r in recorder.wins if r.seat == seat)
    if record.self_draw and record.tile is not None:
        hand.remove(parse_tile(record.tile))
    return tuple(hand.counts)


def test_engine_games_conservation_counts_and_win_records():
    for seed in GAMES:
        seen = {"decisions": 0}

        def hook(game, recorder, player, decision):
            state = build_state(game.state, recorder, player)
            seen["decisions"] += 1
            assert conservation_errors(state) == []
            assert sorted(state.own_hand) == sorted(tile_to_str(t) for t in game.state.hands[player].tiles())
            if game.state.phase == "play":
                counts = compute_counts(state)
                assert list(counts.unseen) == _god_hidden_counts(game.state, recorder, player)

        game, recorder = run_game(seed, on_decision=hook)
        assert seen["decisions"] > 10
        for record in recorder.wins:
            if record.tile is None:
                continue
            hand = Hand(counts=list(game.state.hands[record.seat].counts), melds=list(game.state.hands[record.seat].melds))
            if record.self_draw:
                assert can_win(hand, game.state.void_suits[record.seat])
                hand.remove(parse_tile(record.tile))
            assert parse_tile(record.tile) in ting_tiles(hand, game.state.void_suits[record.seat])


def test_adapter_never_reads_hidden_tiles_of_winners_or_opponents():
    captured = []

    def hook(game, recorder, player, decision):
        if any(game.state.won) and not captured:
            captured.append((game, recorder, player))

    for seed in GAMES:
        run_game(seed, on_decision=hook)
        if captured:
            break
    game, recorder, player = captured[0]
    before = build_state(game.state, recorder, player)
    # 打乱所有别人的暗手牌内容(张数不变),视角玩家看到的状态必须完全不变
    for seat in range(4):
        if seat == player:
            continue
        size = game.state.hands[seat].size
        game.state.hands[seat].counts = [0] * 27
        for i in range(size):
            game.state.hands[seat].counts[(i * 7) % 27] += 1
    after = build_state(game.state, recorder, player)
    assert before == after


def test_god_view_audit_zero_false_safe_and_full_winner_recall():
    stats = {"safe_claims": 0, "impossible_shapes": 0, "winner_checks": 0, "floor_checks": 0}

    def hook(game, recorder, player, decision):
        if game.state.phase != "play":
            return
        state = build_state(game.state, recorder, player)
        report = analyze(state)
        # 换给别家、还没出手的牌,上帝视角下确实在他手里
        for seat, floor in swap_floors(state).items():
            truth = _true_hidden(game.state, recorder, seat) if game.state.won[seat] else tuple(game.state.hands[seat].counts)
            assert all(truth[t] >= floor[t] for t in range(27)), (seat, floor)
            stats["floor_checks"] += 1
        for inference in report.winners:
            if inference.complete:
                truth = _true_hidden(game.state, recorder, inference.seat)
                assert any(tuple(int(x) for x in row) == truth for row in inference.candidates)
                stats["winner_checks"] += 1
        for row in report.danger_rows:
            tile = parse_tile(from_label(row["tile"]))
            for entry in row["per_opponent"]:
                seat = entry["seat"]
                actual = game.state.hands[seat]
                if entry["safe"]:
                    stats["safe_claims"] += 1
                    trial = Hand(counts=list(actual.counts), melds=list(actual.melds))
                    if trial.count(tile) < 4:
                        trial.add(tile)
                        assert not can_win(trial, game.state.void_suits[seat]), (row["tile"], seat)
                for shape in entry["shapes"]:
                    if shape["possible"]:
                        continue
                    stats["impossible_shapes"] += 1
                    needed = {}
                    for name in shape["tiles"]:
                        t = tidx(from_label(name))
                        needed[t] = needed.get(t, 0) + 1
                    held = all(actual.counts[t] >= n for t, n in needed.items())
                    void = game.state.void_suits[seat]
                    holds_void = void is not None and any(Tile.from_index(t).suit is void for t in needed)
                    assert not held or holds_void, (row["tile"], seat, shape)

    for seed in GAMES:
        run_game(seed, on_decision=hook)
    assert stats["safe_claims"] > 50 and stats["impossible_shapes"] > 50 and stats["winner_checks"] > 0 and stats["floor_checks"] > 50
