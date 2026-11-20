"""用真实对局检验:任何视角都看不到胡牌者扣着的手牌,牌面守恒,信念标签与可见信息一致。"""
from assistant.driver import run_game
from engine.tiles import tile_to_str
from state.adapters.from_engine import from_engine
from state.protocol import ObservationStatus
from state.tile_belief import _oracle_tile_locations
from state.visibility import hidden_hand_tiles

GAMES = range(301, 316)


def _unknown_total(protocol_state) -> int:
    pool = protocol_state.statistics.unknown_pool_breakdown.value
    return pool["wall"] + sum(pool["opponents"].values())


def test_no_perspective_sees_winner_hands_and_counts_are_conserved():
    stats = {"after_win": 0}

    def hook(game, recorder, player, decision):
        state = game.state
        if state.phase != "play":
            return
        for perspective in range(4):
            protocol_state = from_engine(state, player_id=perspective)
            for view in protocol_state.facts.players.value:
                if view["player_id"] != perspective:
                    assert view["concealed_hand"].status is ObservationStatus.UNKNOWN
            assert sum(protocol_state.facts.seen_counts.value) + _unknown_total(protocol_state) == 108
            # 标签内部会校验"可见 + 藏着 = 4 张",与可见信息口径不一致就会抛错
            _oracle_tile_locations(state, protocol_state)
        if any(state.won):
            stats["after_win"] += 1

    for seed in GAMES:
        run_game(seed, on_decision=hook)
    assert stats["after_win"] > 50


def test_protocol_state_does_not_change_when_winner_hidden_tiles_are_shuffled():
    captured = []

    def hook(game, recorder, player, decision):
        if game.state.phase == "play" and sum(game.state.won) >= 1 and not captured:
            captured.append((game, player))

    for seed in GAMES:
        run_game(seed, on_decision=hook)
        if captured:
            break
    game, player = captured[0]
    state = game.state
    before = [from_engine(state, player_id=p).to_dict() for p in range(4) if not state.won[p]]

    for seat in range(4):
        if not state.won[seat]:
            continue
        # 暗手牌换成同样张数的别的牌;自摸亮出的那张保持不变
        hidden = hidden_hand_tiles(state, seat)
        shown = list(state.hands[seat].tiles())
        for tile in hidden:
            shown.remove(tile)
        state.hands[seat].counts = [0] * 27
        for tile in shown:
            state.hands[seat].add(tile)
        for i in range(len(hidden)):
            state.hands[seat].counts[(i * 5 + 3) % 27] += 1

    after = [from_engine(state, player_id=p).to_dict() for p in range(4) if not state.won[p]]
    assert before == after


def test_engine_win_records_match_publicly_observed_winning_tiles():
    kinds = set()
    for seed in GAMES:
        game, recorder = run_game(seed)
        for public in recorder.wins:
            record = game.state.win_records[public.seat]
            kinds.add(record.kind)
            assert public.tile == (None if record.tile is None else tile_to_str(record.tile))
            assert record.kind == ("self_draw" if public.self_draw else "rob_kong" if public.robbing_kong else "discard")
            assert record.from_player == public.from_seat
    assert {"discard", "self_draw"} <= kinds
