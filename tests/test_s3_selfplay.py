import pytest

from engine.actions import Action, ActionKind
from policies.base_policy import BasePolicy
from policies.rule_policy import RulePolicy
from selfplay.run_selfplay import (
    SelfplayStats,
    _resolve_pending_discard,
    run_many,
    run_selfplay_game,
    summarize,
)

from state.action_space import action_space_size
from state.protocol import S2ProtocolState



class CapturingPolicy(BasePolicy):
    def __init__(self):
        self.calls = []
        self.delegate = RulePolicy()

    def choose_action(self, protocol_state, legal_mask):
        self.calls.append((protocol_state, list(legal_mask)))
        return self.delegate.choose_action(protocol_state, legal_mask)


class IllegalPolicy(BasePolicy):
    def choose_action(self, protocol_state, legal_mask):
        return Action(ActionKind.DRAW)


def test_selfplay_calls_policies_with_player_protocol_and_fixed_boolean_mask():
    policies = [CapturingPolicy() for _ in range(4)]

    run_selfplay_game(seed=1, max_steps=20, policies=policies)

    assert any(policy.calls for policy in policies)
    for player, policy in enumerate(policies):
        for protocol_state, mask in policy.calls:
            assert isinstance(protocol_state, S2ProtocolState)
            assert protocol_state.perspective_player == player
            assert len(mask) == action_space_size()
            assert all(isinstance(allowed, bool) for allowed in mask)


def test_pending_discard_pass_is_submitted_to_engine_state_machine():
    from engine.game import Game
    from engine.hand import Hand
    from engine.tiles import parse_tile

    game = Game(seed=1)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings(
        "5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split()
    )
    state.hands[1] = Hand.from_strings(
        "9W 9W 1W 2W 3W 1T 2T 3T 1B 2B 3B 4B 5B".split()
    )
    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))

    assert _resolve_pending_discard(
        game,
        [RulePolicy() for _ in range(4)],
    ) == (False, 3)
    assert state.pending_discard is None
    assert state.pending_passers == []




def test_selfplay_rejects_illegal_policy_action_without_replacement():

    with pytest.raises(ValueError, match="fixed action space"):
        run_selfplay_game(
            seed=1,
            max_steps=20,
            policies=[IllegalPolicy(), RulePolicy(), RulePolicy(), RulePolicy()],
        )


def test_rule_selfplay_game_finishes_and_is_zero_sum():

    result = run_selfplay_game(seed=1, max_steps=600)

    assert result.finished is True
    assert sum(result.scores) == 0
    assert result.steps <= 600
    assert 0 <= result.winner_count <= 3
    assert result.drawn is (result.winner_count == 0)


def test_rule_selfplay_many_games_are_reproducible_and_summarized():
    first = run_many(games=5, seed=20, max_steps=600)
    second = run_many(games=5, seed=20, max_steps=600)

    assert first == second
    assert all(result.finished for result in first)
    assert all(sum(result.scores) == 0 for result in first)

    stats = summarize(first)

    assert isinstance(stats, SelfplayStats)
    assert stats.games == 5
    assert stats.unfinished == 0
    assert 0 <= stats.draw_rate <= 1
    assert 0 <= stats.average_winners <= 3
    assert len(stats.average_scores) == 4
    assert sum(stats.score_totals) == 0
    assert sum(stats.winner_count_distribution.values()) == 5
    assert sum(stats.gang_count_distribution.values()) == 5
