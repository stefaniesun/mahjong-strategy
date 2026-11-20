"""规则参数必须真正生效:引擎结算、过庄比番、观测、合法动作、辅助器都读同一份规则。"""
from assistant.adapters.from_engine import build_state
from assistant.recorder import PublicRecorder
from engine.actions import Action, ActionKind
from engine.config import RuleConfig, active_rule_config
from engine.game import Game
from engine.gang import GangKind
from engine.hand import Hand
from engine.tiles import parse_tile
from state.action_space import legal_mask
from state.adapters.from_engine import from_engine

# 清一色 + 七对 = 4 番:封顶 4 时点炮 16 分,封顶 3 时点炮 8 分
BIG_HAND = "1W 1W 2W 2W 3W 3W 4W 4W 5W 5W 6W 6W 7W".split()


def _ready_game(config: RuleConfig, seed: int = 5) -> Game:
    game = Game(seed=seed, config=config)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("7W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B 2B 3B 4B".split())
    state.hands[1] = Hand.from_strings(BIG_HAND)
    return game


def test_discard_win_is_capped_by_the_configured_max_fan():
    for max_fan, expected in ((3, 8), (4, 16)):
        game = _ready_game(RuleConfig(max_fan=max_fan))
        game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("7W")))
        game.step(1, Action(ActionKind.WIN))
        assert game.state.scores[1] == expected


def test_self_draw_mode_changes_what_each_player_pays():
    # 七对 + 清一色封顶 3 番 = 8 分:加底每家 9;加番已顶番,同样只加底
    # 换成没顶番的 2 番牌才看得出差别:加底每家 5,加番每家 8
    pure = "1W 1W 1W 2W 3W 4W 5W 6W 7W 7W 8W 9W 9W 9W".split()
    for mode, expected_each in (("add_di", 5), ("add_fan", 8)):
        game = Game(seed=4, config=RuleConfig(max_fan=3, self_draw_mode=mode))
        state = game.reset()
        state.phase = "play"
        state.current_player = 0
        state.void_suits = [None, None, None, None]
        state.hands[0] = Hand.from_strings(pure)
        game.step(0, Action(ActionKind.SELF_WIN))
        assert state.scores == [expected_each * 3, -expected_each, -expected_each, -expected_each]


def test_kong_money_scales_with_base_score_and_is_not_capped():
    game = Game(seed=17, config=RuleConfig(base_score=3, max_fan=3))
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.wall = [parse_tile("8B"), parse_tile("9B")]
    state.hands[0] = Hand.from_strings("5W 5W 5W 5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split())
    game.step(0, Action(ActionKind.KONG, tile=parse_tile("5W"), kong_kind=GangKind.CONCEALED))
    assert state.scores == [18, -6, -6, -6]


def test_pass_lock_compares_capped_fan_in_engine_and_protocol():
    # 过庄记的是 3 番;这手 4 番的牌在封顶 3 时只算 3 番,不算"更大",不能胡
    game = _ready_game(RuleConfig(max_fan=3))
    state = game.state
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 3
    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("7W")))
    assert 1 not in state.pending_winners

    # 封顶 4 时是 4 番 > 3 番,可以胡
    game = _ready_game(RuleConfig(max_fan=4))
    state = game.state
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 3
    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("7W")))
    assert 1 in state.pending_winners


def test_observation_and_legal_actions_carry_the_game_rules():
    config = RuleConfig(max_fan=3, self_draw_mode="add_fan")
    game = Game(seed=9, config=config)
    state = game.reset()
    protocol = from_engine(state, player_id=0)
    assert protocol.rule_config.value["max_fan"] == 3
    assert protocol.rule_config.value["self_draw_mode"] == "add_fan"
    # 显式传入的规则优先
    assert from_engine(state, player_id=0, rule_config=RuleConfig(max_fan=2)).rule_config.value["max_fan"] == 2

    # 协议层算出的合法动作要和引擎一致:过庄且封顶后不能胡
    locked = _ready_game(config)
    locked.state.passed_hu_lock[1] = True
    locked.state.passed_fan[1] = 3
    locked.step(0, Action(ActionKind.DISCARD, tile=parse_tile("7W")))
    from state.action_space import index_to_action

    kinds = {index_to_action(i)["kind"] for i, ok in enumerate(legal_mask(from_engine(locked.state, player_id=1))) if ok}
    assert "win" not in kinds


def test_assistant_uses_the_game_rules():
    game = Game(seed=9, config=RuleConfig(max_fan=3))
    game.reset()
    assert build_state(game.state, PublicRecorder(game), 0).rule_config.max_fan == 3


def test_active_rules_come_from_the_file_named_by_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "rules.yaml"
    path.write_text("max_fan: 2\nself_draw_mode: add_fan\n", encoding="utf-8")
    monkeypatch.setenv("MAHJONG_RULES", str(path))
    active_rule_config.cache_clear()
    try:
        config = active_rule_config()
        assert config.max_fan == 2 and config.self_draw_mode == "add_fan"
        assert Game(seed=1).config == config
    finally:
        monkeypatch.undo()
        active_rule_config.cache_clear()
