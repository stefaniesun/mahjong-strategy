import pytest
import yaml

from engine.config import RuleConfig, load_rule_config


def test_load_default_rules_config():
    config = load_rule_config("configs/rules.yaml")

    assert isinstance(config, RuleConfig)
    assert config.base_score == 1
    assert config.max_fan == 4
    assert config.self_draw_mode == "add_di"
    assert config.enable_heavenly_hand is False
    assert config.enable_earthly_hand is False
    assert config.enable_eighteen_arhats is False


def test_reject_invalid_self_draw_mode(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_score": 1,
                "max_fan": 4,
                "self_draw_mode": "invalid",
                "enable_heavenly_hand": False,
                "enable_earthly_hand": False,
                "enable_eighteen_arhats": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="self_draw_mode"):
        load_rule_config(path)


def test_reject_non_positive_scores(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_score": 0,
                "max_fan": 4,
                "self_draw_mode": "add_di",
                "enable_heavenly_hand": False,
                "enable_earthly_hand": False,
                "enable_eighteen_arhats": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="base_score"):
        load_rule_config(path)
