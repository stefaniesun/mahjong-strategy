"""测试使用固定规则,改 configs/rules.yaml 不会让既有用例的分数断言失效。"""
import os
from pathlib import Path

os.environ["MAHJONG_RULES"] = str(Path(__file__).with_name("rules_for_tests.yaml"))
