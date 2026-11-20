"""四川麻将打牌辅助器 v1:只用已知条件做计算,不依赖任何训练产物。"""
from assistant.analyzer import Report, analyze
from assistant.state import AssistantState, MeldView, PlayerView, SwapInfo, WinRecord

__all__ = ["AssistantState", "MeldView", "PlayerView", "Report", "SwapInfo", "WinRecord", "analyze"]
