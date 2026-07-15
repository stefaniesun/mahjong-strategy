# Sichuan Mahjong Engine

四川麻将（血战到底）规则引擎，按 `strategy_S1_engine_spec.md` 分阶段实现。

## 当前阶段

S1：纯规则引擎，不包含机器学习。

## 目标

- 正确实现血战到底规则、番型、杠钱与结算。
- 提供 `reset` / `step` / `legal_actions` / `state` / `run` 接口。
- 为后续 S3/S4/S5 提供训练环境与奖励来源。

## 开发

```powershell
pip install -r requirements.txt
pytest
```
