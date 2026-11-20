# 打牌辅助器 v1

只用牌桌上已知的条件做计算,不加载任何模型,不依赖训练产物。规格见 `strategy_assistant_v1_spec.md`,对账结果见 `docs/assistant_v1_audit.md`。

## 怎么用

```bash
# 生成一局回放网页(双击打开,逐手翻看辅助面板)
python tools/assistant_replay.py --seed 7 --player 0

# 500 局上帝视角对账,输出 docs/assistant_v1_audit.md
python tools/assistant_audit.py --games 500 --workers 3

# 测试
python -m pytest tests/test_assistant_*.py -q
```

代码里直接调用:

```python
from assistant import analyze
report = analyze(state)          # state 是 AssistantState(玩家视角的已知信息)
report.to_dict()                 # 五块面板 + 每条结论的档位、依据、假设
```

## 输出三档

| 档位 | 含义 |
|---|---|
| 确定 | 由规则和已知牌必然推出,铁定成立(如"100% 安全""已绝""死听") |
| 计数 | 精确数出来的数字或集合(如"还剩 2 张""只可能是这 4 种组合") |
| 概率 | 在写明的假设下精确算出;统一假设为"未出现的牌随机分布在牌墙和各家暗手牌中" |

牌面不守恒(某牌超过 4 张、总数对不上)时只报警,不输出任何结论。

## 模块

| 文件 | 作用 |
|---|---|
| `state.py` | 输入状态:玩家视角下的已知信息 |
| `counting.py` | 已现/未现/绝张/未知区规模,守恒校验 |
| `winner_inference.py` | 已胡者手牌反推(按花色拆开枚举,numpy 拼接) |
| `known_holdings.py` | 换三张:你换给别家、还没出手的牌一定在他手里(安全度、反推、余牌都用上) |
| `danger.py` | 对每家在场对手的安全度与等待形状 |
| `own_hand.py` | 向听、候选弃牌账目、听牌账目、摸到进张的概率 |
| `opening.py` | 换三张、定缺、换牌去向 |
| `alerts.py` | 过庄、缺门、能否胡、碰杠的账、海底、流局后果 |
| `analyzer.py` | 总入口 `analyze()` |
| `recorder.py` / `adapters/from_engine.py` / `driver.py` | 用引擎对局驱动辅助器(记录胡牌张;屏蔽胡牌者手牌) |
| `viewer/template.html` | 回放网页模板 |

## 已知限制

- **输入层没做**:目前只能用引擎对局驱动。手动录入、截屏或拍照识别另行决定,接入时构造 `AssistantState` 即可。
- **已胡者反推有上限**:单人可能组合超过 2 万种时不展开(多见于中盘、胡牌者没有副露),只给按非缺门花色均摊的期望;两人联合约束的组合对超过 40 万时,各自独立计算并标明"未做联合约束"。
- **"对手手里有搭子"的概率系统性偏低**:500 局对账显示低估约 10~13 个百分点,因为对手会主动留搭子、打孤张,"随机分布"假设在这里不成立。界面已写明它不等于听牌或点炮概率。
- 暗杠牌面按项目约定对所有人可见。
