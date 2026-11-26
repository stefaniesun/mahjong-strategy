# 换后定缺纠结局大样本模拟任务（void_v2 → void_v3）

## 一、任务目的

本任务用于在多核 CPU 主机上，采集**换三张完成后进入“定缺阶段”的纠结局大样本**（自动跳过 `1/5/7` 等闭眼缺单门的送分局），重点解决两个问题：

1. **纯净“定缺后那一刻”供需模型（`--mode moment`，完全剥离后期出牌 AI 噪声）**：
   - 在四家刚刚定缺落地的瞬间，统计三种可选定缺方案下的：
     - **己方可用总供牌**：牌墙真残张 + 正在缺该门的对手手里**开局必打喂牌张数**；
     - **换牌流向信号**：`4/4/5` **撞车（换出==换入）** 与 **非撞车** 下，给牌方/收牌方各自定缺该花色的真实概率与“单行道（3人同缺放任1家独吃）”风险；
     - **`dir=2`（对家互换）与 `dir=±1`（顺逆单向传牌）** 拆分统计（消除之前小样本的抽样误差）。
2. **完整配对全打验证（`--mode full`）**：
   - 对每局换后牌面锁死四家手牌、换牌流向与牌墙，配对全打 `缺万 / 缺条 / 缺筒` 三遍（全部用 `RulePlusPolicy("all")`），输出带 `±95%` 置信区间的净得分、重赔率（`<= -6` 分）与大赢率（`>= +6` 分）。

覆盖 4 类核心纠结桶：
- `4/4/5_crash`：闲家换后 `4/4/5` 且换牌撞车
- `4/4/5_noncrash`：闲家换后 `4/4/5` 且非撞车
- `4/5/5_dealer`：庄家换后 `4/5/5` 或 `4/4/6`
- `3_close`：换后 `3/4/6` 或 `3/5/5` 且缺长门向听不输于缺 3 张短门

---

## 二、拉取代码与冒烟自检（约 5~10 秒）

```powershell
git pull
python -m pytest tests/test_swap_v2.py -q
python tools/run_void_eval.py --mode moment --deals-per-bucket 20 --workers 4 --out outputs/void_moment_check.json
```

看到测试通过且十几秒内打印出完整 3 节统计表，即说明环境正常。

---

## 三、正式大样本运行命令

> 默认自动使用 `逻辑CPU核数 - 1`；如 16 核机器可按需追加 `--workers 14`（或 `16`）。

### 推荐档位 A：纯净“定缺后那一刻”超大样本（先跑这档，约 5~12 分钟）

总共采集 `4,000` 局高价值纠结定缺局面（每桶 1,000 局），把之前 `n=41` 的非撞车细分桶和 `dir=2 / ±1` 全部拉到数百局以上，**零后期出牌噪声**：

```powershell
python -u tools/run_void_eval.py --mode moment --deals-per-bucket 1000 --resume --out outputs/void_moment_large.json
```

*(提示：因为 `4/4/5_noncrash` 在 `swap_v2` 换牌下本身出现概率较低，若希望先极速出全量、对非撞车 `400` 局即够用，可加 `--noncrash-quota 400`。)*

### 推荐档位 B：完整配对全打大样本（约 25~45 分钟）

在采集定缺瞬间供需的同时，每一局配对打完 `缺万 / 缺条 / 缺筒` 共 `3 × 2,000 = 6,000` 场完整牌局，用于最终校准 `void_v3` 并压低得分置信区间：

```powershell
python -u tools/run_void_eval.py --mode full --deals-per-bucket 500 --noncrash-quota 250 --resume --out outputs/void_full_large.json
```

---

## 四、断点续跑与结果回传

1. 脚本每采集完一轮会自动保存断点到 `<out>.ckpt.json`，中途关闭或断电后 **原命令重新执行（带 `--resume`）** 即可续跑。
2. 跑完后通过 Git 带回结果文件（`outputs/` 默认在 `.gitignore` 中，需加 `-f`）：

```powershell
git add -f outputs/void_moment_large.json
# 如果也跑了 full 档，把 outputs/void_full_large.json 一并加上
git commit -m "data: add large-scale post-swap void evaluation dataset"
git push
```
