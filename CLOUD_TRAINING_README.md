# S4 云训练包使用说明

本包用于在云端训练 S4 的两个模型：

1. `belief_s4.pt`：从退化/残缺观测推断未知牌分布、听牌概率、弃牌危险度。
2. `policy_s4.pt`：模仿 S3 规则策略的冷启动策略网络。

## 环境

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 冒烟训练

先跑小规模任务，确认云端环境、PyTorch、数据链路都正常：

```bash
python tools/cloud_train_s4.py --output-dir cloud_outputs/s4_smoke --games 2 --max-steps 120 --sample-limit 16 --hidden-size 32 --residual-blocks 1 --batch-size 4 --device auto
```

成功后会生成：

```text
cloud_outputs/s4_smoke/
  data/s4_decisions.jsonl
  checkpoints/belief_s4.pt
  checkpoints/policy_s4.pt
  reports/s4_training_report.json
  reports/s4_training_report.md
```

## 正式训练

默认脚本会在云端用 S3 `RulePolicy` 生成决策记录，再训练 belief 与 policy：

```bash
python tools/cloud_train_s4.py --output-dir cloud_outputs/s4_full --games 1000 --max-steps 1000 --degradation-profile light_noise --hidden-size 128 --residual-blocks 2 --batch-size 128 --learning-rate 0.001 --device cuda
```

如果已有更大的 S3 决策 JSONL 数据，可直接读取：

```bash
python tools/cloud_train_s4.py --input-jsonl /path/to/s4_decisions.jsonl --output-dir cloud_outputs/s4_full --degradation-profile light_noise --hidden-size 128 --residual-blocks 2 --batch-size 128 --device cuda
```

## 输出检查

重点看：

- `reports/s4_training_report.md`
- `reports/s4_training_report.json`
- `checkpoints/belief_s4.pt`
- `checkpoints/policy_s4.pt`

`--device auto`（默认）会优先使用 CUDA，不可用时回退 CPU；`--device cuda` 要求 CUDA 可用，适用于正式云端训练。

报告里会包含：

- 数据条数与 SHA256 指纹
- belief 训练 loss
- belief vs `PriorBelief` 的 tile log-loss
- tenpai / discard danger ECE
- policy top-1 一致率
- policy 非法 argmax 与非法概率质量

## 注意

- 当前脚本是一键云端训练入口，优先保证可复现与链路跑通。
- 大规模正式验收仍建议后续扩展为 train/val/test 按局划分、多 epoch、早停与退化画像混合报告。
- `cloud_outputs/` 是训练产物目录，不在 zip 包内预置。
