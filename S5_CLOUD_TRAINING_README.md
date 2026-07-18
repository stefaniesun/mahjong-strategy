# S5 上云训练包

本包冻结使用已验收的 `S4/v5_20260718_encoder_v4` belief / policy
checkpoint，不包含体积较大的 S4 原始对局数据。

```bash
unzip s5_cloud_training_package_20260715.zip
cd s5_cloud_training_package_20260715
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

后续命令必须保留 `python -S`。它会在包内任何 Python 代码加载前禁用
`site` / `sitecustomize`；训练入口会拒绝未带 `-S` 的启动。入口仅显式加入
当前解释器（或已激活虚拟环境）的 wheel 目录，不处理可能执行代码的 `.pth`
文件。

## 先做 GPU 烟雾验证

```bash
python -S tools/cloud_train_s5.py --mode smoke --device cuda --output-dir ../s5_cloud_outputs/s5_smoke
```

`smoke` 只运行一次受控的启动诊断，用于确认 CUDA、冻结 S4、PPO、
checkpoint 与报告链路可用。它不是正式训练，也不能作为棋力验收证据。
`--device cuda` 在 CUDA 不可用时会明确失败；只有本地调试才使用
`--device auto`。

运行前，解压包会校验 `s5_cloud_package_manifest.json` 的全部文件哈希；
构包与运行还会校验已验收 S4 checkpoint 的本地 SHA256。不要跳过这些检查。

## 正式 S5 训练

```bash
python -S tools/cloud_train_s5.py --mode train --device cuda \
  --updates 100 --episodes-per-update 32 --arena-games 32 \
  --seed 20260715 --output-dir ../s5_cloud_outputs/s5_train
```

`train` 不使用伪造轨迹或固定胜率：每个 PPO 更新由完整 S1 对局经 S2
观察、冻结 S4 Belief、课程退化和 S3/联赛对手产生；评估在完美与退化两条
赛道都运行真实引擎对局。`--episodes-per-update`、`--arena-games` 与
`--max-game-steps` 都是有界配置，可按云端预算调整。

完成后保留：

```text
cloud_outputs/s5_train/
  checkpoints/latest.pt
  checkpoints/snapshots/s5-step-*.pt
  reports/s5_training_report.json
  reports/s5_training_report.md
  reports/s5_training_report.manifest.json
  s5_cloud_run_manifest.json
```

`s5_cloud_run_manifest.json` 以原子方式在报告对与 checkpoint 都完成哈希后发布。
正式训练仍需使用独立、未见对局进行最终验收，不能只凭训练中 arena 指标发布。

## 本地重建 ZIP

```bash
python -S tools/cloud_train_s5.py --build-package cloud_packages/s5_cloud_training_package_20260715.zip
```

ZIP 使用固定时间戳、排序文件表及 SHA256 manifest；相同源码和 S4 归档输入会
生成相同字节。包中排除 `data/s4_decisions.jsonl`、既有 ZIP、`cloud_outputs/`
与 `__pycache__/`。
