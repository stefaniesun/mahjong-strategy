# S5 Local v5 Rollout Benchmark

This is bounded pre-training evidence, not formal S5 training or a strength result.

- Command: `python -c "from tools.cloud_train_s5 import run_local_v5_prep_smoke; run_local_v5_prep_smoke()"`
- Seed: `20260718`
- Completed real S1 rollouts: `50`
- Rollout elapsed seconds: `124.545`
- Throughput: `24.088` games/minute
- OS: `Windows-10-10.0.19045-SP0`
- CPU logical cores: `4`
- Python: `3.14.3`
- PyTorch threads: `2`
- v5 belief SHA256: `caa9775e65070e6196a2b020fc013783bbd818c54f763911286a0165915f3e01`
- v5 policy SHA256: `3c23f0b7841298ff0cd9fd531a6f261fdb0a07436b08f9896bf32feb090fd8e2`

The bounded run produced 2,205 learner-only trajectory steps, zero illegal
actions, zero zero-sum failures, 10 finite PPO updates, and a checkpoint that
continued for five additional finite updates.  It stores aggregate health and
throughput evidence only; no raw decision or training-data shard is retained.

## Prepared formal cloud command — human approval required

The following is the prepared first formal S5 training command.  It has **not**
been executed by this preparation work:

```bash
python -S tools/cloud_train_s5.py --mode train --device cuda \
  --updates 100 --episodes-per-update 32 --arena-games 32 \
  --seed 20260718 --output-dir ../s5_cloud_outputs/s5_v5_formal_20260718
```

At the measured local rate of 24.088 complete games/minute, this budget has a
20-hour lower bound: 3,200 rollout games plus 25,600 dual-track arena games
when the league remains at its four initial entries.  If every candidate is
promoted and the eight-entry history fills, it reaches about 77,700 games, or
about 54 local CPU hours.  A CUDA instance should be used, but these are not
GPU speed promises: the real game engine and arena orchestration remain CPU
work.  Run the documented CUDA smoke command first; only a human should approve
this formal run after confirming the instance, budget, and smoke result.
