# S4 v5 Belief Bucket Evaluation

## Provenance

- checkpoint: `training_artifacts\S4\v5_20260718_encoder_v4\checkpoints\belief_s4.pt`
- checkpoint SHA256: `caa9775e65070e6196a2b020fc013783bbd818c54f763911286a0165915f3e01`
- encoder version: `s2.v4.encoder.v4`
- source: `deterministic_regeneration`
- validation records: 20000
- validation game-id range: `belief-bucket-20260735..belief-bucket-20263408`
- source games: 3000
- candidate seed range: `20260716..20263715`
- selected validation seeds: 159
- split: cloud `cloud_train_s4_50k_cached._split_name(game_id, seed) == val`
- seed: 20260716
- target validation records: 20000
- profiles: perfect, light_noise, midgame, heavy
- wall buckets: opening >40; midgame 20..40; endgame <20

## Tile log-loss

| Profile | Wall bucket | Samples | Model | Prior | Gain (prior - model) |
| --- | --- | ---: | ---: | ---: | ---: |
| perfect | opening | 9687 | 1.001248 | 1.118587 | 0.117339 |
| perfect | midgame | 7855 | 0.819616 | 1.024122 | 0.204506 |
| perfect | endgame | 2458 | 0.657945 | 0.891477 | 0.233532 |
| light_noise | opening | 9687 | 1.002049 | 1.119416 | 0.117367 |
| light_noise | midgame | 7855 | 0.821858 | 1.024681 | 0.202822 |
| light_noise | endgame | 2458 | 0.661758 | 0.891477 | 0.229719 |
| midgame | opening | 9687 | 1.001244 | 1.118587 | 0.117343 |
| midgame | midgame | 7855 | 0.819626 | 1.024122 | 0.204496 |
| midgame | endgame | 2458 | 0.657691 | 0.891477 | 0.233786 |
| heavy | opening | 9687 | 1.003498 | 1.172211 | 0.168714 |
| heavy | midgame | 7855 | 0.826137 | 1.111617 | 0.285480 |
| heavy | endgame | 2458 | 0.671198 | 1.021786 | 0.350587 |

## Acceptance

**PASS** - all profile/bucket model losses beat prior and every endgame gain exceeds opening gain.
