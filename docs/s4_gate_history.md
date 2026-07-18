# S4 Gate History

This record preserves the S4 iteration gate that authorizes S5 preparation.
The comparable arena uses 500 games with the same seed schedule: the model is
seat 0 and the other three seats use the rule policy.

| Version | Recipe | Gate score (points/game) | Gate result |
| --- | --- | ---: | --- |
| v1 | 1,000 games | -2.56 | failed |
| v2 | 50,000 games + encoder.v3 (opponent discards and exposed melds) | -2.47 | failed |
| v3 | v2 + DAgger round 1 | -1.30 | failed |
| v4 | v3 + DAgger round 2 | -1.26 | failed |
| v5 | v4 + encoder.v4 candidate-discard features | -0.066 +/- 0.42 | accepted |

## Causal diagnostics

The investigation followed the causal sequence **data volume -> distribution shift -> representation capability**.

1. Increasing data volume from v1 to v2 improved supervision coverage but did
   not remove the large arena deficit.
2. DAgger in v3 and v4 reduced the distribution shift between expert-labelled
   states and model-encountered states, producing a material but incomplete
   recovery.
3. The remaining limitation was representation capability: encoder.v4 adds
   candidate-discard features, allowing the policy to distinguish the relevant
   decision contexts. v5 then reached the accepted gate result.

The v5 result, **-0.066 +/- 0.42 points/game**, is statistically
indistinguishable from the S3 baseline of -0.044 on this gate. It is therefore
accepted as the frozen S4 seed for S5 preparation, not as evidence of a
separate performance claim beyond that gate.

## Reproduction

Run from the repository root:

```powershell
python -m learning.eval.arena --seed 90000 --games 500 --model-seat 0 --policy-checkpoint training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/policy_s4.pt --opponent rule --opponent rule --opponent rule
```
