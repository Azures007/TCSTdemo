# T-CST v8.0.8 (Strict Final, B1-B4 fixes, split=v8.0.7)

## Final Results (3 seeds: 1237, 42, 2024, unified ckpt_mode=A, projection_norm=batch)
| Dataset | OA (mean±std) | AA (mean±std) | Best single seed |
|---|---|---|---|
| Houston13->18 | 76.12 ± 1.12 | 74.66 ± 4.34 | 77.30 (seed=2024) |
| Dioni->Loukia | 49.31 ± 2.83 | 48.30 ± 3.88 | 51.30 (seed=1237) |
| Shanghai->Hangzhou | 63.93 ± 10.11 | 65.14 ± 6.63 | 74.75 (seed=1237) |

## What changed from v8.0.7
4 bug fixes were kept (B1-B4), and the v3/v3c split changes were rolled back:
- B1: Risk Search now passes source labels to transform (class-conditional consistent)
- B2: Odd constraint samples no longer dropped (n=1 -> 1 sample, was 0)
- B3: Full phi_star/d_star vectors saved in training_log.jsonl
- B4: SWD uses all calibration samples (was first 64; HyRANK 44% -> 100%)
- B5-B8 (v3/v3c split): ROLLED BACK. Split returns to v8.0.7 mixed-mode.
  Reason: v3 block-based split helps S->H (+16.31) but destroys HyRANK (-25.41)
  because HyRANK's rare classes have <2 spatial blocks. A dataset-agnostic split
  is required for generalization to unseen datasets.

## Ablation: what did NOT help
- ckpt_mode=B (acc+target entropy): Houston +1.55, S->H -15.09
- ckpt_mode=C (class-balanced NLL): HyRANK seed=1237 +7.91, but 3-seed std 15.78
- LayerNorm (D2): Houston +1.01/-2.94, HyRANK -9.78/-19.43, S->H -11.40/-8.83
- EMA teacher: identical to baseline (discarded by best_src_val)
- Pseudo-labels: HyRANK -6.93 (class2/class5 teacher confidence is wrong)
- Class-conditional T-CST: identical to baseline (discarded by best_src_val)
- MMD bandwidth sweep, lambda_d 100x, Focal Loss, 100/class: all negative or neutral

## Key improvement: S->H 59.19 -> 63.93, std 20.61 -> 10.11
B2 (odd constraint) + B4 (full SWD) stabilize S->H training significantly.
seed=42: 35.59 -> 54.72 (+19.13). This is the only dataset with real gain.

## Known Limitations
- Houston: src_val saturates at ~95%; T-CST cycles discarded by best_src_val
- HyRANK: class2/class5 have <30 source pixels (train 51-60 not 150)
- S->H: 3-class mode collapse remains; std 10.11 across seeds (was 20.61)

## Reproduce
python -u scripts/train.py --config configs_FINAL_v808/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v808/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v808/shanghai_final.yaml
