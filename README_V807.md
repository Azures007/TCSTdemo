# T-CST v8.0.7 (Strict Final, unified ckpt_mode=A, 3-seed mean±std)

## Final Results (seeds: 1237, 42, 2024)
| Dataset | OA (mean±std) | AA (mean±std) | Best single seed |
|---|---|---|---|
| Houston13->18 | 76.12 ± 1.12 | 74.66 ± 4.34 | 77.30 (seed=2024) |
| Dioni->Loukia | 48.95 ± 10.95 | 50.89 ± 3.49 | 58.06 (seed=2024) |
| Shanghai->Hangzhou | 59.19 ± 20.61 | 60.05 ± 8.77 | 73.69 (seed=1237) |

## Checkpoint selection: unified ckpt_mode=A
All three datasets use ckpt_mode=A (source val accuracy).
B (acc+entropy): Houston +1.55 but S->H -15.09 (rejected).
C (class-balanced NLL): HyRANK seed=1237 +7.91 but 3-seed mean 48.88±15.78 (rejected).

## Bugs Fixed (vs v8.0.6)
1. A1: Spatial isolation buffer with val<5 fallback
2. A2: Scheduler re-bound after optimizer reset
3. A5: _make_model reads architecture from cfg
4. A6: Spectral TV thresholds tightened to 0.36/0.18
5. A7: Fusion gate init preserved after _init()
6. A8: Clean Loss uses raw source (not augmented)
7. A10: QP accepts optimal_inaccurate; residual uses xi.value
8. A11: MMD bandwidth uses joint source+target median
9. A12: training_log.jsonl records phi_star_stats/d_star_stats

## Known Limitations
- Houston: source val saturates at ~95%; T-CST cycles discarded by best_src_val
- HyRANK: class2/class5 have <30 source pixels (train 9-10 not 150)
- S->H: 3-class mode collapse; std 20.61 across seeds

## Reproduce
python -u scripts/train.py --config configs_FINAL_v807/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v807/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v807/shanghai_final.yaml
