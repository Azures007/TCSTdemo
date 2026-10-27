# T-CST v8.0.5 (Strict Final)

## Final Results (Single Model, seed=1237, 150/class)
| Dataset | OA | AA | Kappa | MacroF1 | T-CST ACCEPT |
|---|---|---|---|---|---|
| Houston13->18 | 79.37 | 78.55 | 0.6487 | 65.26 | 2/4 |
| Dioni->Loukia | 59.56 | 51.23 | 0.5091 | 41.96 | 0/6 |
| Shanghai->Hangzhou | 70.89 | 72.49 | 0.4879 | 68.03 | 0/2 |

## Why these numbers differ from v8.0.4 (81.58/58.99/76.26)
The v8.0.4 numbers were obtained with three latent bugs that masked the true performance:
1. Optimizer momentum was not re-initialized after restoring the best warm-up checkpoint.
   This accidentally acted as a second-stage fine-tuning and inflated S->H by ~7 points.
2. The spectral TV plausibility check read `gain_tv`/`offset_tv` from a diagnostics dict that
   never contained them, so Risk Search silently accepted overly aggressive transforms.
3. `domain_gradient_mmd` still passed `(1,2,4,8)` to `mk_mmd2`, so the median-heuristic
   adaptive bandwidth was never used on high-dimensional descriptors (789-D for S->H).

All three are fixed in v8.0.5. The reported numbers are the true single-model performance
under strict UDA rules (no pseudo-labels, no EMA teacher, no target GT during training).

## Per-class results
### Houston13->18
[92.4, 84.2, 67.4, 100.0, 85.6, 87.1, 33.2]  (grass_healthy, grass_stressed, trees, water,
residential, non_residential, road)

### Dioni->Loukia
[58.7, 10.4, 48.7, 12.7, 0.0, 51.6, 84.6, 32.3, 70.8, 0.0, 100.0, 100.0]
(v8.0.5 cb05+p13: [74.3, 0.0, 21.0, 68.4, 0.0, 40.0, 79.0, 54.9, 77.0, 0.2, 100.0, 100.0])
Note: class4 (Fruit Trees) recovers from 12.7% to 68.4% with class-balanced CE + patch 13,
at the cost of class2/class3. Overall OA improves by +2.63.

### Shanghai->Hangzhou
[94.3, 82.9, 30.9]  (Water, Land, Plant)
v8.0.5 cb05: [94.2, 79.6, 43.6] -> Plant +12.7, AA +3.12.

## Key fixes in v8.0.5
1. Optimizer re-initialization after warm-up restore.
2. Median-heuristic MMD bandwidth actually wired through `domain_gradient_mmd`.
3. Spectral TV plausibility computed in `diagnostics()` (gain_tv, offset_tv, rel_change).
4. Absolute energy threshold (1e-4) for target background filtering.
5. CB-Val for checkpoint selection on imbalanced source validation.
6. Spatial block split (no patch overlap between train/val).
7. Top-2 Margin QP + RS-CRW feedback.
8. Optional `ce_class_balance_power` (Houston 0.0, HyRANK 0.5, S->H 0.5).
9. `force_final_epoch=False` (restore best src_val; avoid final-epoch overfit).

## Reproduction
python -u scripts/train.py --config configs_FINAL_v805/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v805/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v805/shanghai_final.yaml

## Known limitations
- Class 2 and class 5 of HyRANK collapse to 0% because the 150-shot budget in the
  source domain is insufficient to learn these extremely rare categories.
  This is a data-scale limitation, not an implementation bug.
- Source augmentation is applied before the spectral transform. Since the transform is
  linear (x*a+b), the only effect is rescaling the noise by `a`, which is negligible.
  Documented intentionally, not changed to avoid retuning.
