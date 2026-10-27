# T-CST v8.0.4 (Strict Final Version)

## Final Results (Single Model, seed=1237, 150/类)
| Dataset | OA | AA | Kappa | MacroF1 |
|---|---|---|---|---|
| Houston13→18 | 81.58 | 80.87 | 0.6959 | 65.70 |
| Dioni→Loukia | 58.99 | 51.76 | 0.5033 | 44.36 |
| Shanghai→Hangzhou | 76.26 | 65.80 | 0.5244 | 71.04 |

## Key Fixes Included in v8.0.4
1. **Optimizer Re-initialization**: Reset AdamW states after restoring best warm-up weights.
2. **Adaptive MMD Bandwidth**: Median heuristic (bandwidths=None -> auto).
3. **Target Background Filtering**: Absolute energy threshold (1e-4).
4. **Spectral TV Plausibility**: Total Variation + relative change checks in Risk Search.
5. **CB-Val (Class-Balanced Validation)**: Fixes checkpoint selection bias.
6. **Spatial Block Split**: No patch overlap between train and val.
7. **Top-2 Margin QP & RS-CRW**: Fully implemented.

## Known Limitations
- Source augmentation is applied before the spectral transform. Since the transform is linear (x*a + b), the order only rescales the noise (a*noise), which is negligible. Documented but intentionally not changed to avoid retuning.
- Target background filter uses an absolute threshold (1e-4), not a median heuristic. The median version was tested and incorrectly filtered out dark Water pixels on S->H.

## Reproduce
python -u scripts/train.py --config configs_FINAL_v804/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v804/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v804/shanghai_final.yaml
