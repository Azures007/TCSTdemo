# T-CST v8.0.3 Final (Strict Single-Model)

## Final Results
| Dataset | OA | AA | Kappa | MacroF1 |
|---|---|---|---|---|
| Houston13→18 | 80.81 | 78.84 | 0.6846 | 66.05 |
| Dioni→Loukia | 55.97 | 52.27 | 0.4696 | 44.38 |
| Shanghai→Hangzhou | 80.65 | 80.77 | 0.6554 | 79.71 |

## Configuration
- Label budget: 150 train + 150 val per class (all datasets)
- Split: 10x10 Spatial Block (no patch overlap)
- Checkpoint selection: CB-Val (Class-Balanced Validation)
- T-CST: DCT (K=8), rho=0.15 (Houston) / 0.10, lambda_xi=1.0 (Houston)
- Source Augmentation: ON for Houston/S→H, OFF for HyRANK
- No ensembling, no pseudo-labels, no EMA teacher

## Reproduce
python -u scripts/train.py --config configs_FINAL/houston.yaml
python -u scripts/train.py --config configs_FINAL/hyrank.yaml
python -u scripts/train.py --config configs_FINAL/shanghai.yaml
