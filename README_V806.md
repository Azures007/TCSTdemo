# T-CST v8.0.6 (Final, 150/class, 3-seed variance reported)

## Final Results (3 seeds: 1237, 42, 2024)
| Dataset | OA (mean±std) | AA (mean±std) | Best single seed |
|---|---|---|---|
| Houston13->18 | 77.85 ± 2.90 | 70.75 ± 6.36 | 80.78 (seed=1237) |
| Dioni->Loukia | 53.99 ± 9.64 | 51.79 ± 2.08 | 62.25 (seed=1237) |
| Shanghai->Hangzhou | 58.29 ± 25.59 | 60.34 ± 15.30 | 78.27 (seed=1237) |

## Code Structure
- `core/trainer_v8.py`: main training loop (warmup -> cycles)
- `core/spectral_transform.py`: smooth Gain+Offset transform (DCT basis, K=8, 2K=16 params)
- `core/task_geometry.py`: source constraint set + Top-2 pairwise margin
- `core/transport_solver.py`: margin-constrained QP
- `core/risk_controller.py`: Risk Search (V, Vmax, Rsem, swd_rel, spectral TV)
- `core/discrepancy.py`: spectral descriptor + MK-MMD (median heuristic) + SWD
- `core/model.py`: DualReliabilityEncoder backbone
- `core/data.py`: .mat loader, patch extraction, spatial block split
- `core/metrics.py`: OA, AA, Kappa, MacroF1

## Three Datasets
| dataset | source -> target | classes | bands |
|---|---|---|---|
| Houston | Houston13 -> Houston18 | 7 | 48 |
| HyRANK | Dioni -> Loukia | 12 | 176 |
| S->H | Shanghai -> Hangzhou | 3 | 198 |

## What Was Tried and Failed
See `docs/ATTEMPTS.md` for detailed logs.

## Reproduce
python -u scripts/train.py --config configs_FINAL_v806/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v806/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v806/shanghai_final.yaml

## Note on Reproducibility
All numbers are 3-seed mean±std. Single-seed results (especially seed=1237) can be
2-20 points higher than the mean on S->H due to initialization sensitivity (mode collapse).
The v8.0.4 numbers (81.58/58.99/76.26) used latent bugs (optimizer momentum not reset,
spectral TV check reading missing keys, MMD adaptive bandwidth never wired through).
These are fixed in v8.0.6.
