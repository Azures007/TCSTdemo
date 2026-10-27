# T-CST: Test-time Spectral Transform for Hyperspectral Unsupervised Domain Adaptation

## Baseline Results (150-shot, Single Model, Seed=1237)

| Dataset | OA (%) | AA (%) | Kappa |
|---|---|---|---|
| Houston13→18 | **80.12** | 73.45 | 0.6605 |
| Dioni→Loukia (HyRANK) | **57.54** | 47.61 | 0.4767 |
| Shanghai→Hangzhou | **76.45** | 72.19 | 0.5350 |

## Repository Structure
- `core/`: Core model and training loops (`trainer_v8.py`, `model.py`, etc.)
- `configs/`: Base configurations for each dataset.
- `configs_final_three/`: The exact optimal configs used to achieve the baselines above.
- `scripts/`: Execution scripts (`train.py`).
- `requirements.txt`: Python dependencies.

## How to Run
```bash
python -u scripts/train.py --config configs_final_three/houston.yaml
python -u scripts/train.py --config configs_final_three/hyrank.yaml
python -u scripts/train.py --config configs_final_three/shanghai.yaml
