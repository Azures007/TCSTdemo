#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/more_amp

# 1. HyRANK (L2 + Amp)
python - << 'PYEOF'
import yaml
base = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
base['training']['amplitude_channel'] = True
base['training']['use_zscore'] = False
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/more_amp/hyrank_l2_amp'
yaml.dump(base, open('configs/hyrank_l2_amp.yaml','w'), sort_keys=False)

# 2. S→H (Z-score + Amp)
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
base['training']['amplitude_channel'] = True
base['training']['use_zscore'] = True
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/more_amp/sh_zs_amp'
yaml.dump(base, open('configs/sh_zs_amp.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

echo "=== RUN: HyRANK (L2 + Amp) ==="
python -u scripts/train.py --config configs/hyrank_l2_amp.yaml 2>&1 | grep -E "OA |AA |Kappa" | tail -3

echo ""
echo "=== RUN: S→H (Z-score + Amp) ==="
python -u scripts/train.py --config configs/sh_zs_amp.yaml 2>&1 | grep -E "OA |AA |Kappa" | tail -3
