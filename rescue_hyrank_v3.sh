#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_final_best/rescue_v3

echo "=================================================="
echo "  HyRANK 救场: 降低伪标签阈值 (thr=0.7)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
c = copy.deepcopy(base)
c['training']['use_zscore'] = False
c['training']['amplitude_channel'] = True
c['training']['ckpt_mode'] = 'D'
c['training']['lambda_spec'] = 0.4
c['training']['ce_class_balance_power'] = 0.2
c['training']['use_pseudo'] = True
c['training']['pseudo_threshold'] = 0.7  # 降低阈值，让极难的小类也能被选进来
c['training']['pseudo_per_class'] = 200
c['training']['patch_size'] = 11
c['output']['seed'] = 1237
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_final_best/rescue_v3/hyrank'
yaml.dump(c, open('configs/rescue_hyrank_v3.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

python -u scripts/train.py --config configs/rescue_hyrank_v3.yaml 2>&1 | \
    tee results_final_best/rescue_v3/log_hyrank.log | \
    grep -E "CB-selected|P1.2|OA |AA |Kappa" | tail -6

