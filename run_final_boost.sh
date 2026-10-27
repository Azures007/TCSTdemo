#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/final_boost

echo "=================================================="
echo "  终极提分：L2 + Amp + Spec(0.5) + Mode D"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    
    # 终极统一架构
    base['training']['use_zscore'] = False
    base['training']['amplitude_channel'] = True
    base['training']['lambda_spec'] = 0.5
    base['training']['ckpt_mode'] = 'D'  # 开启 D 模式，提取 cycles 增益
    
    base['output']['seed'] = 1237
    base['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/final_boost/{ds}'
    yaml.dump(base, open(f'configs/final_boost_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个终极配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (终极版)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/final_boost_${ds}.yaml" 2>&1 | \
        tee "results_ablation/final_boost/log_${ds}.log" | \
        grep -E "P1.2|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 终极架构 提分对比 (Mode A vs Mode D)"
echo "=================================================="
python - << 'PYEOF'
import json, os
datasets = ['houston', 'hyrank', 'shanghai']
print(f"{'Dataset':<12} | {'Mode A (原)' :^20} | {'Mode D (新)' :^20} | {'提升':^10}")
print('-' * 70)
for ds in datasets:
    pA = f'results_ablation/unified_spec_modeA/{ds}/final_metrics.json'
    pD = f'results_ablation/final_boost/{ds}/final_metrics.json'
    oaA = json.load(open(pA))['target_metrics']['OA'] if os.path.exists(pA) else 0
    oaD = json.load(open(pD))['target_metrics']['OA'] if os.path.exists(pD) else 0
    print(f"{ds.upper():<12} | {oaA:>8.2f}            | {oaD:>8.2f}            | {oaD-oaA:+.2f}")
PYEOF
