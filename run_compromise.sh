#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/compromise

echo "=================================================="
echo "  折中方案: cb=0.2, spec=0.4, L2+Amp+D"
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
    c = copy.deepcopy(base)
    
    # 折中统一架构
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['lambda_spec'] = 0.4
    c['training']['ce_class_balance_power'] = 0.2
    
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/compromise/{ds}'
    yaml.dump(c, open(f'configs/comp_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个折中配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (折中版 cb=0.2, spec=0.4)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/comp_${ds}.yaml" 2>&1 | \
        tee "results_ablation/compromise/log_${ds}.log" | \
        grep -E "P1.2|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 历史版本对比 (寻找最佳统一点)"
echo "=================================================="
python - << 'PYEOF'
import json, os
datasets = ['houston', 'hyrank', 'shanghai']
configs = [
    ('终极版 (无cb, spec=0.5)', 'results_ablation/final_boost'),
    ('激进救场 (cb=0.5, spec=0.1)', 'results_ablation/rescue_hyrank/spec0.1_cb0.5'),
    ('折中方案 (cb=0.2, spec=0.4)', 'results_ablation/compromise')
]

print(f"{'Config':<32} | {'Houston':^10} | {'HyRANK':^10} | {'S→H':^10} | {'Mean OA':^10}")
print('-' * 85)
for name, base_dir in configs:
    oas = []
    for ds in datasets:
        p = f'{base_dir}/{ds}/final_metrics.json'
        if os.path.exists(p):
            oas.append(json.load(open(p))['target_metrics']['OA'])
        else:
            oas.append(0.0)
    mean_oa = sum(oas) / 3
    print(f"{name:<32} | {oas[0]:>10.2f} | {oas[1]:>10.2f} | {oas[2]:>10.2f} | {mean_oa:>10.2f}")
PYEOF
