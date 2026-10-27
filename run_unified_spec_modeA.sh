#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/unified_spec_modeA

echo "=================================================="
echo "  标准统一架构: L2 + Amp + Mode A + Spectral Aux (0.5)"
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
    # 标准架构设定
    base['training']['use_zscore'] = False
    base['training']['amplitude_channel'] = True
    base['training']['ckpt_mode'] = 'A'      # 取消 D 模式，回归 A
    base['training']['lambda_spec'] = 0.5    # 保留光谱辅助损失
    
    base['output']['seed'] = 1237
    base['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/unified_spec_modeA/{ds}'
    yaml.dump(base, open(f'configs/uni_spec_A_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个标准统一配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (L2+Amp+A+Spec)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/uni_spec_A_${ds}.yaml" 2>&1 | \
        tee "results_ablation/unified_spec_modeA/log_${ds}.log" | \
        grep -E "Phase C|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 标准统一基线 (L2+Amp+A+Spec) 汇总"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston',  'results_ablation/unified_spec_modeA/houston/final_metrics.json'),
    ('HyRANK',   'results_ablation/unified_spec_modeA/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/unified_spec_modeA/shanghai/final_metrics.json')
]
print(f"{'Dataset':<12} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 90)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<12} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<12} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {pc}")
PYEOF
