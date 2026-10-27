#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/unified_pipeline

echo "=================================================="
echo "  统一架构测试: L2 + Amp + ckpt_mode=D"
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
    # 统一开启: L2 (use_zscore=False), Amp (amplitude_channel=True), D模式 (ckpt_mode='D')
    base['training']['use_zscore'] = False
    base['training']['amplitude_channel'] = True
    base['training']['ckpt_mode'] = 'D'
    base['output']['seed'] = 1237
    base['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/unified_pipeline/{ds}'
    yaml.dump(base, open(f'configs/unified_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个统一配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (统一 L2+Amp+D)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/unified_${ds}.yaml" 2>&1 | \
        tee "results_ablation/unified_pipeline/log_${ds}.log" | \
        grep -E "P1.2|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 统一架构 (L2+Amp+D) 最终汇总"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston',  'results_ablation/unified_pipeline/houston/final_metrics.json'),
    ('HyRANK',   'results_ablation/unified_pipeline/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/unified_pipeline/shanghai/final_metrics.json')
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
