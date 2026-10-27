#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/l2_with_amp

echo "=================================================="
echo "  L2 Norm + Amplitude Channel (恢复物理能量特征)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    c = copy.deepcopy(base)
    c['training']['use_zscore'] = False
    # 开启模型自带的幅值分支
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'A'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/l2_with_amp/{ds}'
    yaml.dump(c, open(f'configs/l2_amp_{ds}.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

for ds in houston shanghai; do
    echo ""
    echo "  RUN: ${ds} L2 + Amplitude   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "configs/l2_amp_${ds}.yaml" 2>&1 | \
        tee "results_ablation/l2_with_amp/log_${ds}.log" | \
        grep -E "OA |AA |Kappa" | tail -3
done

echo ""
echo "=================================================="
echo "  L2 + Amplitude vs 之前的对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston (L2 原版)', 'results_baseline_s1/houston_s1237/final_metrics.json'),
    ('Houston (L2 + Amp)', 'results_ablation/l2_with_amp/houston/final_metrics.json'),
    ('S→H (L2 原版)', 'results_baseline_s1/shanghai_s1237/final_metrics.json'),
    ('S→H (L2 + Amp)', 'results_ablation/l2_with_amp/shanghai/final_metrics.json'),
]
print(f"{'Variant':<24} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 90)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<24} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<24} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {pc}")
PYEOF

