#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/sh_patch1

echo "=================================================="
echo "  S→H Dataset: patch_size = 1 探索实验"
echo "  对比 ckpt_mode=A (标准协议) 与 ckpt_mode=D (T-CST完整) "
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
SEED = 1237

variants = {
    # patch=1 + 标准 ckpt_mode=A
    'patch1_A': {
        'training': {'ckpt_mode': 'A', 'patch_size': 1, 'use_warmup_augmentation': False},
    },
    # patch=1 + ckpt_mode=D (完整 T-CST)
    'patch1_D': {
        'training': {'ckpt_mode': 'D', 'patch_size': 1, 'use_warmup_augmentation': False},
        'tcst': {'use_qp': True, 'use_risk_search': True},
    }
}

for vname, changes in variants.items():
    c = copy.deepcopy(base)
    for sec, kv in changes.items():
        for k, v in kv.items():
            c[sec][k] = v
    c['output']['seed'] = SEED
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/sh_patch1/{vname}'
    yaml.dump(c, open(f'configs/sh_{vname}.yaml','w'), sort_keys=False)
print("✅ 2 个配置已生成")
PYEOF

for vname in patch1_A patch1_D; do
    echo ""
    echo "=================================================="
    echo "  RUN: $vname   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/sh_${vname}.yaml" 2>&1 | \
        tee "results_ablation/sh_patch1/log_${vname}.log" | \
        grep -E "\[qp\]|\[risk\]|P1.2|OA |AA |Kappa" | tail -8
done

echo ""
echo "=================================================="
echo "  S→H patch_size=1 结果汇总"
echo "=================================================="
python - << 'PYEOF'
import json, os

variants = [
    ('Patch=11, Mode=A (原基线)', 'results_baseline_s1/shanghai_s1237/final_metrics.json'),
    ('Patch=1,  Mode=A',          'results_ablation/sh_patch1/patch1_A/final_metrics.json'),
    ('Patch=1,  Mode=D (Full)',   'results_ablation/sh_patch1/patch1_D/final_metrics.json')
]

print(f"\n{'Variant':<26} | {'OA':^8} | {'AA':^8} | {'Kappa':^10} | Per-Class")
print('-' * 100)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<26} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<26} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>10.4f} | {pc}")
print()
PYEOF

