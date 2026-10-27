#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/sh_10percent

echo "=================================================="
echo "  S→H Dataset: 10% 源域数据 + L2 + Amp + Mode D"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))

# 关键修改：放宽样本上限，按 10% 比例抽取
base['training']['source_train_ratio'] = 0.1
base['training']['per_class_train_cap'] = 0  # 0 表示不设固定上限，全按比例走
base['training']['use_zscore'] = False
base['training']['amplitude_channel'] = True
base['training']['ckpt_mode'] = 'D'
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/sh_10percent'

yaml.dump(base, open('configs/sh_10percent.yaml','w'), sort_keys=False)
print("✅ 10% 源域配置已生成")
PYEOF

echo ""
echo "  RUN: S→H (10% 源域)   $(date +%H:%M:%S)"
python -u scripts/train.py --config "configs/sh_10percent.yaml" 2>&1 | \
    tee "results_ablation/sh_10percent/log.txt" | \
    grep -E "source-split|per-class train counts|P1.2|OA |AA |Kappa" | tail -8

echo ""
echo "=================================================="
echo "  S→H 结果对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('S→H (150 shot, L2+Amp+D)', 'results_ablation/unified_pipeline/shanghai/final_metrics.json'),
    ('S→H (10% 源域, L2+Amp+D)', 'results_ablation/sh_10percent/final_metrics.json')
]
print(f"{'Variant':<30} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 95)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<30} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<30} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {pc}")
PYEOF

