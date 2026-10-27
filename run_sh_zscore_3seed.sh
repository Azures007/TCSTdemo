#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/sh_zscore_3seed

echo "=================================================="
echo "  S→H Z-score 3-seed 稳定性测试"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
SEEDS = [1237, 42, 2024]
for seed in SEEDS:
    c = copy.deepcopy(base)
    c['training']['use_zscore'] = True
    c['training']['ckpt_mode'] = 'A'
    c['output']['seed'] = seed
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/sh_zscore_3seed/s{seed}'
    yaml.dump(c, open(f'configs/sh_zscore_s{seed}.yaml','w'), sort_keys=False)
print("✅ 3 个配置已生成")
PYEOF

for seed in 1237 42 2024; do
    echo ""
    echo "  RUN: S→H Z-score seed=$seed   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "configs/sh_zscore_s${seed}.yaml" 2>&1 | \
        tee "results_ablation/sh_zscore_3seed/log_s${seed}.log" | \
        grep -E "OA |AA |Kappa" | tail -3
done

echo ""
echo "=================================================="
echo "  S→H Z-score 3-seed 汇总"
echo "=================================================="
python - << 'PYEOF'
import json, os, statistics
SEEDS = [1237, 42, 2024]
oas, aas, ks = [], [], []
print(f"{'Seed':<8} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 90)
for seed in SEEDS:
    p = f'results_ablation/sh_zscore_3seed/s{seed}/final_metrics.json'
    if not os.path.exists(p): continue
    m = json.load(open(p))['target_metrics']
    oas.append(m['OA']); aas.append(m['AA']); ks.append(m['Kappa'])
    print(f"{seed:<8} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {[round(x,1) for x in m['class_accuracy']]}")

if len(oas) >= 2:
    print('-' * 90)
    print(f"{'Mean±Std':<8} | {statistics.mean(oas):>5.2f}±{statistics.stdev(oas):<4.2f} | {statistics.mean(aas):>5.2f}±{statistics.stdev(aas):<4.2f} | {statistics.mean(ks):>6.4f}")
PYEOF
