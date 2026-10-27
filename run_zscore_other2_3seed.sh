#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/zscore_other2

echo "=================================================="
echo "  Houston & HyRANK: Z-score 3-seed 稳定性测试"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston': 'configs/hou_H6_ls05.yaml',
    'hyrank':  'configs/hy_Y2_warm40.yaml'
}
SEEDS = [1237, 42, 2024]

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['training']['use_zscore'] = True
        c['training']['ckpt_mode'] = 'A' # 统一标准基线协议
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/zscore_other2/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/zscore_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 6 个配置已生成")
PYEOF

for ds in houston hyrank; do
    for seed in 1237 42 2024; do
        echo ""
        echo "  RUN: ${ds} Z-score seed=${seed}   $(date +%H:%M:%S)"
        python -u scripts/train.py --config "configs/zscore_${ds}_s${seed}.yaml" 2>&1 | \
            tee "results_ablation/zscore_other2/log_${ds}_s${seed}.log" | \
            grep -E "warm [0-9]+/|OA |AA |Kappa" | tail -3
    done
done

echo ""
echo "=================================================="
echo "  Z-score vs L2 (3-seed mean±std) 终极对比"
echo "=================================================="
python - << 'PYEOF'
import json, os, statistics
SEEDS = [1237, 42, 2024]

# 历史 L2 基线数据
L2_BASELINE = {
    'houston': {'OA': 76.12, 'AA': 74.66, 'Kappa': 0.6008},
    'hyrank':  {'OA': 54.28, 'AA': 47.69, 'Kappa': 0.4522},
    'shanghai':{'OA': 63.93, 'AA': 65.14, 'Kappa': 0.3522}
}

print(f"\n{'Dataset / Normalization':<30} | {'OA (mean±std)':^16} | {'AA (mean±std)':^16} | {'Kappa':^8}")
print('-' * 80)

for ds in ['houston', 'hyrank', 'shanghai']:
    # L2 行
    l2 = L2_BASELINE[ds]
    print(f"{ds.upper() + ' (L2 Norm)':<30} | {l2['OA']:>7.2f}±{'?':<6} | {l2['AA']:>7.2f}±{'?':<6} | {l2['Kappa']:>8.4f}")

    # Z-score 行
    if ds == 'shanghai':
        prefix = 'results_ablation/sh_zscore_3seed/s'
    else:
        prefix = f'results_ablation/zscore_other2/{ds}_s'
    
    oas, aas, ks = [], [], []
    for seed in SEEDS:
        p = f'{prefix}{seed}/final_metrics.json'
        if os.path.exists(p):
            m = json.load(open(p))['target_metrics']
            oas.append(m['OA']); aas.append(m['AA']); ks.append(m['Kappa'])
    
    if len(oas) >= 2:
        moa, soa = statistics.mean(oas), statistics.stdev(oas)
        maa, saa = statistics.mean(aas), statistics.stdev(aas)
        mka = statistics.mean(ks)
        print(f"{ds.upper() + ' (Z-score)':<30} | {moa:>7.2f}±{soa:<6.2f} | {maa:>7.2f}±{saa:<6.2f} | {mka:>8.4f}")
    else:
        print(f"{ds.upper() + ' (Z-score)':<30} | (missing)")
    print('-' * 80)

# 详细 per-class 对比
print("\n--- Z-score 详细 Per-Class 结果 (seed=1237) ---")
for ds in ['houston', 'hyrank']:
    p = f'results_ablation/zscore_other2/{ds}_s1237/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        print(f"{ds.upper():<10}: {[round(x,1) for x in m['class_accuracy']]}")
PYEOF

