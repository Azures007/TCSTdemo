#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/Houston_Full8

echo "=================================================="
echo "  只跑缺失的组合: Only_Safe (× × ✓) 和 QP_Risk (✓ ✓ ×)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
SEED = 1237

variants = {
    # 1. Only Safe (无 QP, 无 Risk) 
    'Only_Safe': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': True},
        'tcst': {'use_qp': False, 'use_risk_search': False},
    },
    # 2. QP + Risk (无 Safe，Hard Replace)
    'QP_Risk': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': False},
        'tcst': {'use_qp': True, 'use_risk_search': True},
    }
}

for vname, changes in variants.items():
    c = copy.deepcopy(base)
    for sec, kv in changes.items():
        for k, v in kv.items():
            c[sec][k] = v
    c['output']['seed'] = SEED
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/Houston_Full8/{vname}'
    yaml.dump(c, open(f'configs/houston_full8_{vname}.yaml','w'), sort_keys=False)
print("✅ 2 个缺失配置已生成")
PYEOF

for vname in Only_Safe QP_Risk; do
    echo ""
    echo "=================================================="
    echo "  RUN: $vname   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/houston_full8_${vname}.yaml" 2>&1 | \
        tee "results_ablation/Houston_Full8/log_${vname}.log" | \
        grep -E "\[qp\]|\[risk\]|P1.2|OA |AA |Kappa" | tail -5
done

# 汇总 8 种全排列组合
echo ""
echo "=================================================="
echo "  TABLE V: Houston 完整 8 种全组合消融"
echo "=================================================="
python - << 'PYEOF' | tee results_ablation/Houston_Full8/SUMMARY_FULL8.txt
import json, os

# 定义 8 个变体及其对应的历史存放路径
variants = [
    ('No T-CST (Source Only)', 'Source_Only',  'results_ablation/Houston_TableV/Source_Only',   '×', '×', '×'),
    ('+ Safe Only',            'Only_Safe',    'results_ablation/Houston_Full8/Only_Safe',      '×', '×', '✓'),
    ('+ Risk Only',            'Risk_Only',    'results_ablation/Houston_TableV/Risk_Only',     '×', '✓', '×'),
    ('+ Risk + Safe (w/o QP)', 'w_o_QP',       'results_ablation/Houston_TableV/w_o_QP',        '×', '✓', '✓'),
    ('+ QP Only',              'QP_Only',      'results_ablation/Houston_TableV_Strict/QP_Only','✓', '×', '×'),
    ('+ QP + Safe',            'QP_Safe',      'results_ablation/Houston_TableV/QP_Only',       '✓', '×', '✓'),
    ('+ QP + Risk',            'QP_Risk',      'results_ablation/Houston_Full8/QP_Risk',        '✓', '✓', '×'),
    ('Full T-CST',             'Full_TCST',    'results_ablation/Houston_TableV/Full_TCST',     '✓', '✓', '✓')
]

print(f"\n{'Variant':<26} | {'QP':^3} | {'Risk':^4} | {'Safe':^4} | {'Houston OA':^10} | {'AA':^10} | {'Kappa':^10}")
print('-' * 95)

for name, folder, base_dir, qp, risk, safe in variants:
    p = f'{base_dir}/final_metrics.json'
    if not os.path.exists(p):
        print(f"{name:<26} | {qp:^3} | {risk:^4} | {safe:^4} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    print(f"{name:<26} | {qp:^3} | {risk:^4} | {safe:^4} | {m['OA']:>10.2f} | {m['AA']:>10.2f} | {m['Kappa']:>10.4f}")
print()
PYEOF

