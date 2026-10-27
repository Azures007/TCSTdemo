#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/Houston_TableV

echo "=================================================="
echo "  Houston TABLE V 格式主组件消融 (seed=1237)"
echo "  强制启用 ckpt_mode=D 以观察组件真实效果"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {'houston': 'configs/hou_H6_ls05.yaml'}
SEED = 1237

variants = {
    # 1. Source Only (纯 warmup，无 cycles)
    'Source_Only': {
        'training': {'num_cycles': 0, 'run_mode': 'warmup_only', 'ckpt_mode': 'A'},
    },
    # 2. + QP Only (有 QP，无 Risk，有 Safe)
    'QP_Only': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': True},
        'tcst': {'use_qp': True, 'use_risk_search': False},
    },
    # 3. + Risk Search Only (无 QP，有 Risk，无 Safe)
    'Risk_Only': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': False},
        'tcst': {'use_qp': False, 'use_risk_search': True},
    },
    # 4. w/o QP (无 QP，有 Risk，有 Safe)
    'w_o_QP': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': True},
        'tcst': {'use_qp': False, 'use_risk_search': True},
    },
    # 5. Full T-CST (有 QP，有 Risk，有 Safe)
    'Full_TCST': {
        'training': {'ckpt_mode': 'D', 'use_safe_training': True},
        'tcst': {'use_qp': True, 'use_risk_search': True},
    }
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for vname, changes in variants.items():
        c = copy.deepcopy(base)
        for sec, kv in changes.items():
            for k, v in kv.items():
                c[sec][k] = v
        c['output']['seed'] = SEED
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/Houston_TableV/{vname}'
        yaml.dump(c, open(f'configs/houston_tableV_{vname}.yaml','w'), sort_keys=False)
print("✅ 5 个配置已生成")
PYEOF

for vname in Source_Only QP_Only Risk_Only w_o_QP Full_TCST; do
    echo ""
    echo "=================================================="
    echo "  RUN: $vname   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/houston_tableV_${vname}.yaml" 2>&1 | \
        tee "results_ablation/Houston_TableV/log_${vname}.log" | \
        grep -E "\[qp\]|\[risk\]|P1.2|OA |AA |Kappa" | tail -6
done

echo ""
echo "=================================================="
echo "  TABLE V: Houston 主组件消融结果"
echo "=================================================="
python - << 'PYEOF' | tee results_ablation/Houston_TableV/SUMMARY.txt
import json, os

variants = [
    ('Source Only',        'Source_Only', '×', '×', '×'),
    ('+ QP Only',          'QP_Only',     '✓', '×', '✓'),
    ('+ Risk Search Only', 'Risk_Only',   '×', '✓', '×'),
    ('w/o QP',             'w_o_QP',      '×', '✓', '✓'),
    ('Full T-CST',         'Full_TCST',   '✓', '✓', '✓')
]

print(f"\n{'Variant':<22} | {'QP':^4} | {'Risk':^4} | {'Safe':^4} | {'Houston OA':^10} | {'AA':^10} | {'Kappa':^10}")
print('-' * 90)

for name, folder, qp, risk, safe in variants:
    p = f'results_ablation/Houston_TableV/{folder}/final_metrics.json'
    if not os.path.exists(p):
        print(f"{name:<22} | {qp:^4} | {risk:^4} | {safe:^4} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    print(f"{name:<22} | {qp:^4} | {risk:^4} | {safe:^4} | {m['OA']:>10.2f} | {m['AA']:>10.2f} | {m['Kappa']:>10.4f}")
print()
PYEOF

