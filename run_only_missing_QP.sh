#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/Houston_TableV_Strict

echo "=================================================="
echo "  仅补跑缺失的严格版 QP Only (无 Safe, Hard Replace)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c = copy.deepcopy(base)
# 强制 ckpt_mode=D，关闭 Safe (Hard Replace)，只开 QP
c['training']['ckpt_mode'] = 'D'
c['training']['use_safe_training'] = False
c['tcst']['use_qp'] = True
c['tcst']['use_risk_search'] = False
c['output']['seed'] = 1237
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/Houston_TableV_Strict/QP_Only'
yaml.dump(c, open('configs/houston_strict_QP_Only.yaml','w'), sort_keys=False)
print("✅ 严格版 QP Only 配置已生成")
PYEOF

echo ""
echo "  RUN: QP_Only (严格无 Safe)   $(date +%H:%M:%S)"
python -u scripts/train.py --config "configs/houston_strict_QP_Only.yaml" 2>&1 | \
    tee "results_ablation/Houston_TableV_Strict/log_QP_Only.log" | \
    grep -E "\[qp\]|\[risk\]|P1.2|OA |AA |Kappa" | tail -6

# 从旧结果目录复用其他已经跑过的变体数据，避免重新训练
echo ""
echo "  复用已有的变体数据..."
cp -r results_ablation/Houston_TableV/Source_Only results_ablation/Houston_TableV_Strict/ 2>/dev/null || true
cp -r results_ablation/Houston_TableV/Risk_Only results_ablation/Houston_TableV_Strict/ 2>/dev/null || true
cp -r results_ablation/Houston_TableV/w_o_QP results_ablation/Houston_TableV_Strict/ 2>/dev/null || true
cp -r results_ablation/Houston_TableV/Full_TCST results_ablation/Houston_TableV_Strict/ 2>/dev/null || true

echo ""
echo "=================================================="
echo "  TABLE V (Strict): Houston 最终严格消融"
echo "=================================================="
python - << 'PYEOF' | tee results_ablation/Houston_TableV_Strict/SUMMARY_FINAL.txt
import json, os

variants = [
    ('Source Only',        'Source_Only', '×', '×', '×'),
    ('+ QP Only',          'QP_Only',     '✓', '×', '×'),
    ('+ Risk Search Only', 'Risk_Only',   '×', '✓', '×'),
    ('w/o QP',             'w_o_QP',      '×', '✓', '✓'),
    ('Full T-CST',         'Full_TCST',   '✓', '✓', '✓')
]

print(f"\n{'Variant':<22} | {'QP':^3} | {'Risk':^4} | {'Safe':^4} | {'Houston OA':^10} | {'AA':^10} | {'Kappa':^10}")
print('-' * 90)

for name, folder, qp, risk, safe in variants:
    p = f'results_ablation/Houston_TableV_Strict/{folder}/final_metrics.json'
    if not os.path.exists(p):
        print(f"{name:<22} | {qp:^3} | {risk:^4} | {safe:^4} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    print(f"{name:<22} | {qp:^3} | {risk:^4} | {safe:^4} | {m['OA']:>10.2f} | {m['AA']:>10.2f} | {m['Kappa']:>10.4f}")
print()
PYEOF

