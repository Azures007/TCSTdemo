#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/Full8_Other_DS

echo "=================================================="
echo "  HyRANK & S→H 完整 8 种组合消融 (seed=1237)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEED = 1237

combos = [
    ('Source_Only',  'warmup_only', 'A', False, False, False),
    ('Only_Safe',    'full',        'D', False, False, True),
    ('Risk_Only',    'full',        'D', False, True,  False),
    ('Risk_Safe',    'full',        'D', False, True,  True),
    ('QP_Only',      'full',        'D', True,  False, False),
    ('QP_Safe',      'full',        'D', True,  False, True),
    ('QP_Risk',      'full',        'D', True,  True,  False),
    ('Full_TCST',    'full',        'D', True,  True,  True),
]

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for vname, mode, ckpt, use_qp, use_risk, use_safe in combos:
        c = copy.deepcopy(base)
        c['training']['run_mode'] = mode
        c['training']['ckpt_mode'] = ckpt
        c['training']['use_safe_training'] = use_safe
        c['tcst']['use_qp'] = use_qp
        c['tcst']['use_risk_search'] = use_risk
        if mode == 'warmup_only':
            c['training']['num_cycles'] = 0
        c['output']['seed'] = SEED
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/Full8_Other_DS/{ds}/{vname}'
        yaml.dump(c, open(f'configs/full8_{ds}_{vname}.yaml','w'), sort_keys=False)
print("✅ 16 个配置已生成")
PYEOF

for ds in hyrank shanghai; do
    for vname in Source_Only Only_Safe Risk_Only Risk_Safe QP_Only QP_Safe QP_Risk Full_TCST; do
        mkdir -p "results_ablation/Full8_Other_DS/${ds}"
        echo ""
        echo "  RUN: ${ds} / ${vname}   $(date +%H:%M:%S)"
        python -u scripts/train.py --config "configs/full8_${ds}_${vname}.yaml" 2>&1 | \
            tee "results_ablation/Full8_Other_DS/${ds}/log_${vname}.log" | \
            grep -E "OA |AA |Kappa" | tail -3
    done
done

echo ""
echo "=================================================="
echo "  TABLE V: 三数据集完整 8 种全组合消融"
echo "=================================================="
python - << 'PYEOF' | tee results_ablation/Full8_Other_DS/SUMMARY.txt
import json, os

variants = [
    ('Source Only',        'Source_Only'),
    ('+ Safe Only',        'Only_Safe'),
    ('+ Risk Only',        'Risk_Only'),
    ('+ Risk + Safe (w/o QP)', 'Risk_Safe'),
    ('+ QP Only',          'QP_Only'),
    ('+ QP + Safe',        'QP_Safe'),
    ('+ QP + Risk',        'QP_Risk'),
    ('Full T-CST',         'Full_TCST')
]

def get_oa(ds, folder, is_houston):
    if is_houston:
        p = f'results_ablation/Houston_Full8/{folder}/final_metrics.json'
    else:
        p = f'results_ablation/Full8_Other_DS/{ds}/{folder}/final_metrics.json'
    if not os.path.exists(p): return 0.0
    return json.load(open(p))['target_metrics']['OA']

print(f"\n{'Variant':<26} | {'Houston':^10} | {'HyRANK':^10} | {'S→H':^10}")
print('-' * 65)
for name, folder in variants:
    h = get_oa('houston', folder, True)
    hy = get_oa('hyrank', folder, False)
    sh = get_oa('shanghai', folder, False)
    print(f"{name:<26} | {h:>10.2f} | {hy:>10.2f} | {sh:>10.2f}")
print()
PYEOF

