#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_final_best/reproduce

echo "=================================================="
echo "  重新跑三大数据集：确认全部恢复到巅峰基线"
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
    c = copy.deepcopy(base)
    
    # 核心统一架构
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['ce_class_balance_power'] = 0.2
    c['training']['per_class_train_cap'] = 150
    
    # 特定数据集最优设定
    if ds == 'houston':
        c['training']['lambda_spec'] = 0.5
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = False
    elif ds == 'hyrank':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200
    elif ds == 'shanghai':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 5
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200

    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_final_best/reproduce/{ds}'
    yaml.dump(c, open(f'configs/reproduce_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个最终配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (巅峰基线复现)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/reproduce_${ds}.yaml" 2>&1 | \
        tee "results_final_best/reproduce/log_${ds}.log" | \
        grep -E "CB-selected|Phase C|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🏆 终极复现成绩单 (对比历史巅峰)"
echo "=================================================="
python - << 'PYEOF'
import json, os
targets = {
    'houston': '80.12',
    'hyrank': '58.72',
    'shanghai': '77.02'
}
print(f"{'Dataset':<12} | {'历史巅峰 OA':^12} | {'当前复现 OA':^12} | {'状态':^10} | Per-Class")
print('-' * 90)
for ds in ['houston', 'hyrank', 'shanghai']:
    p = f'results_final_best/reproduce/{ds}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        status = "✅ 恢复" if abs(m['OA'] - float(targets[ds])) < 3 else "⚠️ 偏差大"
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{ds.upper():<12} | {targets[ds]:^12} | {m['OA']:>12.2f} | {status:^10} | {pc}")
PYEOF

