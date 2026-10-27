#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_final_best

echo "=================================================="
echo "  验证最终最优基线 (恢复 150-shot 纯净架构)"
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
    
    # 恢复 150-shot 限制
    c['training']['per_class_train_cap'] = 150
    c['training']['use_mixup'] = False
    
    # 统一架构设定
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['ce_class_balance_power'] = 0.2
    
    # 基于之前调优的特定配置
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
        c['training']['patch_size'] = 5   # 之前验证的最优感受野
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200

    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_final_best/{ds}'
    yaml.dump(c, open(f'configs/final_best_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个最终配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (最佳基线验证)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/final_best_${ds}.yaml" 2>&1 | \
        tee "results_final_best/log_${ds}.log" | \
        grep -E "\[Pseudo\]|Phase C|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🏆 最终基线成绩单 (应该恢复为之前的巅峰)"
echo "=================================================="
python - << 'PYEOF'
import json, os
for ds in ['houston', 'hyrank', 'shanghai']:
    p = f'results_final_best/{ds}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        print(f"  {ds.upper():<10} OA = {m['OA']:.2f}%  AA = {m['AA']:.2f}%  Kappa = {m['Kappa']:.4f}  Per-Class: {[round(x,1) for x in m['class_accuracy']]}")
PYEOF

