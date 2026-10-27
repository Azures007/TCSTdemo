#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo

echo "=================================================="
echo "  1. 补回缺失的 _apply_source_aug"
echo "  2. 彻底去掉 Houston 的 cb_power"
echo "=================================================="

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 修复 1：把 _apply_source_aug 补回 safe 分支
old_safe = '''            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                _safe_executed = True
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                with _freeze_bn(model):'''
new_safe = '''            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                _safe_executed = True
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)  # <-- 补回这行！
                with _freeze_bn(model):'''
assert old_safe in s, "未找到 safe 分支"
s = s.replace(old_safe, new_safe, 1)

p.write_text(s)
ast.parse(p.read_text())
print("✅ 已修复：_apply_source_aug 已补回 _train_step")
PYEOF

echo ""
echo "=================================================="
echo "  3. 重新跑三大数据集 (精准匹配历史最佳)"
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
    c['training']['per_class_train_cap'] = 150
    
    # 特定数据集最优设定
    if ds == 'houston':
        c['training']['lambda_spec'] = 0.5
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = False
        c['training']['ce_class_balance_power'] = 0.0  # <-- 修复：Houston 不开类别均衡！
    elif ds == 'hyrank':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200
        c['training']['ce_class_balance_power'] = 0.2
    elif ds == 'shanghai':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 5
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200
        c['training']['ce_class_balance_power'] = 0.2

    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_final_best/surgical/{ds}'
    yaml.dump(c, open(f'configs/surgical_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个外科手术级配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (精准修复版)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/surgical_${ds}.yaml" 2>&1 | \
        tee "results_final_best/surgical/log_${ds}.log" | \
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
    p = f'results_final_best/surgical/{ds}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        diff = abs(m['OA'] - float(targets[ds]))
        status = "✅ 完美恢复" if diff < 1.5 else ("🟡 接近" if diff < 3 else "⚠️ 仍有偏差")
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{ds.upper():<12} | {targets[ds]:^12} | {m['OA']:>12.2f} | {status:^10} | {pc}")
PYEOF

