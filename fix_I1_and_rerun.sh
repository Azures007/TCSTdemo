#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_I1fix_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# ============ 1) 修 I1：raw_gradient 加入 status 白名单 ============
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 找 status 检查行
old = '''            if qp["status"] not in ("optimal", "optimal_inaccurate", "no_constraints") or qp["d_norm"] < 1e-8:'''
new = '''            # I1 fix: raw_gradient 也允许进入 Risk Search（不是拦截）
            _acceptable_status = ("optimal", "optimal_inaccurate", "no_constraints", "raw_gradient")
            if qp["status"] not in _acceptable_status or qp["d_norm"] < 1e-8:'''
assert old in s, "I1 fix: status 检查行未找到"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ I1 fix: raw_gradient 加入 status 白名单")
PYEOF

echo ""
echo "===== 2) 冒烟：I1 no_qp 应该走 Risk Search（4/4 ACCEPT）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['tcst']['use_qp'] = False
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_I1fix_smoke/no_qp'
yaml.dump(c, open('configs/I1fix_no_qp.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_I1fix_smoke/no_qp
python -u scripts/train.py --config configs/I1fix_no_qp.yaml 2>&1 | grep -E "\[qp\]|\[risk\]|OA " | head -6

echo ""
echo "===== 3) 重跑 I1 no_qp（3 数据集 × seed=1237，约 15 分钟）====="
mkdir -p results_ablation/innovation_s1237/I1_no_qp
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    c = copy.deepcopy(base)
    c['tcst']['use_qp'] = False
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/innovation_s1237/I1_no_qp/{ds}'
    yaml.dump(c, open(f'configs/innov_I1_no_qp_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个 I1 配置已生成")
PYEOF

rm -rf results_ablation/innovation_s1237/I1_no_qp/*

for ds in houston hyrank shanghai; do
    echo ""
    echo "  RUN: I1_no_qp_${ds}"
    mkdir -p "results_ablation/innovation_s1237/I1_no_qp"
    python -u scripts/train.py --config "configs/innov_I1_no_qp_${ds}.yaml" 2>&1 | \
        tee "results_ablation/innovation_s1237/I1_no_qp/log_${ds}.log" | \
        grep -E "\[qp\]|\[risk\]|ACCEPT|REJECT|OA " | tail -8
done

echo ""
echo "===== 4) I1 no_qp 重跑结果 ====="
python - << 'PYEOF'
import json, os
print(f"{'dataset':<12} {'I1 no_qp OA':>15} {'R2 OA':>10} {'Δ':>8}")
print('-' * 50)
BASELINE = {'houston': 75.99, 'hyrank': 51.00, 'shanghai': 74.75}
for ds in ['houston', 'hyrank', 'shanghai']:
    p = f'results_ablation/innovation_s1237/I1_no_qp/{ds}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        print(f"  {ds:<10} {m['OA']:>15.2f} {BASELINE[ds]:>10.2f} {m['OA']-BASELINE[ds]:>+8.2f}")
        print(f"    per-class: {[round(x,1) for x in m['class_accuracy']]}")
PYEOF

echo ""
echo "✅ I1 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
