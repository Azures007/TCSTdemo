#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_tta_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# 1) 升级 TTA 逻辑为 6 路
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

old_tta = '''            if use_tta:
                logits = model(xb).logits
                logits = logits + model(torch.flip(xb, dims=(-1,))).logits
                logits = logits + model(torch.flip(xb, dims=(-2,))).logits
                logits = logits / 3.0
                preds.append(logits.argmax(1).cpu().numpy())'''
new_tta = '''            if use_tta:
                # 6路 TTA：原图、水平翻转、垂直翻转、旋转90、180、270
                outs = [
                    model(xb).logits,
                    model(torch.flip(xb, dims=(-1,))).logits,
                    model(torch.flip(xb, dims=(-2,))).logits,
                    model(torch.rot90(xb, k=1, dims=(-2,-1))).logits,
                    model(torch.rot90(xb, k=2, dims=(-2,-1))).logits,
                    model(torch.rot90(xb, k=3, dims=(-2,-1))).logits,
                ]
                logits = torch.stack(outs, dim=0).mean(dim=0)
                preds.append(logits.argmax(1).cpu().numpy())'''
assert old_tta in s, "旧 TTA 代码未找到"
s = s.replace(old_tta, new_tta, 1)
p.write_text(s)
ast.parse(s)
print("✅ trainer_v8.py 已升级为 6 路 TTA")
PYEOF

# 2) 配置生成（统一架构）
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
    
    # 终极统一架构（包含之前的均衡伪标签）
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['lambda_spec'] = 0.4
    c['training']['ce_class_balance_power'] = 0.2
    
    c['training']['use_pseudo'] = True
    c['training']['pseudo_threshold'] = 0.8
    c['training']['pseudo_per_class'] = 200
    c['training']['pseudo_weight'] = 0.5
    
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/tta_boost/{ds}'
    yaml.dump(c, open(f'configs/tta_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个 TTA 配置已生成")
PYEOF

mkdir -p results_ablation/tta_boost
for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (终极架构 + 6路 TTA)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/tta_${ds}.yaml" 2>&1 | \
        tee "results_ablation/tta_boost/log_${ds}.log" | \
        grep -E "\[Pseudo\]|Phase C|OA |AA |Kappa" | tail -6
done

echo ""
echo "=================================================="
echo "  🚀 6路 TTA 提分效果对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston',  'results_ablation/final_boost/houston/final_metrics.json', 'results_ablation/tta_boost/houston/final_metrics.json'),
    ('HyRANK',   'results_ablation/cb_pseudo/hyrank/final_metrics.json',    'results_ablation/tta_boost/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/cb_pseudo/shanghai/final_metrics.json',  'results_ablation/tta_boost/shanghai/final_metrics.json')
]
print(f"{'Dataset':<10} | {'原基线 OA':^10} | {'6路 TTA OA':^12} | {'提升':^10}")
print('-' * 55)
for name, p_base, p_tta in variants:
    oa_base = json.load(open(p_base))['target_metrics']['OA'] if os.path.exists(p_base) else 0
    if os.path.exists(p_tta):
        m = json.load(open(p_tta))['target_metrics']
        oa_tta = m['OA']
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{name:<10} | {oa_base:>10.2f} | {oa_tta:>12.2f} | {oa_tta-oa_base:+.2f} | {pc}")
    else:
        print(f"{name:<10} | {oa_base:>10.2f} | (missing)")
PYEOF

