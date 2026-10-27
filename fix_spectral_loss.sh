#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_specloss_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 1) 修改 clean 分支，加入 spectral aux loss
old_clean = '''            loss_clean = _ce(model(xb_raw).logits, yb)'''
new_clean = '''            _lambda_spec = float(cfg["training"].get("lambda_spec", 0.0))
            out_clean = model(xb_raw)
            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean = loss_clean + _lambda_spec * _ce(out_clean.spectral_logits, yb)'''
assert old_clean in s, "clean loss 未找到"
s = s.replace(old_clean, new_clean, 1)

# 2) 修改 safe 分支，加入 spectral aux loss
old_safe = '''                with _freeze_bn(model):
                    loss_safe = _ce(model(xb_safe).logits, yb)'''
new_safe = '''                with _freeze_bn(model):
                    out_safe = model(xb_safe)
                    loss_safe = _ce(out_safe.logits, yb)
                    if _lambda_spec > 0:
                        loss_safe = loss_safe + _lambda_spec * _ce(out_safe.spectral_logits, yb)'''
assert old_safe in s, "safe loss 未找到"
s = s.replace(old_safe, new_safe, 1)

p.write_text(s)
ast.parse(s)
print("✅ trainer_v8.py 已加入 spectral auxiliary loss")
PYEOF

# 生成 S→H 测试配置 (统一 L2+Amp+D，加入光谱辅助权重)
python - << 'PYEOF'
import yaml
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
base['training']['use_zscore'] = False
base['training']['amplitude_channel'] = True
base['training']['ckpt_mode'] = 'D'
base['training']['lambda_spec'] = 0.5  # 光谱分支辅助 Loss 权重
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/spec_aux_loss/sh_0.5'
yaml.dump(base, open('configs/sh_spec_aux.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

mkdir -p results_ablation/spec_aux_loss
echo ""
echo "=================================================="
echo "  RUN: S→H (L2+Amp+D) + Spectral Aux Loss (0.5)"
echo "=================================================="
python -u scripts/train.py --config configs/sh_spec_aux.yaml 2>&1 | \
    tee results_ablation/spec_aux_loss/log.txt | \
    grep -E "OA |AA |Kappa" | tail -3

echo ""
echo "=================================================="
echo "  光谱辅助监督 效果对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('S→H 基线 (无光谱监督)', 'results_ablation/unified_pipeline/shanghai/final_metrics.json'),
    ('S→H (加光谱监督 λ=0.5)', 'results_ablation/spec_aux_loss/sh_0.5/final_metrics.json')
]
print(f"{'Variant':<26} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 90)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<26} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<26} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {pc}")
PYEOF

