#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p04_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) 495 行：硬编码 "10% source train labels" ============
old_1 = '''    _log(f"  - 10% source train labels")'''
new_1 = '''    # P0.4: 动态计算实际训练样本占比，不再硬编码 "10%"
    _train_ratio_val = float(cfg["training"].get("source_train_ratio", 1.0))
    if _train_ratio_val >= 1.0 - 1e-6:
        _log(f"  - source train ratio: 100% (with per_class_train_cap as upper bound)")
    else:
        _log(f"  - source train ratio: {_train_ratio_val*100:.0f}%")'''
assert old_1 in s, "P0.4-1: 495 行硬编码 10% 未找到"
s = s.replace(old_1, new_1, 1)
print("✅ 1) 495 行硬编码 10% 改为动态比例")

# ============ 2) 551-552 行：per-class train counts 加"实际/上限"说明 ============
old_2 = '''    _log(f"  per-class train counts: "
         f"{np.bincount(src_tr_y, minlength=num_classes).tolist()}")'''
new_2 = '''    # P0.4: 明确 per_class_train_cap 是上限，并打印每类实际训练数 vs 该类总标注数
    _cap = int(cfg["training"].get("per_class_train_cap", 0))
    _tr_counts = np.bincount(src_tr_y, minlength=num_classes).tolist()
    _all_counts = [int((src_gt == c).sum()) for c in range(1, num_classes + 1)]
    _log(f"  per-class train counts (cap={_cap}, upper bound): {_tr_counts}")
    _log(f"  per-class total labeled (before split): {_all_counts}")
    _short = [c for c in range(num_classes) if _tr_counts[c] < _cap]
    if _short:
        _log(f"  note: classes {_short} have train < cap because total labeled pixels are insufficient")'''
assert old_2 in s, "P0.4-2: per-class train counts 未找到"
s = s.replace(old_2, new_2, 1)
print("✅ 2) per-class train counts 加 cap 说明和实际/总数对比")

p.write_text(s)
ast.parse(s)
print("✅ 3) 语法正确")
PYEOF

echo ""
echo "===== 验证 ====="
grep -n "10% source train labels\|source train ratio: 100%\|per_class_train_cap as upper bound\|upper bound)" core/trainer_v8.py

echo ""
echo "===== 冒烟（Houston 1 epoch，看新日志）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p04_smoke/smoke'
yaml.dump(c, open('configs/p04_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p04_smoke/smoke
python -u scripts/train.py --config configs/p04_smoke.yaml 2>&1 | grep -E "source train ratio|source-split|per-class train|per-class total|note:|warm [0-9]+/|OA |AA " | head -15

echo ""
echo "✅ P0.4 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
