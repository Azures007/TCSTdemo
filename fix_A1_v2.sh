#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_A1v2_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) 给 _spatial_block_split 加 patch_size 参数 ============
old_sig = '''def _spatial_block_split(gt, train_ratio, constraint_per_class, calibration_per_class,
                         val_frac_from_train, seed, cal_pool_mult=3,
                         per_class_train_cap=0, per_class_val_cap=0,
                         per_class_val_target=0, grid_size=10):'''
new_sig = '''def _spatial_block_split(gt, train_ratio, constraint_per_class, calibration_per_class,
                         val_frac_from_train, seed, cal_pool_mult=3,
                         per_class_train_cap=0, per_class_val_cap=0,
                         per_class_val_target=0, grid_size=10, patch_size=11):'''
assert old_sig in s, "签名未找到"
s = s.replace(old_sig, new_sig, 1)

# ============ 2) 把 cfg["training"]["patch_size"] 改为 patch_size 参数 ============
old_cfg = '''            _ps = int(cfg["training"]["patch_size"])'''
new_cfg = '''            _ps = int(patch_size)'''
assert old_cfg in s, "cfg 引用未找到"
s = s.replace(old_cfg, new_cfg, 1)

# ============ 3) 调用点传入 patch_size ============
# 找调用 _spatial_block_split 的地方
old_call = '''        split, stats, mixed_classes = _spatial_block_split(
            src_gt, train_ratio, cn_per, ca_per, val_frac,
            seed=cfg["output"]["seed"] + 11,
            cal_pool_mult=cal_mult,
            per_class_train_cap=int(cfg["training"].get("per_class_train_cap", 0)),
            per_class_val_cap=int(cfg["training"].get("per_class_val_cap", 0)),
            per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)),
            grid_size=grid_size)'''
new_call = '''        split, stats, mixed_classes = _spatial_block_split(
            src_gt, train_ratio, cn_per, ca_per, val_frac,
            seed=cfg["output"]["seed"] + 11,
            cal_pool_mult=cal_mult,
            per_class_train_cap=int(cfg["training"].get("per_class_train_cap", 0)),
            per_class_val_cap=int(cfg["training"].get("per_class_val_cap", 0)),
            per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)),
            grid_size=grid_size, patch_size=patch_size)'''
assert old_call in s, "调用点未找到"
s = s.replace(old_call, new_call, 1)

p.write_text(s)
ast.parse(s)
print("✅ 修复: _spatial_block_split 加 patch_size 参数，调用点传入")
PYEOF

# 验证
echo ""
echo "=== 验证 ==="
python - << 'PYEOF'
from pathlib import Path
tv = Path('core/trainer_v8.py').read_text()
checks = {
    '签名有 patch_size 参数': 'per_class_val_target=0, grid_size=10, patch_size=11):' in tv,
    '混合模式用 patch_size 变量': '_ps = int(patch_size)' in tv,
    '混合模式不再用 cfg': 'cfg["training"]["patch_size"]' not in tv,
    '调用点传 patch_size': 'grid_size=grid_size, patch_size=patch_size)' in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

# 语法检查 + 轻量 smoke（只跑 setup，不跑训练）
echo ""
echo "=== 轻量 smoke（只跑 split 检查，不训练） ==="
python - << 'PYEOF'
import numpy as np
from core.trainer_v8 import _spatial_block_split
# 用随机 gt 测 _spatial_block_split
rng = np.random.default_rng(0)
gt = np.zeros((100, 200), dtype=np.int64)
# 3 个类，每个类随机分布在 10x20 像素
for c in [1, 2, 3]:
    for _ in range(30):
        r = rng.integers(0, 100); col = rng.integers(0, 200)
        gt[r, col] = c
try:
    masks, stats, mixed = _spatial_block_split(
        gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
        val_frac_from_train=0.10, seed=42, grid_size=10, patch_size=11)
    print(f"  ✅ _spatial_block_split 调用成功，stats={stats}, mixed={mixed}")
except Exception as e:
    print(f"  ❌ 报错: {e}")
    raise
PYEOF

echo ""
echo "✅ A1 v2 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
