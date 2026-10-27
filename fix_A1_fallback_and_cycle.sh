#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_A1fb_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ 备份 $BACKUP"

echo ""
echo "=== 1) 诊断 cycle 里的 _eval_source 实际代码 ==="
grep -n -B 2 -A 4 "src_val = _eval_source" core/trainer_v8.py | head -40

python - << 'PYEOF'
from pathlib import Path
import ast, re

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 修 1: A1 fallback (val < 5 时不隔离) ============
old_a1 = '''                va_coords = np.array(_keep, dtype=np.int64) if _keep else _cand[:n_val]'''
new_a1 = '''                # A1 fallback: if isolation leaves too few val samples, fall back to unisolated.
                # Without this, classes whose pixels are spatially clustered (e.g. HyRANK class4/class10)
                # end up with val=1 or val=4, making src_val selection extremely noisy.
                if len(_keep) >= max(5, n_val // 2):
                    va_coords = np.array(_keep, dtype=np.int64)
                else:
                    va_coords = _cand[:n_val]'''
assert old_a1 in s, "A1 修复行未找到"
s = s.replace(old_a1, new_a1, 1)

# ============ 修 2: cycle 里的 _eval_source 调用 ============
# 用正则兜底匹配所有 _eval_source(... use_cb_val=... ) 调用
# 先把已经改好的 warmup 调用标记出来，避免二次替换
if 'ckpt_mode=ckpt_mode, num_classes=num_classes' in s:
    # 已有部分替换，用正则找剩余的旧调用
    old_pattern = re.compile(
        r'src_val = _eval_source\(model, src_val_p, src_val_y, device, use_cb_val=cfg\["training"\]\.get\("use_cb_val", False\)\)',
        re.DOTALL
    )
    n_remaining = len(old_pattern.findall(s))
    if n_remaining > 0:
        def _replace(m):
            return '''src_val = _eval_source(
                model, src_val_p, src_val_y, device,
                use_cb_val=cfg["training"].get("use_cb_val", False),
                ckpt_mode=ckpt_mode, num_classes=num_classes,
                target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None)'''
        s = old_pattern.sub(_replace, s)
        print(f"✅ 修 2: 替换了 {n_remaining} 处 cycle 里的 _eval_source 调用")
    else:
        print("⚠️ 没找到未替换的 cycle 调用，可能之前的正则已经部分生效")
else:
    print("⚠️ 没有发现 ckpt_mode 参数，说明阶段 1 的 _eval_source 修改可能没成功")

p.write_text(s)
ast.parse(s)
print("✅ 语法正确")
PYEOF

echo ""
echo "=== 2) 验证 cycle 调用是否已修复 ==="
grep -c "ckpt_mode=ckpt_mode, num_classes=num_classes" core/trainer_v8.py
echo "  ↑ 应该是 2（warmup 1 + cycle 1）；如果是 1，说明 cycle 还没修"

echo ""
echo "=== 3) 验证 A1 fallback ==="
grep -A 2 "A1 fallback" core/trainer_v8.py

echo ""
echo "=== 4) 重跑 A15 诊断，确认 HyRANK val 恢复 ==="
python - << 'PYEOF'
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'core')
import numpy as np
from core.data import load_scene
from core.trainer_v8 import _spatial_block_split

img = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni.mat'
gtp = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni_12gt.mat'
_, gt_arr = load_scene(img, gtp, None, 12, verbose=False)
masks, stats, mixed = _spatial_block_split(
    gt_arr, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
    val_frac_from_train=0.10, seed=1237+11, grid_size=10,
    per_class_train_cap=150, per_class_val_target=150, patch_size=13)
print(f"\nHyRANK mixed_classes={mixed}")
print(f"{'cls':<4} {'train':>8} {'val':>6}")
for c in range(1, 13):
    tr = int((masks['train'] & (gt_arr == c)).sum())
    va = int((masks['val'] & (gt_arr == c)).sum())
    print(f"  {c-1:<3} {tr:>8} {va:>6}")
PYEOF

echo ""
echo "✅ 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
