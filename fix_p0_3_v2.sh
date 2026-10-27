#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p03v2_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast, re

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) 签名加 strict_mixed_isolation 参数 ============
old_sig = '''def _spatial_block_split(gt, train_ratio, constraint_per_class, calibration_per_class,
                         val_frac_from_train, seed, cal_pool_mult=3,
                         per_class_train_cap=0, per_class_val_cap=0,
                         per_class_val_target=0, grid_size=10, patch_size=11):'''
new_sig = '''def _spatial_block_split(gt, train_ratio, constraint_per_class, calibration_per_class,
                         val_frac_from_train, seed, cal_pool_mult=3,
                         per_class_train_cap=0, per_class_val_cap=0,
                         per_class_val_target=0, grid_size=10, patch_size=11,
                         strict_mixed_isolation=False):'''
assert old_sig in s, "P0.3: 签名未找到"
s = s.replace(old_sig, new_sig, 1)
print("✅ 1) 签名加 strict_mixed_isolation=False")

# ============ 2) 第 338 行的 fallback（_keep_wide 也空时）============
old_338 = '''                        va_coords = _cand[:n_val] if len(_cand) > 0 else np.empty((0, 2), dtype=np.int64)'''
new_338 = '''                        if strict_mixed_isolation:
                            # P0.3: strict 模式禁止 fallback 到未隔离，隔离后剩多少用多少
                            va_coords = np.empty((0, 2), dtype=np.int64)
                            _log(f"  [split] class {c} mixed: STRICT, 0 isolated val (fallback disabled)")
                        else:
                            va_coords = _cand[:n_val] if len(_cand) > 0 else np.empty((0, 2), dtype=np.int64)'''
assert old_338 in s, "P0.3: 338 行 fallback 未找到"
s = s.replace(old_338, new_338, 1)
print("✅ 2) 338 行 fallback 加 strict 保护")

# ============ 3) 第 340 行的 fallback（tr_coords 或 _cand 为空时）============
# 注意：340 行和 338 行内容相同（都是 _cand[:n_val]），但缩进不同
# 用行号精确定位，避免误替换
lines = s.split('\n')
# 找 340 行（1-indexed），现在因为 338 行被展开成 6 行，行号会偏移
# 所以用 "在 mixed_classes.append 之后的第二个 va_coords = _cand[:n_val]" 来定位
old_340 = '''            else:
                va_coords = _cand[:n_val]'''
new_340 = '''            else:
                if strict_mixed_isolation:
                    va_coords = np.empty((0, 2), dtype=np.int64)
                    _log(f"  [split] class {c} mixed: STRICT, 0 isolated val (no candidate)")
                else:
                    va_coords = _cand[:n_val]'''
# 只替换 mixed 分支里的那个（前面有 if len(tr_coords) > 0 and len(_cand) > 0）
# 用更长的上下文定位
old_340_ctx = '''            if len(tr_coords) > 0 and len(_cand) > 0:'''
assert old_340_ctx in s, "P0.3: mixed 分支 if 未找到"
# 找 old_340_ctx 之后的第一个 old_340
idx = s.find(old_340_ctx)
assert idx >= 0, "P0.3: mixed 分支 if 位置异常"
tail = s[idx:]
if old_340 in tail:
    # 在 tail 里替换第一个 old_340
    tail_new = tail.replace(old_340, new_340, 1)
    s = s[:idx] + tail_new
    print("✅ 3) 340 行 fallback 加 strict 保护")
else:
    raise SystemExit("P0.3: 340 行 fallback 未找到（在 mixed 分支 if 之后）")

p.write_text(s)
ast.parse(s)
print("✅ 4) 语法正确")
PYEOF

echo ""
echo "===== 5) 调用点传 strict_mixed_isolation ====="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 找调用点（两种可能：空间划分启用 / 禁用）
old_call = '''            per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)),
            grid_size=grid_size, patch_size=patch_size)'''
new_call = '''            per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)),
            grid_size=grid_size, patch_size=patch_size,
            strict_mixed_isolation=bool(cfg["training"].get("strict_mixed_isolation", False)))'''
assert old_call in s, "P0.3: 调用点未找到"
s = s.replace(old_call, new_call, 1)
p.write_text(s)
ast.parse(s)
print("✅ 5) 调用点传 strict_mixed_isolation")
PYEOF

echo ""
echo "===== 验证 ====="
grep -n "strict_mixed_isolation" core/trainer_v8.py
python -c "import ast; ast.parse(open('core/trainer_v8.py').read()); print('✅ 语法正确')"

echo ""
echo "===== 冒烟 A: strict=false（应和 v8.0.9 一致）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['training']['strict_mixed_isolation'] = False
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p03_smoke/loose'
yaml.dump(c, open('configs/p03_loose.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p03_smoke/loose
python -u scripts/train.py --config configs/p03_loose.yaml 2>&1 | grep -E "\[split\]|warm [0-9]+/|OA |AA " | tail -10

echo ""
echo "===== 冒烟 B: strict=true（看 val 是否减少）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['training']['strict_mixed_isolation'] = True
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p03_smoke/strict'
yaml.dump(c, open('configs/p03_strict.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p03_smoke/strict
python -u scripts/train.py --config configs/p03_strict.yaml 2>&1 | grep -E "\[split\]|warm [0-9]+/|OA |AA " | tail -15

echo ""
echo "✅ P0.3 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
