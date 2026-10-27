#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p03_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 1) 加配置开关读取（在 mixed 分支的兜底处）
old = '''                if len(_keep_wide) >= 1:
                    va_coords = np.array(_keep_wide, dtype=np.int64)
                else:
                    # 最后兜底: 未隔离样本
                    va_coords = _cand[:n_val] if len(_cand) > 0 else np.empty((0, 2), dtype=np.int64)'''
new = '''                if len(_keep_wide) >= 1:
                    va_coords = np.array(_keep_wide, dtype=np.int64)
                else:
                    # P0.3 修复: strict_mixed_isolation=True 时禁止 fallback 到未隔离
                    # 默认 False 保持 v8.0.9 原行为，便于严格消融
                    _strict = bool(cfg["training"].get("strict_mixed_isolation", False))
                    if _strict:
                        # 隔离后有多少用多少（哪怕只有 5 个），绝不用未隔离样本
                        va_coords = np.empty((0, 2), dtype=np.int64)
                    else:
                        va_coords = _cand[:n_val] if len(_cand) > 0 else np.empty((0, 2), dtype=np.int64)'''
assert old in s, "mixed 兜底逻辑未找到"
s = s.replace(old, new, 1)

# 2) 在函数开头加日志：哪些类 mixed、哪些类 fallback
old2 = '''        greedy_ok = (train_pix >= 100 and val_pix >= 50 and len(val_blocks) >= 1)'''
new2 = '''        greedy_ok = (train_pix >= 100 and val_pix >= 50 and len(val_blocks) >= 1)
        _strict_mixed = bool(cfg["training"].get("strict_mixed_isolation", False))'''
assert old2 in s, "greedy_ok 行未找到"
s = s.replace(old2, new2, 1)

# 3) 在 mixed 分支末尾加日志
old3 = '''            if len(_keep) >= max(5, n_val // 2):
                va_coords = np.array(_keep, dtype=np.int64)'''
new3 = '''            if len(_keep) >= max(5, n_val // 2):
                va_coords = np.array(_keep, dtype=np.int64)
                _log(f"  [split] class {c} mixed: {len(va_coords)} isolated val")'''
assert old3 in s, "mixed _keep 分支未找到"
s = s.replace(old3, new3, 1)

p.write_text(s)
ast.parse(s)
print("✅ P0.3: strict_mixed_isolation 开关已加（默认 False = v8.0.9 行为）")
print("   配置 strict_mixed_isolation=true 时：mixed 模式禁止 fallback 到未隔离")
PYEOF

echo ""
echo "===== 验证 ====="
grep -n "strict_mixed_isolation" core/trainer_v8.py
python -c "import ast; ast.parse(open('core/trainer_v8.py').read()); print('✅ 语法正确')"

echo ""
echo "===== 冒烟 A: strict_mixed_isolation=false（默认，应和 v8.0.9 一致）====="
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
python -u scripts/train.py --config configs/p03_loose.yaml 2>&1 | grep -E "split|warm [0-9]+/|OA |AA " | tail -8

echo ""
echo "===== 冒烟 B: strict_mixed_isolation=true（严格，检查 val 数量变化）====="
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
python -u scripts/train.py --config configs/p03_strict.yaml 2>&1 | grep -E "split|warm [0-9]+/|OA |AA " | tail -12

echo ""
echo "✅ P0.3 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
