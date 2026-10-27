#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_sched_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 找当前 total_epochs 计算（warmup scheduler 用的）
old = '''    total_epochs = (cfg["training"]["epochs_warmup"]
                    + cfg["training"]["num_cycles"] * cfg["training"]["epochs_per_cycle"])'''
new = '''    # S1 fix: warmup scheduler 用 warmup 自己的周期（不是总周期）
    # 原逻辑 T_max=total_epochs，cosine 在 warmup 结束时 lr 还没降到 min，停在尖锐局部最优
    # 改成 T_max=warmup_epochs，cosine 在 warmup 结束时 lr 降到 min，收敛到更平坦的解
    total_epochs = cfg["training"]["epochs_warmup"]
    _cycles_total = cfg["training"]["num_cycles"] * cfg["training"]["epochs_per_cycle"]
    _log(f"[scheduler] warmup T_max={total_epochs}, cycles T_max={_cycles_total}")'''
assert old in s, "total_epochs 未找到"
s2 = s.replace(old, new, 1)
try:
    ast.parse(s2)
    p.write_text(s2)
    print("✅ S1: warmup scheduler T_max = warmup_epochs")
except SyntaxError as e:
    print(f"❌ 语法错误 {e}")
    raise

# 验证 T4 的 cycles scheduler 仍然是 remaining_epochs
assert 'T_max={remaining_epochs}' in p.read_text(), "T4 的 cycles T_max 丢了"
print("✅ T4: cycles scheduler T_max = 剩余周期（保留）")
PYEOF

echo ""
echo "===== 验证 ====="
grep -n "total_epochs = cfg\|_cycles_total\|T_max={remaining_epochs}" core/trainer_v8.py

echo ""
echo "===== 语法检查 ====="
python -c "import ast; ast.parse(open('core/trainer_v8.py').read()); print('✅ trainer_v8.py')"

echo ""
echo "===== 冒烟测试（Houston 1 epoch，确认 scheduler 日志正常）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_s1_smoke/smoke'
yaml.dump(c, open('configs/s1_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_s1_smoke/smoke
python -u scripts/train.py --config configs/s1_smoke.yaml 2>&1 | grep -E "scheduler|warm [0-9]+/|restored|OA |AA " | head -10

echo ""
echo "✅ S1 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
