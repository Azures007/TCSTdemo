#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_t6bcd_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

RISK = 'core/risk_controller.py'

def patch_independent(name, old, new, verify_str):
    """每个补丁独立读文件、写文件，不共享变量。"""
    p = Path(RISK)
    s = p.read_text()
    if verify_str in s:
        print(f"  ⏭️ {name}: 已存在")
        return True
    if old not in s:
        print(f"  ❌ {name}: 未匹配")
        return False
    s2 = s.replace(old, new, 1)
    try:
        ast.parse(s2)
    except SyntaxError as e:
        print(f"  ❌ {name}: 语法错误 {e}")
        return False
    p.write_text(s2)
    print(f"  ✅ {name}")
    return True

# ============ T6d: backtracking_search 签名（先修这个，否则冒烟崩溃）============
print("===== T6d =====")
patch_independent(
    "T6d bs 签名",
    '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5):''',
    '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05):''',
    'rsem_delta=0.05):'
)

# ============ T6b: 三行参数化 ============
print("\n===== T6b =====")
patch_independent(
    "T6b 三行参数化",
    '''    if gain_tv > 0.36:
        return False, f"Gain TV too high ({gain_tv:.3f})"
    if offset_tv > 0.18:
        return False, f"Offset TV too high ({offset_tv:.3f})"
    if rel_change > 0.15:
        return False, f"Relative spectral change too high ({rel_change:.3f})"''',
    '''    if gain_tv > gain_tv_max:
        return False, f"Gain TV too high ({gain_tv:.3f} > {gain_tv_max})"
    if offset_tv > offset_tv_max:
        return False, f"Offset TV too high ({offset_tv:.3f} > {offset_tv_max})"
    if rel_change > rel_change_max:
        return False, f"Relative spectral change too high ({rel_change:.3f} > {rel_change_max})"''',
    'gain_tv > gain_tv_max'
)

# ============ T6c: Rsem 参数化 ============
print("\n===== T6c =====")
patch_independent(
    "T6c Rsem 参数化",
    'Rsem = float((torch.relu((margin0 - margin1) - 0.05) ** 2).mean().item())',
    'Rsem = float((torch.relu((margin0 - margin1) - rsem_delta) ** 2).mean().item())',
    'rsem_delta) ** 2'
)

print("\n===== 补丁完成 =====")
PYEOF

echo ""
echo "===== 验证 T6b/c/d（真实代码）====="
echo "--- T6b ---"
grep -n "gain_tv >\|offset_tv >\|rel_change >" core/risk_controller.py
echo "--- T6c ---"
grep -n "Rsem = float" core/risk_controller.py
echo "--- T6d ---"
grep -n -A 6 "def backtracking_search" core/risk_controller.py | head -8

echo ""
echo "===== 语法检查 ====="
python -c "import ast; ast.parse(open('core/risk_controller.py').read()); print('✅ risk_controller.py')"

echo ""
echo "===== 冒烟测试（Houston 1 epoch）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p0_smoke/smoke4'
yaml.dump(c, open('configs/p0_smoke4.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p0_smoke/smoke4
python -u scripts/train.py --config configs/p0_smoke4.yaml 2>&1 | tail -30

echo ""
echo "✅ T6b/c/d 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
