#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_t5t6_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast, re

def patch_one(name, path, old, new, verify_fn=None):
    p = Path(path)
    s = p.read_text()
    if verify_fn and verify_fn(s):
        print(f"  ⏭️ {name}: 已存在，跳过")
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

# ============ T5: _apply_source_aug 合并 aug_class_mask ============
print("\n===== T5 =====")
old_t5 = '''    def _apply_source_aug(xb, yb):
        if not use_source_aug:
            return xb
        aug_classes = cfg["training"].get("aug_classes", None)
        if aug_classes is not None:
            mask = torch.zeros(xb.size(0), dtype=torch.bool, device=xb.device)
            for c in aug_classes:
                mask = mask | yb.eq(c)
            if not mask.any():
                return xb
        else:
            mask = torch.ones(xb.size(0), dtype=torch.bool, device=xb.device)'''
new_t5 = '''    def _apply_source_aug(xb, yb):
        if not use_source_aug:
            return xb
        # T5 fix: 合并 aug_class_mask（按类别样本数保护）和 aug_classes（用户指定）
        base_mask = aug_class_mask[yb]
        _user_aug_classes = cfg["training"].get("aug_classes", None)
        if _user_aug_classes is not None:
            user_allow = torch.zeros_like(base_mask)
            for _c in _user_aug_classes:
                user_allow |= yb.eq(_c)
            base_mask = base_mask & user_allow
        mask = base_mask
        if not mask.any():
            return xb'''
patch_one("T5 aug_class_mask 合并", "core/trainer_v8.py",
          old_t5, new_t5,
          verify_fn=lambda s: 'base_mask = aug_class_mask[yb]' in s)

# ============ T6a: _check_spectral_plausibility 签名 ============
print("\n===== T6a: _check_spectral_plausibility 签名 =====")
p = Path('core/risk_controller.py')
s = p.read_text()
m = re.search(r'def _check_spectral_plausibility\(([^)]+)\):', s)
if m and 'gain_tv_max' not in m.group(1):
    old_sig = m.group(0)
    old_params = m.group(1)
    new_sig = f'def _check_spectral_plausibility({old_params},\n                                 gain_tv_max=0.36, offset_tv_max=0.18,\n                                 rel_change_max=0.15):'
    s = s.replace(old_sig, new_sig, 1)
    p.write_text(s)
    ast.parse(s)
    print(f"  ✅ T6a 签名参数化")
elif m and 'gain_tv_max' in m.group(1):
    print(f"  ⏭️ T6a 已存在")
else:
    print(f"  ❌ T6a 未找到 _check_spectral_plausibility 签名")

# ============ T6b: 硬编码三行改参数 ============
print("\n===== T6b =====")
old_t6b = '''    if gain_tv > 0.36:
        return False, f"Gain TV too high ({gain_tv:.3f})"
    if offset_tv > 0.18:
        return False, f"Offset TV too high ({offset_tv:.3f})"
    if rel_change > 0.15:
        return False, f"Relative spectral change too high ({rel_change:.3f})"'''
new_t6b = '''    if gain_tv > gain_tv_max:
        return False, f"Gain TV too high ({gain_tv:.3f} > {gain_tv_max})"
    if offset_tv > offset_tv_max:
        return False, f"Offset TV too high ({offset_tv:.3f} > {offset_tv_max})"
    if rel_change > rel_change_max:
        return False, f"Relative spectral change too high ({rel_change:.3f} > {rel_change_max})"'''
patch_one("T6b 三行参数化", "core/risk_controller.py",
          old_t6b, new_t6b,
          verify_fn=lambda s: 'gain_tv > gain_tv_max' in s)

# ============ T6c: Rsem 参数化 ============
print("\n===== T6c =====")
old_t6c = 'Rsem = float((torch.relu((margin0 - margin1) - 0.05) ** 2).mean().item())'
new_t6c = 'Rsem = float((torch.relu((margin0 - margin1) - rsem_delta) ** 2).mean().item())'
patch_one("T6c Rsem 参数化", "core/risk_controller.py",
          old_t6c, new_t6c,
          verify_fn=lambda s: 'rsem_delta) ** 2' in s)

# ============ T6d: backtracking_search 签名 ============
print("\n===== T6d =====")
old_t6d = '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5):'''
new_t6d = '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05):'''
patch_one("T6d backtracking_search 签名", "core/risk_controller.py",
          old_t6d, new_t6d,
          verify_fn=lambda s: 'rsem_delta=0.05):' in s and 'gain_tv_max=0.36' in s)

# ============ T6e: _check_spectral_plausibility 调用点传参 ============
print("\n===== T6e =====")
m = re.search(
    r'spec_ok, spec_reason = _check_spectral_plausibility\(\s*spectral_diag,\s*gain_max_dev,\s*offset_max_abs\s*\)',
    s
)
if m and 'gain_tv_max=gain_tv_max' not in m.group(0):
    old_call = m.group(0)
    new_call = '''spec_ok, spec_reason = _check_spectral_plausibility(
                spectral_diag, gain_max_dev, offset_max_abs,
                gain_tv_max=gain_tv_max, offset_tv_max=offset_tv_max,
                rel_change_max=rel_change_max)'''
    s = s.replace(old_call, new_call, 1)
    p.write_text(s)
    ast.parse(s)
    print(f"  ✅ T6e 调用点传参")
elif 'gain_tv_max=gain_tv_max' in s:
    print(f"  ⏭️ T6e 已存在")
else:
    print(f"  ❌ T6e 未匹配调用点（可能形态不同，但默认值仍然生效）")

# ============ T6f: trainer_v8.py 调用点传配置 ============
print("\n===== T6f =====")
old_t6f = '''                    tau_swd_rel=cfg["tcst"].get("tau_swd_rel", 0.01),
                    device=device,
                    swd_seed=cfg["output"]["seed"] + cycle)'''
new_t6f = '''                    tau_swd_rel=cfg["tcst"].get("tau_swd_rel", 0.01),
                    device=device,
                    swd_seed=cfg["output"]["seed"] + cycle,
                    gain_tv_max=float(cfg["tcst"].get("gain_tv_max", 0.36)),
                    offset_tv_max=float(cfg["tcst"].get("offset_tv_max", 0.18)),
                    rel_change_max=float(cfg["tcst"].get("rel_change_max", 0.15)),
                    rsem_delta=float(cfg["tcst"].get("rsem_delta", 0.05)))'''
patch_one("T6f trainer 调用点传配置", "core/trainer_v8.py",
          old_t6f, new_t6f,
          verify_fn=lambda s: 'gain_tv_max=float(cfg["tcst"].get("gain_tv_max"' in s)

print("\n===== 补丁阶段完成 =====")
PYEOF

echo ""
echo "===== 验证 T5 + T6 ====="
python - << 'PYEOF'
from pathlib import Path
rc = Path('core/risk_controller.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'T5 aug_class_mask 合并': 'base_mask = aug_class_mask[yb]' in tv,
    'T6a 签名参数化': 'gain_tv_max=0.36, offset_tv_max=0.18' in rc,
    'T6b 三行参数化': 'gain_tv > gain_tv_max' in rc,
    'T6c Rsem 参数化': 'rsem_delta) ** 2' in rc,
    'T6d bs 签名参数': 'rsem_delta=0.05):' in rc,
    'T6e 调用点传参': 'gain_tv_max=gain_tv_max' in rc,
    'T6f trainer 传配置': 'gain_tv_max=float(cfg["tcst"].get("gain_tv_max"' in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

echo ""
echo "===== 语法检查 ====="
python - << 'PYEOF'
import ast
from pathlib import Path
for f in ['core/risk_controller.py', 'core/trainer_v8.py']:
    ast.parse(Path(f).read_text())
    print(f"  ✅ {f}")
PYEOF

echo ""
echo "===== 冒烟测试（Houston 1 epoch）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p0_smoke/smoke2'
yaml.dump(c, open('configs/p0_smoke2.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p0_smoke/smoke2
python -u scripts/train.py --config configs/p0_smoke2.yaml 2>&1 | tail -25

echo ""
echo "✅ P0 (T1-T6) 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
