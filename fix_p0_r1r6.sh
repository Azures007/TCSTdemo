#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p0_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# ============ T1: B4 真正修复 ============
echo ""
echo "===== T1: SWD 不等长 + trainer 传全部 target ====="
python - << 'PYEOF'
from pathlib import Path
import ast

# 1a) discrepancy.py: sliced_wasserstein 支持不等长（用分位数插值）
p = Path('core/discrepancy.py')
s = p.read_text()
old_swd = '''@torch.no_grad()
def sliced_wasserstein(x, y, num_projections=64, seed=0):
    """SWD requires equal sample counts; truncate to min(N)."""
    device = x.device
    n = min(x.size(0), y.size(0))
    if n == 0:
        return 0.0
    x = x[:n]; y = y[:n]
    g = torch.Generator(device=device).manual_seed(int(seed))
    D = x.shape[1]
    proj = F.normalize(torch.randn(D, num_projections, generator=g, device=device), dim=0)
    px, _ = torch.sort(x @ proj, dim=0)
    py, _ = torch.sort(y @ proj, dim=0)
    return float((px - py).abs().mean())'''
new_swd = '''@torch.no_grad()
def sliced_wasserstein(x, y, num_projections=64, seed=0):
    """SWD with unequal sample counts (empirical quantile interpolation)."""
    device = x.device
    if x.size(0) == 0 or y.size(0) == 0:
        return 0.0
    g = torch.Generator(device=device).manual_seed(int(seed))
    D = x.shape[1]
    proj = F.normalize(torch.randn(D, num_projections, generator=g, device=device), dim=0)
    px, _ = torch.sort(x @ proj, dim=0)   # [Nx, L]
    py, _ = torch.sort(y @ proj, dim=0)   # [Ny, L]
    nx, ny = px.size(0), py.size(0)
    if nx == ny:
        return float((px - py).abs().mean())
    # 在 [0,1] 网格上取 empirical quantile，做插值
    qs = torch.linspace(0.0, 1.0, steps=256, device=device, dtype=px.dtype)
    def _quantile(sorted_vals, q):
        # sorted_vals: [N, L], q: [Q]
        n = sorted_vals.size(0)
        idx = q * (n - 1)
        lo = torch.floor(idx).long().clamp(0, n - 1)
        hi = torch.ceil(idx).long().clamp(0, n - 1)
        w = (idx - lo.float()).view(-1, 1)
        return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w   # [Q, L]
    qx = _quantile(px, qs)   # [Q, L]
    qy = _quantile(py, qs)   # [Q, L]
    return float((qx - qy).abs().mean())'''
assert old_swd in s, "sliced_wasserstein 未找到"
s = s.replace(old_swd, new_swd, 1)
p.write_text(s)
ast.parse(s)
print("✅ T1a: sliced_wasserstein 支持不等长样本")

# 1b) trainer_v8.py: 传全部 tgt_calibration_p，不再 [:64]
p = Path('core/trainer_v8.py')
s = p.read_text()
old_call = '''                    src_ca_p, src_ca_y,
                    tgt_calibration_p[:64],'''
new_call = '''                    src_ca_p, src_ca_y,
                    tgt_calibration_p,'''
assert old_call in s, "trainer 里 tgt_calibration_p[:64] 未找到"
s = s.replace(old_call, new_call, 1)
p.write_text(s)
ast.parse(s)
print("✅ T1b: trainer 传全部 target calibration")
PYEOF

# ============ T2+T3: B3 完整向量写入 jsonl + qp=None ============
echo ""
echo "===== T2+T3: jsonl 写入 full + qp 初始化 ====="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

# T2: jsonl 写入增加 full
old_jsonl = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
new_jsonl = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "phi_star_full": _a12_phi_full,
                                "d_star_full": _a12_d_full,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
assert old_jsonl in s, "jsonl 写入未找到"
s = s.replace(old_jsonl, new_jsonl, 1)
print("✅ T2: jsonl 写入 phi_star_full / d_star_full")

# T3: 每 cycle 开始前 qp = None
old_cycle_start = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()
'''
new_cycle_start = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()
        qp = None   # T3: 每 cycle 重置，防止跨 cycle 污染
'''
assert old_cycle_start in s, "cycle 起始未找到"
s = s.replace(old_cycle_start, new_cycle_start, 1)
print("✅ T3: 每 cycle 重置 qp = None")

p.write_text(s)
ast.parse(s)
PYEOF

# ============ T4: Scheduler T_max = 剩余周期 ============
echo ""
echo "===== T4: Scheduler T_max = 剩余周期 ====="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()
# 在重建 scheduler 的地方，T_max 改为剩余周期（cycles）
old_rebind = '''        if scheduler is not None:
            scheduler = _build_scheduler(optimizer, scheduler_name, total_epochs)
            _log(f"[scheduler] re-bound to new optimizer: {scheduler_name}")'''
new_rebind = '''        if scheduler is not None:
            # T4 fix: T_max = 剩余周期（cycles），不是总周期
            remaining_epochs = cfg["training"]["num_cycles"] * cfg["training"]["epochs_per_cycle"]
            scheduler = _build_scheduler(optimizer, scheduler_name, remaining_epochs)
            _log(f"[scheduler] re-bound: {scheduler_name}, T_max={remaining_epochs} (remaining after warmup)")'''
assert old_rebind in s, "scheduler rebind 未找到"
s = s.replace(old_rebind, new_rebind, 1)
p.write_text(s)
ast.parse(s)
print("✅ T4: Scheduler T_max = 剩余周期")
PYEOF

# ============ T5: aug_class_mask 合并 ============
echo ""
echo "===== T5: aug_class_mask 合并到 _apply_source_aug ====="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()
old_aug = '''    def _apply_source_aug(xb, yb):
        if not use_source_aug:
            return xb
        aug_mask = aug_class_mask[yb]
        if not aug_mask.any():
            return xb'''
new_aug = '''    def _apply_source_aug(xb, yb):
        if not use_source_aug:
            return xb
        # T5 fix: 同时考虑 aug_class_mask（按类别样本数保护）和 aug_classes（用户指定）
        base_mask = aug_class_mask[yb]
        if aug_classes is not None:
            user_allow = torch.zeros_like(base_mask)
            for c in aug_classes:
                user_allow |= yb.eq(c)
            base_mask = base_mask & user_allow
        aug_mask = base_mask
        if not aug_mask.any():
            return xb'''
assert old_aug in s, "_apply_source_aug 未找到"
s = s.replace(old_aug, new_aug, 1)
p.write_text(s)
ast.parse(s)
print("✅ T5: aug_class_mask 与 aug_classes 合并")
PYEOF

# ============ T6: Risk Search 阈值配置化 ============
echo ""
echo "===== T6: Risk Search 阈值从 cfg 读 ====="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/risk_controller.py')
s = p.read_text()

# 6a) _check_spectral_plausibility 接受阈值参数
old_sig = '''def _check_spectral_plausibility(diag, gain_max_dev, offset_max_abs):
    """Return (ok, reason) for spectral plausibility."""'''
new_sig = '''def _check_spectral_plausibility(diag, gain_max_dev, offset_max_abs,
                                 gain_tv_max=0.36, offset_tv_max=0.18,
                                 rel_change_max=0.15):
    """Return (ok, reason) for spectral plausibility. T6: thresholds configurable."""'''
assert old_sig in s, "spectral check 签名未找到"
s = s.replace(old_sig, new_sig, 1)

# 6b) 内部用参数而不是硬编码
old_body = '''    if gain_tv > 0.36:
        return False, f"Gain TV too high ({gain_tv:.3f})"
    if offset_tv > 0.18:
        return False, f"Offset TV too high ({offset_tv:.3f})"
    if rel_change > 0.15:
        return False, f"Relative spectral change too high ({rel_change:.3f})"'''
new_body = '''    if gain_tv > gain_tv_max:
        return False, f"Gain TV too high ({gain_tv:.3f} > {gain_tv_max})"
    if offset_tv > offset_tv_max:
        return False, f"Offset TV too high ({offset_tv:.3f} > {offset_tv_max})"
    if rel_change > rel_change_max:
        return False, f"Relative spectral change too high ({rel_change:.3f} > {rel_change_max})"'''
assert old_body in s, "spectral check body 未找到"
s = s.replace(old_body, new_body, 1)

# 6c) backtracking_search 签名加参数
old_bs_sig = '''def backtracking_search(model, transform, d_star,
                        source_cal_patches, source_cal_labels,
                        target_cal_patches, eta_candidates,
                        tau_sem, tau_flip, tau_class_flip,
                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5):'''
new_bs_sig = '''def backtracking_search(model, transform, d_star,
                        source_cal_patches, source_cal_labels,
                        target_cal_patches, eta_candidates,
                        tau_sem, tau_flip, tau_class_flip,
                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05):'''
assert old_bs_sig in s, "backtracking_search 签名未找到"
s = s.replace(old_bs_sig, new_bs_sig, 1)

# 6d) 调用 _check_spectral_plausibility 时传参数
old_check_call = '''            spec_ok, spec_reason = _check_spectral_plausibility(
                spectral_diag, gain_max_dev, offset_max_abs)'''
new_check_call = '''            spec_ok, spec_reason = _check_spectral_plausibility(
                spectral_diag, gain_max_dev, offset_max_abs,
                gain_tv_max=gain_tv_max, offset_tv_max=offset_tv_max,
                rel_change_max=rel_change_max)'''
assert old_check_call in s, "spectral check 调用未找到"
s = s.replace(old_check_call, new_check_call, 1)

# 6e) Rsem 用 rsem_delta 参数
old_rsem = '''            Rsem = float((torch.relu((margin0 - margin1) - 0.05) ** 2).mean().item())'''
new_rsem = '''            Rsem = float((torch.relu((margin0 - margin1) - rsem_delta) ** 2).mean().item())'''
assert old_rsem in s, "Rsem 未找到"
s = s.replace(old_rsem, new_rsem, 1)

p.write_text(s)
ast.parse(s)
print("✅ T6: Risk Search 阈值参数化（gain_tv_max/offset_tv_max/rel_change_max/rsem_delta）")
PYEOF

# ============ 验证 6 个修复 ============
echo ""
echo "===== 验证 T1-T6 ====="
python - << 'PYEOF'
from pathlib import Path
dc = Path('core/discrepancy.py').read_text()
rc = Path('core/risk_controller.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'T1a SWD 不等长': 'empirical quantile' in dc or '_quantile' in dc,
    'T1b trainer 全 target': 'tgt_calibration_p,' in tv and 'tgt_calibration_p[:64],' not in tv,
    'T2 jsonl full': '"phi_star_full": _a12_phi_full' in tv,
    'T3 qp 重置': 'qp = None   # T3' in tv,
    'T4 T_max 剩余': 'T_max={remaining_epochs}' in tv,
    'T5 aug_mask 合并': 'aug_mask = base_mask' in tv,
    'T6 阈值参数化': 'gain_tv_max=gain_tv_max' in rc and 'rsem_delta' in rc,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

echo ""
echo "===== 语法检查 ====="
python - << 'PYEOF'
import ast
from pathlib import Path
for f in ['core/discrepancy.py', 'core/risk_controller.py', 'core/trainer_v8.py']:
    ast.parse(Path(f).read_text())
    print(f"  ✅ {f}")
PYEOF

echo ""
echo "===== 冒烟测试（Houston 1 epoch） ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p0_smoke/smoke'
yaml.dump(c, open('configs/p0_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p0_smoke/smoke
python -u scripts/train.py --config configs/p0_smoke.yaml 2>&1 | tail -20

echo ""
echo "✅ P0 (T1-T6) 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
