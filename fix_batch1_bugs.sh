#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_batch1_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

# ============ A6: TV 阈值改成数学上界的 60% ============
p = Path('core/risk_controller.py')
s = p.read_text()
old = '''    if max(abs(g_min - 1.0), abs(g_max - 1.0)) > gain_max_dev:
        return False, f"|a-1|_inf > {gain_max_dev}"
    if max(abs(o_min), abs(o_max)) > offset_max_abs:
        return False, f"|b|_inf > {offset_max_abs}"
    gain_tv = diag.get("gain_tv", 0.0)
    offset_tv = diag.get("offset_tv", 0.0)
    rel_change = diag.get("relative_spectral_change", 0.0)
    if gain_tv > 2.0:
        return False, f"Gain TV too high ({gain_tv:.2f})"
    if offset_tv > 1.0:
        return False, f"Offset TV too high ({offset_tv:.2f})"
    if rel_change > 0.15:
        return False, f"Relative spectral change too high ({rel_change:.2f})"'''
new = '''    if max(abs(g_min - 1.0), abs(g_max - 1.0)) > gain_max_dev:
        return False, f"|a-1|_inf > {gain_max_dev}"
    if max(abs(o_min), abs(o_max)) > offset_max_abs:
        return False, f"|b|_inf > {offset_max_abs}"
    gain_tv = diag.get("gain_tv", 0.0)
    offset_tv = diag.get("offset_tv", 0.0)
    rel_change = diag.get("relative_spectral_change", 0.0)
    # A6 fix: TV thresholds were 2.0/1.0, unreachable because rho_a<=0.15 limits
    # gain_tv to ~0.32 and offset_tv to ~0.30. Tighten to 60% of the mathematical bound.
    # With rho_a=0.15: max gain swing per adjacent pair <= 2*0.30=0.60, so TV cap 0.36.
    # With rho_a=0.10: max gain swing <= 2*0.21=0.42, so TV cap 0.25.
    if gain_tv > 0.36:
        return False, f"Gain TV too high ({gain_tv:.3f})"
    if offset_tv > 0.18:
        return False, f"Offset TV too high ({offset_tv:.3f})"
    if rel_change > 0.15:
        return False, f"Relative spectral change too high ({rel_change:.3f})"'''
assert old in s, "A6: risk_controller TV 检查未找到"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ A6: TV 阈值改成 0.36/0.18（数学上界的 60%）")

# ============ A7: 融合门初始化不被 _init() 覆盖 ============
p = Path('core/model.py')
s = p.read_text()
# 在 _init() 之后，重新应用融合门专用初始化
old_init_call = '''        self.spectral_fusion = SpectralReliabilityFusion(
            projection_dim, mode=spectral_fusion_mode)
        self._init()'''
new_init_call = '''        self.spectral_fusion = SpectralReliabilityFusion(
            projection_dim, mode=spectral_fusion_mode)
        self._init()
        # A7 fix: _init() overrides the fusion gate's specialized init.
        # Re-apply the intended init so gate starts near-zero weight (fusion ~= spatial).
        import math as _math
        with __import__('torch').no_grad():
            self.spectral_fusion.gate[-1].weight.zero_()
            _init_val = 0.12 / self.spectral_fusion.max_gate
            self.spectral_fusion.gate[-1].bias.fill_(
                _math.log(_init_val / (1 - _init_val)))'''
assert old_init_call in s, "A7: spectral_fusion + _init 未找到"
s = s.replace(old_init_call, new_init_call, 1)
p.write_text(s)
ast.parse(s)
print("✅ A7: 融合门初始化在 _init() 之后重新应用")

# ============ A8: Clean Loss 用原始数据，Safe 用变换后的数据 ============
p = Path('core/trainer_v8.py')
s = p.read_text()
old_aug = '''        xb = _apply_source_aug(xb, yb)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \\
                  if focal_gamma > 0 else \\
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            loss_clean = _ce(model(xb).logits, yb)
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                loss_safe = _ce(model(transform(xb, phi_star, labels=yb)).logits, yb)
                loss = (loss_clean + safe_w * loss_safe) / (1 + safe_w)
            else:
                loss = loss_clean'''
new_aug = '''        # A8 fix: clean branch uses RAW source (no aug); safe branch uses transform, then aug.
        xb_raw = xb
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \\
                  if focal_gamma > 0 else \\
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            loss_clean = _ce(model(xb_raw).logits, yb)
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)
                loss_safe = _ce(model(xb_safe).logits, yb)
                loss = (loss_clean + safe_w * loss_safe) / (1 + safe_w)
            else:
                loss = loss_clean'''
assert old_aug in s, "A8: _train_step 增强块未找到"
s = s.replace(old_aug, new_aug, 1)

# ============ A10: trainer 接受 optimal_inaccurate ============
old_accept = '''            if qp["status"] not in ("optimal", "no_constraints") or qp["d_norm"] < 1e-8:'''
new_accept = '''            if qp["status"] not in ("optimal", "optimal_inaccurate", "no_constraints") or qp["d_norm"] < 1e-8:'''
assert old_accept in s, "A10: trainer qp status 检查未找到"
s = s.replace(old_accept, new_accept, 1)
p.write_text(s)
ast.parse(s)
print("✅ A8: Clean Loss 用原始数据; ✅ A10: trainer 接受 optimal_inaccurate")

# ============ A11: MMD 带宽用源+目标联合中位数 ============
p = Path('core/discrepancy.py')
s = p.read_text()
old_bw = '''        valid_dxx = dxx[torch.triu(torch.ones_like(dxx), diagonal=1) > 0]
        valid_dxy = dxy.flatten()
        if valid_dxx.numel() > 0:
            median_dist = torch.median(valid_dxx).sqrt().item()
        else:
            median_dist = 1.0'''
new_bw = '''        # A11 fix: include cross-domain distances in the median estimate.
        # Previously only source internal distances were used, which can underestimate
        # the true scale when domain shift is large (over-saturating the kernel).
        valid_dxx = dxx[torch.triu(torch.ones_like(dxx), diagonal=1) > 0]
        valid_dxy = dxy.flatten()
        valid_all = torch.cat([valid_dxx, valid_dxy]) if valid_dxx.numel() > 0 else valid_dxy
        if valid_all.numel() > 0:
            median_dist = torch.median(valid_all).sqrt().item()
        else:
            median_dist = 1.0'''
assert old_bw in s, "A11: mk_mmd2 中位数估计未找到"
s = s.replace(old_bw, new_bw, 1)
p.write_text(s)
ast.parse(s)
print("✅ A11: MMD 带宽用源+目标联合中位数")

# ============ A10 补: transport_solver 用 xi.value 做残差诊断 ============
p = Path('core/transport_solver.py')
s = p.read_text()
old_res = '''    d_star = d.value.astype(np.float32)
    Gt = G @ d_star.astype(np.float64)
    slack = np.maximum(0.0, -eps - Gt)
    # 约束残差：Gd + eps + slack 超过 0 的部分（应接近 0）
    residual = np.maximum(0.0, -(Gt + eps + slack))'''
new_res = '''    d_star = d.value.astype(np.float32)
    Gt = G @ d_star.astype(np.float64)
    # A10 fix: use solver's actual xi.value for diagnostics, not reconstructed slack.
    # Reconstructed slack satisfies the constraint by construction and hides violations.
    if xi.value is not None:
        slack = np.maximum(0.0, np.asarray(xi.value, dtype=np.float64).flatten())
    else:
        slack = np.maximum(0.0, -eps - Gt)
    # True residual: how much the constraint Gd + eps + xi >= 0 is violated by solver's d and xi
    residual = np.maximum(0.0, -(Gt + eps + slack))'''
assert old_res in s, "A10: transport_solver 残差块未找到"
s = s.replace(old_res, new_res, 1)
p.write_text(s)
ast.parse(s)
print("✅ A10 补: 残差用 xi.value 诊断")
PYEOF

echo ""
echo "=== 验证 ==="
python - << 'PYEOF'
from pathlib import Path
checks = {
    'A6 TV 0.36/0.18': 'gain_tv > 0.36' in Path('core/risk_controller.py').read_text(),
    'A7 融合门重init': 'fusion gate' in Path('core/model.py').read_text().lower() or 'A7 fix' in Path('core/model.py').read_text(),
    'A8 Clean 用原始': 'xb_raw = xb' in Path('core/trainer_v8.py').read_text(),
    'A10 trainer 接受 opt_ina': 'optimal_inaccurate", "no_constraints' in Path('core/trainer_v8.py').read_text(),
    'A10 残差用 xi.value': 'xi.value' in Path('core/transport_solver.py').read_text(),
    'A11 联合中位数': 'torch.cat([valid_dxx, valid_dxy])' in Path('core/discrepancy.py').read_text(),
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

echo ""
echo "✅ 第 1 批修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
