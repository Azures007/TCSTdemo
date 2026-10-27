#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo

python - << 'PYEOF'
from pathlib import Path
import ast
import re

p = Path('core/trainer_v8.py')
s = p.read_text()

# 匹配并替换整个 _train_step 函数
pattern = re.compile(r'    def _train_step\(patches, labels, phi_star, safe_w\):.*?(?=\n    # ------------- warm-up)', re.DOTALL)

new_step = '''    def _train_step(patches, labels, phi_star, safe_w):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if isinstance(patches, np.ndarray):
            xb = torch.from_numpy(patches).to(device)
        else:
            xb = patches.to(device, non_blocking=True)
        if isinstance(labels, np.ndarray):
            yb = torch.from_numpy(labels).to(device)
        else:
            yb = labels.to(device, non_blocking=True)
        
        xb_raw = xb
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \\
                  if focal_gamma > 0 else \\
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            
            _lambda_spec = float(cfg["training"].get("lambda_spec", 0.0))
            out_clean = model(xb_raw)
            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)
            
            _use_safe_training = bool(cfg["training"].get("use_safe_training", True))
            _safe_executed = False
            loss_safe = None
            
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                _safe_executed = True
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                with _freeze_bn(model):
                    out_safe = model(xb_safe)
                    loss_safe = _ce(out_safe.logits, yb)
                    if _lambda_spec > 0:
                        loss_safe += _lambda_spec * _ce(out_safe.spectral_logits, yb)
            
            if loss_safe is not None and not _use_safe_training:
                loss = loss_safe
            elif loss_safe is not None:
                loss = (loss_clean + safe_w * loss_safe) / (1 + safe_w)
            else:
                loss = loss_clean
            
            # Pseudo-label loss
            _use_pseudo = bool(cfg["training"].get("use_pseudo", False))
            if _use_pseudo and pseudo_state["xb"] is not None:
                pb_size = xb.size(0)
                idx_p = np.random.choice(len(pseudo_state["xb"]), pb_size, replace=True)
                p_xb = torch.from_numpy(pseudo_state["xb"][idx_p]).to(device)
                p_yb = torch.from_numpy(pseudo_state["yb"][idx_p]).to(device)
                pseudo_w = float(cfg["training"].get("pseudo_weight", 0.5))
                
                out_pseudo = model(p_xb)
                loss_pseudo = _ce(out_pseudo.logits, p_yb)
                if _lambda_spec > 0:
                    loss_pseudo += _lambda_spec * _ce(out_pseudo.spectral_logits, p_yb)
                loss = loss + pseudo_w * loss_pseudo

        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        return {
            "loss": float(loss.detach()),
            "clean_loss": float(loss_clean.detach()),
            "safe_loss": float(loss_safe.detach()) if loss_safe is not None else None,
            "safe_executed": _safe_executed,
        }
'''

m = pattern.search(s)
if m:
    s = s[:m.start()] + new_step + s[m.end():]
    p.write_text(s)
    ast.parse(p.read_text())
    print("✅ _train_step 已暴力重写为绝对纯净版 (无 Mixup 残留)")
else:
    print("❌ 没找到 _train_step 函数，请手动检查")
PYEOF

echo ""
echo "=================================================="
echo "  验证运行: S→H (Patch=5, 纯净版)"
echo "=================================================="
python -u scripts/train.py --config configs/sh_verify.yaml 2>&1 | grep -E "CB-selected|OA |AA |Kappa" | tail -5
