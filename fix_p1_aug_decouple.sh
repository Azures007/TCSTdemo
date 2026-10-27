#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p1aug_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) _train_step 加独立增强分支 ============
old_ts = '''        xb_raw = xb
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \\
                  if focal_gamma > 0 else \\
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            loss_clean = _ce(model(xb_raw).logits, yb)
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)
                # T8: Safe 分支 forward 时冻结 BN 统计（避免 Clean/Safe 双更新）
                with _freeze_bn(model):
                    loss_safe = _ce(model(xb_safe).logits, yb)
                loss = (loss_clean + safe_w * loss_safe) / (1 + safe_w)
            else:
                loss = loss_clean'''

new_ts = '''        xb_raw = xb
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \\
                  if focal_gamma > 0 else \\
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            loss_clean = _ce(model(xb_raw).logits, yb)
            # P1: 独立普通源域增强分支（默认关闭，保持 v8.0.9 行为）
            _use_warmup_aug = bool(cfg["training"].get("use_warmup_augmentation", False))
            _lambda_aug = float(cfg["training"].get("lambda_aug", 0.0))
            _aug_executed = False
            loss_aug = None
            if _use_warmup_aug and _lambda_aug > 0:
                xb_aug = _apply_source_aug(xb_raw, yb)
                loss_aug = _ce(model(xb_aug).logits, yb)
                _aug_executed = True
            # Safe 分支（T-CST 变换）
            _safe_executed = (phi_star is not None and float(phi_star.abs().sum()) > 1e-8)
            loss_safe = None
            if _safe_executed:
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)
                with _freeze_bn(model):
                    loss_safe = _ce(model(xb_safe).logits, yb)
            # 三路融合：clean + lambda_aug * aug + safe_w * safe
            _terms = [loss_clean]
            _weights = [1.0]
            if loss_aug is not None:
                _terms.append(loss_aug)
                _weights.append(_lambda_aug)
            if loss_safe is not None:
                _terms.append(loss_safe)
                _weights.append(safe_w)
            _wsum = sum(_weights)
            loss = sum(w * t for w, t in zip(_weights, _terms)) / _wsum'''
assert old_ts in s, "P1: _train_step 主体未找到"
s = s.replace(old_ts, new_ts, 1)
print("✅ 1) _train_step 加独立增强分支（三路融合）")

# ============ 2) _train_step 返回值加 aug_executed ============
old_ret = '''        _safe_exec = (phi_star is not None and float(phi_star.abs().sum()) > 1e-8)
        return {
            "loss": float(loss.detach()),
            "clean_loss": float(loss_clean.detach()),
            "safe_loss": float(loss_safe.detach()) if _safe_exec else None,
            "safe_executed": _safe_exec,
        }'''
new_ret = '''        return {
            "loss": float(loss.detach()),
            "clean_loss": float(loss_clean.detach()),
            "safe_loss": float(loss_safe.detach()) if loss_safe is not None else None,
            "aug_loss": float(loss_aug.detach()) if loss_aug is not None else None,
            "safe_executed": bool(_safe_executed),
            "aug_executed": bool(_aug_executed),
        }'''
assert old_ret in s, "P1: _train_step 返回值未找到"
s = s.replace(old_ret, new_ret, 1)
print("✅ 2) _train_step 返回值加 aug_executed")

p.write_text(s)
ast.parse(s)
print("✅ 3) 语法正确")
PYEOF

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 3) warmup 日志加 aug 执行统计 ============
old_warm_log = '''        _log(f"  warm {ep:03d}/{cfg['training']['epochs_warmup']}  "
             f"loss={np.mean([l['loss'] for l in losses]):.4f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"src_val={src_val:.2f}  best={best_src_val:.2f}  "
             f"t={(time.time()-t0)/60:.1f}m")'''
new_warm_log = '''        _aug_count = sum(1 for l in losses if l.get("aug_executed", False))
        _log(f"  warm {ep:03d}/{cfg['training']['epochs_warmup']}  "
             f"loss={np.mean([l['loss'] for l in losses]):.4f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"src_val={src_val:.2f}  best={best_src_val:.2f}  "
             f"aug={_aug_count}/{len(losses)}  t={(time.time()-t0)/60:.1f}m")'''
assert old_warm_log in s, "P1: warmup 日志未找到"
s = s.replace(old_warm_log, new_warm_log, 1)
print("✅ 3) warmup 日志加 aug 执行统计")

# ============ 4) warmup manifest entry 加 aug ============
old_warm_entry = '''        _warm_entry = {
            "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,
            "safe_executed_count": 0,
            "is_best": False,
        }'''
new_warm_entry = '''        _warm_entry = {
            "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,
            "aug_loss_mean": float(np.mean([l["aug_loss"] for l in losses if l.get("aug_loss") is not None])) if any(l.get("aug_loss") is not None for l in losses) else None,
            "safe_executed_count": 0,
            "aug_executed_count": sum(1 for l in losses if l.get("aug_executed", False)),
            "is_best": False,
        }'''
assert old_warm_entry in s, "P1: warmup entry 未找到"
s = s.replace(old_warm_entry, new_warm_entry, 1)
print("✅ 4) warmup manifest 加 aug 统计")

# ============ 5) cycle manifest entry 加 aug ============
old_cyc_entry = '''            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "safe_executed_count": _safe_count,
                "safe_total_batches": len(losses),
                "is_best": False,
            }'''
new_cyc_entry = '''            _aug_count_cyc = sum(1 for l in losses if l.get("aug_executed", False))
            _aug_losses = [l["aug_loss"] for l in losses if l.get("aug_loss") is not None]
            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "aug_loss_mean": float(np.mean(_aug_losses)) if _aug_losses else None,
                "safe_executed_count": _safe_count,
                "aug_executed_count": _aug_count_cyc,
                "safe_total_batches": len(losses),
                "is_best": False,
            }'''
assert old_cyc_entry in s, "P1: cycle entry 未找到"
s = s.replace(old_cyc_entry, new_cyc_entry, 1)
print("✅ 5) cycle manifest 加 aug 统计")

# ============ 6) cycle 日志加 aug 执行 ============
old_cyc_log = '''            _log(f"    cyc{cycle} ep{ep:02d}  loss={np.mean([l['loss'] for l in losses]):.4f}  "
                 f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                 f"src_val={src_val:.2f}  best={best_global['src_val']:.2f}  "
                 f"safe={_safe_count}/{len(losses)}{marker}")'''
new_cyc_log = '''            _log(f"    cyc{cycle} ep{ep:02d}  loss={np.mean([l['loss'] for l in losses]):.4f}  "
                 f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                 f"src_val={src_val:.2f}  best={best_global['src_val']:.2f}  "
                 f"aug={_aug_count_cyc}/{len(losses)}  safe={_safe_count}/{len(losses)}{marker}")'''
assert old_cyc_log in s, "P1: cycle 日志未找到"
s = s.replace(old_cyc_log, new_cyc_log, 1)
print("✅ 6) cycle 日志加 aug 执行")

p.write_text(s)
ast.parse(s)
print("✅ 7) 语法正确")
PYEOF

echo ""
echo "===== 冒烟 A: 默认关闭（use_warmup_augmentation=false，行为和 v8.0.9 一致）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['training']['use_warmup_augmentation'] = False
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1aug_smoke/default'
yaml.dump(c, open('configs/p1aug_default.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1aug_smoke/default
python -u scripts/train.py --config configs/p1aug_default.yaml 2>&1 | grep -E "warm [0-9]+/|cyc[0-9]+ ep|OA |AA " | head -8

echo ""
echo "===== 冒烟 B: 开启增强（Houston，lambda_aug=0.5）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['training']['use_warmup_augmentation'] = True
c['training']['lambda_aug'] = 0.5
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1aug_smoke/with_aug'
yaml.dump(c, open('configs/p1aug_with.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1aug_smoke/with_aug
python -u scripts/train.py --config configs/p1aug_with.yaml 2>&1 | grep -E "warm [0-9]+/|cyc[0-9]+ ep|OA |AA " | head -8

echo ""
echo "===== 冒烟 C: S→H 开启增强（它之前 T-CST 全 REJECT，aug 之前完全没生效）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['training']['use_warmup_augmentation'] = True
c['training']['lambda_aug'] = 0.5
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1aug_smoke/sh_with_aug'
yaml.dump(c, open('configs/p1aug_sh.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1aug_smoke/sh_with_aug
python -u scripts/train.py --config configs/p1aug_sh.yaml 2>&1 | grep -E "warm [0-9]+/|cyc[0-9]+ ep|OA |AA " | head -8

echo ""
echo "===== 检查 manifest 里的 aug 统计 ====="
for ds in default with_aug; do
    p="results_p1aug_smoke/${ds}/checkpoint_manifest.json"
    if [ -f "$p" ]; then
        echo "--- $ds ---"
        python -c "
import json
d = json.load(open('$p'))
wm = d.get('warmup_manifest', [])
cm = d.get('cycle_manifest', [])
print(f'  warmup aug_count: {[e.get(\"aug_executed_count\", 0) for e in wm]}')
print(f'  cycle aug_count: {[e.get(\"aug_executed_count\", 0) for e in cm]}')
"
    fi
done

echo ""
echo "✅ P1 增强解耦完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
