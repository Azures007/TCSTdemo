#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p11p13_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) _train_step 返回更多信息（clean_loss, safe_loss, safe_executed）============
old_ts = '''        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        return float(loss.detach())'''
new_ts = '''        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        # P1.3: 返回 clean/safe 分解，便于判断 T-CST 是否真的执行
        _safe_exec = (phi_star is not None and float(phi_star.abs().sum()) > 1e-8)
        return {
            "loss": float(loss.detach()),
            "clean_loss": float(loss_clean.detach()),
            "safe_loss": float(loss_safe.detach()) if _safe_exec else None,
            "safe_executed": _safe_exec,
        }'''
assert old_ts in s, "P1.3: _train_step 尾部未找到"
s = s.replace(old_ts, new_ts, 1)
print("✅ 1) _train_step 返回 clean/safe 分解")

# ============ 2) warmup 循环：保存所有 warmup checkpoint 的元数据 ============
old_warm = '''    best_src_val = -1.0
    best_state = None'''
new_warm = '''    best_src_val = -1.0
    best_state = None
    # P1.1: 记录所有 warmup checkpoint 的元数据（不保存 state_dict，只记指标）
    warmup_ckpt_manifest = []'''
assert old_warm in s, "P1.1: warmup 循环开头未找到"
s = s.replace(old_warm, new_warm, 1)

old_warm_loop = '''        if src_val > best_src_val:
            best_src_val = src_val
            best_state = copy.deepcopy(model.state_dict())'''
new_warm_loop = '''        # P1.1: 记录每个 warmup epoch 的指标
        _warm_entry = {
            "epoch": ep, "src_val": float(src_val),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,  # warmup 没有 safe 分支
            "safe_executed_count": 0,
            "is_best": False,
        }
        if src_val > best_src_val:
            best_src_val = src_val
            best_state = copy.deepcopy(model.state_dict())
            _warm_entry["is_best"] = True
        warmup_ckpt_manifest.append(_warm_entry)'''
assert old_warm_loop in s, "P1.1: warmup best 判断未找到"
s = s.replace(old_warm_loop, new_warm_loop, 1)
print("✅ 2) warmup 循环记录每个 epoch 的元数据")

# ============ 3) warmup 循环：losses 现在是 dict 列表，改 np.mean ============
old_log = '''        _log(f"  warm {ep:03d}/{cfg['training']['epochs_warmup']}  "
             f"loss={np.mean(losses):.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"src_val={src_val:.2f}  best={best_src_val:.2f}  "
             f"t={(time.time()-t0)/60:.1f}m")'''
new_log = '''        _log(f"  warm {ep:03d}/{cfg['training']['epochs_warmup']}  "
             f"loss={np.mean([l['loss'] for l in losses]):.4f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"src_val={src_val:.2f}  best={best_src_val:.2f}  "
             f"t={(time.time()-t0)/60:.1f}m")'''
assert old_log in s, "P1.1: warmup 日志未找到"
s = s.replace(old_log, new_log, 1)

# ============ 4) cycle 循环：记录每个 epoch 的元数据 + safe 执行统计 ============
old_cycle_init = '''    cycle_logs = []
    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict())}'''
new_cycle_init = '''    cycle_logs = []
    # P1.1: 记录所有 cycle checkpoint 的元数据
    cycle_ckpt_manifest = []
    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict()),
                   "source": "warmup", "cycle": None, "epoch": None}'''
assert old_cycle_init in s, "P1.1: cycle_logs 初始化未找到"
s = s.replace(old_cycle_init, new_cycle_init, 1)
print("✅ 3) cycle 循环加 cycle_ckpt_manifest")

old_cycle_loop = '''        for ep in range(1, cfg["training"]["epochs_per_cycle"] + 1):
            losses = []
            for xb_batch, yb_batch in _iter_cycle():
                losses.append(_train_step(xb_batch, yb_batch, phi_star,
                                          cfg["tcst"]["lambda_safe"]))'''
new_cycle_loop = '''        for ep in range(1, cfg["training"]["epochs_per_cycle"] + 1):
            losses = []
            for xb_batch, yb_batch in _iter_cycle():
                losses.append(_train_step(xb_batch, yb_batch, phi_star,
                                          cfg["tcst"]["lambda_safe"]))
            # P1.3: 统计本 epoch 的 safe 执行次数
            _safe_count = sum(1 for l in losses if l.get("safe_executed", False))
            _safe_losses = [l["safe_loss"] for l in losses if l.get("safe_loss") is not None]
            _clean_losses = [l["clean_loss"] for l in losses]'''
assert old_cycle_loop in s, "P1.1: cycle epoch 循环未找到"
s = s.replace(old_cycle_loop, new_cycle_loop, 1)

old_cycle_best = '''            if src_val > best_global["src_val"]:
                best_global["src_val"] = src_val
                best_global["state"] = copy.deepcopy(model.state_dict())
                marker = " *BEST*"
            else:
                marker = ""'''
new_cycle_best = '''            # P1.1: 记录每个 cycle epoch 的元数据
            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "safe_executed_count": _safe_count,
                "safe_total_batches": len(losses),
                "is_best": False,
            }
            if src_val > best_global["src_val"]:
                best_global["src_val"] = src_val
                best_global["state"] = copy.deepcopy(model.state_dict())
                best_global["source"] = "cycle"
                best_global["cycle"] = cycle
                best_global["epoch"] = ep
                _cyc_entry["is_best"] = True
                marker = " *BEST*"
            else:
                marker = ""
            cycle_ckpt_manifest.append(_cyc_entry)'''
assert old_cycle_best in s, "P1.1: cycle best 判断未找到"
s = s.replace(old_cycle_best, new_cycle_best, 1)
print("✅ 4) cycle 循环记录每个 epoch 的元数据 + safe 执行统计")

# ============ 5) cycle 日志：loss 现在是 dict，改 np.mean ============
old_cyc_log = '''            _log(f"    cyc{cycle} ep{ep:02d}  loss={np.mean(losses):.4f}  "
                 f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                 f"src_val={src_val:.2f}  best={best_global['src_val']:.2f}{marker}")'''
new_cyc_log = '''            _log(f"    cyc{cycle} ep{ep:02d}  loss={np.mean([l['loss'] for l in losses]):.4f}  "
                 f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                 f"src_val={src_val:.2f}  best={best_global['src_val']:.2f}  "
                 f"safe={_safe_count}/{len(losses)}{marker}")'''
assert old_cyc_log in s, "P1.1: cycle 日志未找到"
s = s.replace(old_cyc_log, new_cyc_log, 1)
print("✅ 5) cycle 日志加 safe 执行次数")

# ============ 6) cycle_logs 增加 safe 统计和 checkpoint manifest ============
old_cyc_append = '''        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "phi_star_full": _a12_phi_full,
                            "d_star_full": _a12_d_full,
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
new_cyc_append = '''        # P1.3: 本 cycle 的 safe 执行统计
        _cycle_safe_counts = [e for e in cycle_ckpt_manifest if e["cycle"] == cycle]
        _total_safe = sum(e["safe_executed_count"] for e in _cycle_safe_counts)
        _total_batches = sum(e["safe_total_batches"] for e in _cycle_safe_counts)
        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "best_source": best_global.get("source", "warmup"),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "phi_star_full": _a12_phi_full,
                            "d_star_full": _a12_d_full,
                            "accepted_eta": float(diag.get("accepted_eta", 0.0)),
                            "safe_executed_total": int(_total_safe),
                            "safe_batches_total": int(_total_batches),
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
assert old_cyc_append in s, "P1.3: cycle_logs.append 未找到"
s = s.replace(old_cyc_append, new_cyc_append, 1)
print("✅ 6) cycle_logs 加 safe 执行统计和 best_source")

# ============ 7) 最终保存：checkpoint manifest + 参数差异 ============
old_save = '''    torch.save({"model": model.state_dict(), "transform": transform.state_dict(),
                "args": cfg, "metrics": metrics},
               save_dir / "final_state.pt")'''
new_save = '''    # P1.1 + P1.3: 保存 checkpoint manifest 和适应生效统计
    _manifest = {
        "warmup_manifest": warmup_ckpt_manifest if 'warmup_ckpt_manifest' in dir() else [],
        "cycle_manifest": cycle_ckpt_manifest if 'cycle_ckpt_manifest' in dir() else [],
        "final_selected_source": best_global.get("source", "warmup") if 'best_global' in dir() else "warmup",
        "final_selected_cycle": best_global.get("cycle", None) if 'best_global' in dir() else None,
        "final_selected_epoch": best_global.get("epoch", None) if 'best_global' in dir() else None,
        "final_selected_src_val": float(best_global.get("src_val", -1)) if 'best_global' in dir() else -1,
        "warmup_best_src_val": float(best_src_val) if 'best_src_val' in dir() else -1,
        "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
        "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
    }
    (save_dir / "checkpoint_manifest.json").write_text(
        json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.save({"model": model.state_dict(), "transform": transform.state_dict(),
                "args": cfg, "metrics": metrics},
               save_dir / "final_state.pt")'''
assert old_save in s, "P1.1: torch.save 未找到"
s = s.replace(old_save, new_save, 1)
print("✅ 7) 最终保存 checkpoint_manifest.json")

# ============ 8) 最终保存：参数差异范数（warmup best vs final）============
old_save2 = '''    _log(f"Saved: {save_dir / 'final_metrics.json'}")'''
new_save2 = '''    # P1.3: 计算 final vs warmup 的参数差异范数（判断适应是否真正改变了模型）
    _param_diff = None
    try:
        if 'best_state' in dir() and best_state is not None:
            import torch as _torch
            _final_sd = model.state_dict()
            _diffs = []
            for k in best_state.keys():
                if k in _final_sd and _final_sd[k].dtype.is_floating_point:
                    _diffs.append(_torch.norm(_final_sd[k].float() - best_state[k].float()).item())
            _param_diff = float(sum(_diffs))
    except Exception as e:
        _log(f"[P1.3] param diff calculation failed: {e}")

    if _param_diff is not None:
        _log(f"[P1.3] param diff (warmup best -> final): L1-sum-of-L2 = {_param_diff:.4f}")
        _manifest["param_diff_warmup_to_final"] = _param_diff
        (save_dir / "checkpoint_manifest.json").write_text(
            json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    _log(f"Saved: {save_dir / 'final_metrics.json'}")'''
assert old_save2 in s, "P1.3: Saved 日志未找到"
s = s.replace(old_save2, new_save2, 1)
print("✅ 8) 参数差异范数日志")

p.write_text(s)
ast.parse(s)
print("✅ 语法正确")
PYEOF

echo ""
echo "===== 验证 ====="
grep -c "warmup_ckpt_manifest\|cycle_ckpt_manifest\|checkpoint_manifest\|safe_executed" core/trainer_v8.py
python -c "import ast; ast.parse(open('core/trainer_v8.py').read()); print('✅ 语法正确')"

echo ""
echo "===== 冒烟 A: Houston 1 epoch（应该 ACCEPT）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p11_smoke/houston'
yaml.dump(c, open('configs/p11_houston.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p11_smoke/houston
python -u scripts/train.py --config configs/p11_houston.yaml 2>&1 | grep -E "warm|cyc|BEST|Phase|P1.3|Saved" | head -15

echo ""
echo "===== 冒烟 B: HyRANK 1 epoch（应该 REJECT）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p11_smoke/hyrank'
yaml.dump(c, open('configs/p11_hyrank.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p11_smoke/hyrank
python -u scripts/train.py --config configs/p11_hyrank.yaml 2>&1 | grep -E "warm|cyc|BEST|Phase|P1.3|Saved" | head -15

echo ""
echo "===== 检查 checkpoint_manifest.json ====="
for ds in houston hyrank; do
    p="results_p11_smoke/${ds}/checkpoint_manifest.json"
    if [ -f "$p" ]; then
        echo "--- $p ---"
        python -c "
import json
d = json.load(open('$p'))
print(f\"  final_selected_source: {d['final_selected_source']}\")
print(f\"  warmup_best_src_val: {d['warmup_best_src_val']}\")
print(f\"  final_selected_src_val: {d['final_selected_src_val']}\")
print(f\"  param_diff: {d.get('param_diff_warmup_to_final', 'N/A')}\")
print(f\"  cycle_manifest entries: {len(d['cycle_manifest'])}\")
for e in d['cycle_manifest']:
    print(f\"    cyc={e['cycle']} ep={e['epoch']} src_val={e['src_val']} safe={e['safe_executed_count']}/{e['safe_total_batches']} is_best={e['is_best']}\")
"
    fi
done

echo ""
echo "✅ P1.1 + P1.3 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
