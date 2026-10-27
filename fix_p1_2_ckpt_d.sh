#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p12_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) _eval_source 加 return_details，总是计算 nll ============
old_eval = '''    # nan 保护：如果 acc 是 nan（空 val 或 accs 为空），返回 0.0
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        return 0.0
    if ckpt_mode == "A":
        return acc

    if ckpt_mode == "C":
        # class-balanced NLL, 越大越好（取负号）
        if num_classes is None:
            num_classes = int(y_true.max()) + 1
        y = torch.from_numpy(y_true).long()
        log_probs = torch.log_softmax(logits, dim=1)
        per_nll = -log_probs.gather(1, y[:, None]).squeeze(1).numpy()
        class_nlls = []
        for c in range(num_classes):
            m = (y_true == c)
            if m.sum() > 0:
                class_nlls.append(per_nll[m].mean())
        nll = float(np.mean(class_nlls)) if class_nlls else 0.0
        return -nll  # 负 NLL: 越大越好，量级 ~ -0.1 到 -2

    if ckpt_mode == "B":
        # accuracy + target marginal entropy tie-breaker
        # 保留 acc 为主指标, H 归一化到 [0,1] 后乘 0.1 作为辅助
        if target_patches is None or len(target_patches) == 0 or num_classes is None:
            return acc
        with torch.no_grad():
            tp = []
            for s in range(0, len(target_patches), 256):
                xb = torch.from_numpy(target_patches[s:s+256]).to(device)
                tp.append(model(xb).logits.argmax(1).cpu().numpy())
            tp = np.concatenate(tp) if tp else np.empty(0, dtype=np.int64)
        if len(tp) == 0:
            return acc
        counts = np.bincount(tp, minlength=num_classes).astype(np.float64)
        p = counts / max(1.0, counts.sum())
        H = -float(np.sum(p * np.log(p + 1e-8)))
        H_norm = H / np.log(max(2, num_classes))
        return acc + 0.1 * H_norm  # 量级 acc ~ 90, 加 0~0.1'''

new_eval = '''    # nan 保护：如果 acc 是 nan（空 val 或 accs 为空），返回 0.0
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        acc = 0.0

    # P1.2: 总是计算 class-balanced NLL（不管 ckpt_mode，manifest 需要）
    if num_classes is None:
        num_classes = int(y_true.max()) + 1 if len(y_true) > 0 else 1
    nll = float('inf')
    try:
        if len(y_true) > 0 and logits.numel() > 0:
            y = torch.from_numpy(y_true).long()
            log_probs = torch.log_softmax(logits, dim=1)
            per_nll = -log_probs.gather(1, y[:, None]).squeeze(1).numpy()
            class_nlls = []
            for c in range(num_classes):
                m = (y_true == c)
                if m.sum() > 0:
                    class_nlls.append(per_nll[m].mean())
            nll = float(np.mean(class_nlls)) if class_nlls else 0.0
    except Exception:
        nll = float('inf')

    if ckpt_mode == "A":
        score = acc
    elif ckpt_mode == "C":
        score = -nll if nll != float('inf') else -1e9
    elif ckpt_mode == "D":
        # P1.2: D 模式内层只返回 acc，最终选择在外层做（合格集 + NLL 排序）
        score = acc
    elif ckpt_mode == "B":
        if target_patches is None or len(target_patches) == 0 or num_classes is None:
            score = acc
        else:
            with torch.no_grad():
                tp = []
                for s in range(0, len(target_patches), 256):
                    xb = torch.from_numpy(target_patches[s:s+256]).to(device)
                    tp.append(model(xb).logits.argmax(1).cpu().numpy())
                tp = np.concatenate(tp) if tp else np.empty(0, dtype=np.int64)
            if len(tp) == 0:
                score = acc
            else:
                counts = np.bincount(tp, minlength=num_classes).astype(np.float64)
                p = counts / max(1.0, counts.sum())
                H = -float(np.sum(p * np.log(p + 1e-8)))
                H_norm = H / np.log(max(2, num_classes))
                score = acc + 0.1 * H_norm
    else:
        score = acc

    if return_details:
        return score, {"acc": float(acc), "nll": float(nll)}
    return score'''

assert old_eval in s, "P1.2: _eval_source 主体未找到"
s = s.replace(old_eval, new_eval, 1)
print("✅ 1) _eval_source 总是计算 nll + 支持 return_details")

# 2) _eval_source 签名加 return_details
old_sig = '''def _eval_source(model, patches, labels, device, use_cb_val=False,
                 ckpt_mode="A", num_classes=None, target_patches=None):'''
new_sig = '''def _eval_source(model, patches, labels, device, use_cb_val=False,
                 ckpt_mode="A", num_classes=None, target_patches=None,
                 return_details=False):'''
assert old_sig in s, "P1.2: _eval_source 签名未找到"
s = s.replace(old_sig, new_sig, 1)
print("✅ 2) _eval_source 签名加 return_details")

p.write_text(s)
ast.parse(s)
print("✅ 3) 语法正确")
PYEOF

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 4) warmup 循环：记录 src_nll ============
old_warm_eval = '''        src_val = _eval_source(
            model, src_val_p, src_val_y, device,
            use_cb_val=cfg["training"].get("use_cb_val", False),
            ckpt_mode=ckpt_mode, num_classes=num_classes,
            target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None)'''
new_warm_eval = '''        src_val, _warm_details = _eval_source(
            model, src_val_p, src_val_y, device,
            use_cb_val=cfg["training"].get("use_cb_val", False),
            ckpt_mode=ckpt_mode, num_classes=num_classes,
            target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None,
            return_details=True)
        src_nll = _warm_details["nll"]'''
# 注意：warmup 循环里的 src_val 调用可能有两处（warmup_only 和正常 warmup），只替换第一处
assert old_warm_eval in s, "P1.2: warmup src_val 调用未找到"
s = s.replace(old_warm_eval, new_warm_eval, 1)
print("✅ 4) warmup 循环记录 src_nll")

# 5) warmup manifest entry 加 src_nll
old_warm_entry = '''        _warm_entry = {
            "epoch": ep, "src_val": float(src_val),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,  # warmup 没有 safe 分支
            "safe_executed_count": 0,
            "is_best": False,
        }'''
new_warm_entry = '''        _warm_entry = {
            "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,
            "safe_executed_count": 0,
            "is_best": False,
        }'''
assert old_warm_entry in s, "P1.2: warmup entry 未找到"
s = s.replace(old_warm_entry, new_warm_entry, 1)
print("✅ 5) warmup manifest 加 src_nll")

p.write_text(s)
ast.parse(s)
print("✅ 6) 语法正确")
PYEOF

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 7) cycle 循环：记录 src_nll ============
old_cyc_eval = '''            src_val = _eval_source(
            model, src_val_p, src_val_y, device,
            use_cb_val=cfg["training"].get("use_cb_val", False),
            ckpt_mode=ckpt_mode, num_classes=num_classes,
            target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None)'''
new_cyc_eval = '''            src_val, _cyc_details = _eval_source(
                model, src_val_p, src_val_y, device,
                use_cb_val=cfg["training"].get("use_cb_val", False),
                ckpt_mode=ckpt_mode, num_classes=num_classes,
                target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None,
                return_details=True)
            src_nll = _cyc_details["nll"]'''
assert old_cyc_eval in s, "P1.2: cycle src_val 调用未找到"
s = s.replace(old_cyc_eval, new_cyc_eval, 1)
print("✅ 7) cycle 循环记录 src_nll")

# 8) cycle manifest entry 加 src_nll
old_cyc_entry = '''            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "safe_executed_count": _safe_count,
                "safe_total_batches": len(losses),
                "is_best": False,
            }'''
new_cyc_entry = '''            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "safe_executed_count": _safe_count,
                "safe_total_batches": len(losses),
                "is_best": False,
            }'''
assert old_cyc_entry in s, "P1.2: cycle entry 未找到"
s = s.replace(old_cyc_entry, new_cyc_entry, 1)
print("✅ 8) cycle manifest 加 src_nll")

p.write_text(s)
ast.parse(s)
print("✅ 9) 语法正确")
PYEOF

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 10) cycle 循环里实现 D 模式选择 ============
old_d_init = '''    cycle_logs = []
    # P1.1: 记录所有 cycle checkpoint 的元数据
    cycle_ckpt_manifest = []
    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict()),
                   "source": "warmup", "cycle": None, "epoch": None}'''
new_d_init = '''    cycle_logs = []
    cycle_ckpt_manifest = []
    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict()),
                   "source": "warmup", "cycle": None, "epoch": None}
    # P1.2: D 模式的独立选择（合格条件 + NLL 排序）
    _d_tolerance = float(cfg["training"].get("ckpt_d_tolerance", 2.0))
    _d_base_src_val = float(best_src_val)  # warmup best 作为基准
    _d_best_state = None
    _d_best_nll = float('inf')
    _d_best_meta = None'''
assert old_d_init in s, "P1.2: cycle_logs 初始化未找到"
s = s.replace(old_d_init, new_d_init, 1)
print("✅ 10) D 模式初始化")

# 11) 在 cycle epoch 后加 D 模式选择逻辑
old_after_best = '''            if src_val > best_global["src_val"]:
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
new_after_best = '''            if src_val > best_global["src_val"]:
                best_global["src_val"] = src_val
                best_global["state"] = copy.deepcopy(model.state_dict())
                best_global["source"] = "cycle"
                best_global["cycle"] = cycle
                best_global["epoch"] = ep
                _cyc_entry["is_best"] = True
                marker = " *BEST*"
            else:
                marker = ""
            cycle_ckpt_manifest.append(_cyc_entry)

            # P1.2: D 模式选择（合格条件 + NLL 排序，与 best_global 独立）
            if ckpt_mode == "D":
                if src_val >= _d_base_src_val - _d_tolerance:
                    if src_nll < _d_best_nll:
                        _d_best_nll = src_nll
                        _d_best_state = copy.deepcopy(model.state_dict())
                        _d_best_meta = {"cycle": cycle, "epoch": ep,
                                        "src_val": float(src_val), "src_nll": float(src_nll)}'''
assert old_after_best in s, "P1.2: cycle best 判断未找到"
s = s.replace(old_after_best, new_after_best, 1)
print("✅ 11) cycle 循环加 D 模式选择")

# 12) 最终选择逻辑加 D 模式分支
old_final = '''    force_final = bool(cfg["training"].get("force_final_epoch", False))  # v802: 统一为 False
    if force_final:
        _log(f"\\n[Opt3] force_final_epoch=True, keeping final model "
             f"(src_val={best_global['src_val']:.2f} at best)")
    else:
        model.load_state_dict(best_global["state"])
        _log(f"\\n[Phase C] restored best src_val checkpoint "
             f"(src_val={best_global['src_val']:.2f})")'''
new_final = '''    force_final = bool(cfg["training"].get("force_final_epoch", False))
    _d_selected = False
    if ckpt_mode == "D" and _d_best_state is not None:
        model.load_state_dict(_d_best_state)
        best_global["src_val"] = _d_best_meta["src_val"]
        best_global["source"] = "D_cycle"
        best_global["cycle"] = _d_best_meta["cycle"]
        best_global["epoch"] = _d_best_meta["epoch"]
        _d_selected = True
        _log(f"\\n[P1.2] ckpt_mode=D: selected cycle {_d_best_meta['cycle']} ep {_d_best_meta['epoch']} "
             f"(src_val={_d_best_meta['src_val']:.2f}, nll={_d_best_meta['src_nll']:.4f}, "
             f"tol={_d_tolerance}, base={_d_base_src_val:.2f})")
    elif ckpt_mode == "D" and _d_best_state is None:
        _log(f"\\n[P1.2] ckpt_mode=D: no qualified cycle checkpoint "
             f"(all cycles have src_val < {_d_base_src_val - _d_tolerance:.2f}), keeping warmup")

    if not _d_selected:
        if force_final:
            _log(f"\\n[Opt3] force_final_epoch=True, keeping final model "
                 f"(src_val={best_global['src_val']:.2f} at best)")
        else:
            model.load_state_dict(best_global["state"])
            _log(f"\\n[Phase C] restored best src_val checkpoint "
                 f"(src_val={best_global['src_val']:.2f})")'''
assert old_final in s, "P1.2: 最终选择逻辑未找到"
s = s.replace(old_final, new_final, 1)
print("✅ 12) 最终选择逻辑加 D 模式分支")

# 13) manifest 加 D 模式信息
old_manifest = '''    _manifest_data = {
        "warmup_manifest": warmup_ckpt_manifest,
        "cycle_manifest": cycle_ckpt_manifest,
        "final_selected_source": best_global.get("source", "warmup"),
        "final_selected_cycle": best_global.get("cycle", None),
        "final_selected_epoch": best_global.get("epoch", None),
        "final_selected_src_val": float(best_global.get("src_val", -1)),
        "warmup_best_src_val": float(best_src_val),
        "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
        "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
    }'''
new_manifest = '''    _manifest_data = {
        "warmup_manifest": warmup_ckpt_manifest,
        "cycle_manifest": cycle_ckpt_manifest,
        "final_selected_source": best_global.get("source", "warmup"),
        "final_selected_cycle": best_global.get("cycle", None),
        "final_selected_epoch": best_global.get("epoch", None),
        "final_selected_src_val": float(best_global.get("src_val", -1)),
        "warmup_best_src_val": float(best_src_val),
        "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
        "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
        "ckpt_d_tolerance": _d_tolerance if ckpt_mode == "D" else None,
        "ckpt_d_base_src_val": _d_base_src_val if ckpt_mode == "D" else None,
        "ckpt_d_selected": {
            "cycle": _d_best_meta["cycle"], "epoch": _d_best_meta["epoch"],
            "src_val": _d_best_meta["src_val"], "src_nll": _d_best_meta["src_nll"],
        } if _d_best_meta is not None else None,
    }'''
assert old_manifest in s, "P1.2: manifest 构造未找到"
s = s.replace(old_manifest, new_manifest, 1)
print("✅ 13) manifest 加 D 模式信息")

p.write_text(s)
ast.parse(s)
print("✅ 14) 语法正确")
PYEOF

echo ""
echo "===== 冒烟 A: Houston 完整配置 ckpt_mode=A（基线）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p12_smoke/houston_A'
yaml.dump(c, open('configs/p12_houston_A.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p12_smoke/houston_A
python -u scripts/train.py --config configs/p12_houston_A.yaml 2>&1 | grep -E "Phase C|P1.2|param diff" | head -5

echo ""
echo "===== 冒烟 B: Houston 完整配置 ckpt_mode=D（新策略）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'D'
c['training']['ckpt_d_tolerance'] = 2.0
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p12_smoke/houston_D'
yaml.dump(c, open('configs/p12_houston_D.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p12_smoke/houston_D
python -u scripts/train.py --config configs/p12_houston_D.yaml 2>&1 | grep -E "Phase C|P1.2|param diff" | head -5

echo ""
echo "===== manifest 对比（A vs D）====="
for mode in A D; do
    p="results_p12_smoke/houston_${mode}/checkpoint_manifest.json"
    if [ -f "$p" ]; then
        echo "--- houston_${mode} ---"
        python -c "
import json
d = json.load(open('$p'))
print(f\"  source: {d['final_selected_source']}\")
print(f\"  src_val: {d['final_selected_src_val']:.2f}\")
print(f\"  warmup_best: {d['warmup_best_src_val']:.2f}\")
if d.get('ckpt_d_selected'):
    print(f\"  D selected: {d['ckpt_d_selected']}\")
"
    fi
done

echo ""
echo "===== 对比 target OA（A vs D）====="
for mode in A D; do
    p="results_p12_smoke/houston_${mode}/final_metrics.json"
    if [ -f "$p" ]; then
        python -c "
import json
m = json.load(open('$p'))['target_metrics']
print(f'  houston_{mode}: OA={m[\"OA\"]:.2f}  AA={m[\"AA\"]:.2f}  per-class={[round(x,1) for x in m[\"class_accuracy\"]]}')
"
    fi
done

echo ""
echo "✅ P1.2 ckpt_mode=D 完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
