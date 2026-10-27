#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p1manifest_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) _final_eval 签名加 manifest 和 best_state ============
old_sig = '''def _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                stats, cycle_logs, ds_name, num_classes, bands, tgt_cube, patch_size):'''
new_sig = '''def _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                stats, cycle_logs, ds_name, num_classes, bands, tgt_cube, patch_size,
                manifest_data=None, best_state_for_diff=None):'''
assert old_sig in s, "P1: _final_eval 签名未找到"
s = s.replace(old_sig, new_sig, 1)
print("✅ 1) _final_eval 签名加 manifest_data + best_state_for_diff")

# ============ 2) 删掉 _final_eval 里旧的 manifest 构造（从 dir() 读的）============
old_manifest = '''    # P1.1 + P1.3: 保存 checkpoint manifest 和适应生效统计
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
'''
new_manifest = '''    # P1.1 + P1.3: manifest 由 run_tcst_v8 传入，不再从 dir() 读局部变量
    _manifest = manifest_data if manifest_data is not None else {}
    (save_dir / "checkpoint_manifest.json").write_text(
        json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
'''
assert old_manifest in s, "P1: 旧 manifest 构造未找到"
s = s.replace(old_manifest, new_manifest, 1)
print("✅ 2) _final_eval 里旧的 manifest 构造已删")

# ============ 3) _final_eval 里的 param_diff 改用参数 ============
old_diff = '''    # P1.3: 计算 final vs warmup 的参数差异范数（判断适应是否真正改变了模型）
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
'''
new_diff = '''    # P1.3: 计算 final vs warmup 的参数差异范数（用传入的 best_state_for_diff）
    _param_diff = None
    if best_state_for_diff is not None:
        try:
            import torch as _torch
            _final_sd = model.state_dict()
            _diffs = []
            for k in best_state_for_diff.keys():
                if k in _final_sd and _final_sd[k].dtype.is_floating_point:
                    _diffs.append(_torch.norm(_final_sd[k].float() - best_state_for_diff[k].float()).item())
            _param_diff = float(sum(_diffs))
        except Exception as e:
            _log(f"[P1.3] param diff calculation failed: {e}")

    if _param_diff is not None:
        _log(f"[P1.3] param diff (warmup best -> final): L1-sum-of-L2 = {_param_diff:.4f}")
        _manifest["param_diff_warmup_to_final"] = _param_diff
        (save_dir / "checkpoint_manifest.json").write_text(
            json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
'''
assert old_diff in s, "P1: 旧 param_diff 未找到"
s = s.replace(old_diff, new_diff, 1)
print("✅ 3) _final_eval 里 param_diff 改用参数")

# ============ 4) run_tcst_v8 里：构造 manifest 并传给 _final_eval ============
old_call = '''    return _final_eval(model, transform, cfg, device, save_dir,
                       best_global["src_val"], stats, cycle_logs, ds_name,
                       num_classes, bands, tgt_cube, patch_size)'''
new_call = '''    # P1.1 + P1.3: 在 run_tcst_v8 里构造完整 manifest，传给 _final_eval
    _manifest_data = {
        "warmup_manifest": warmup_ckpt_manifest,
        "cycle_manifest": cycle_ckpt_manifest,
        "final_selected_source": best_global.get("source", "warmup"),
        "final_selected_cycle": best_global.get("cycle", None),
        "final_selected_epoch": best_global.get("epoch", None),
        "final_selected_src_val": float(best_global.get("src_val", -1)),
        "warmup_best_src_val": float(best_src_val),
        "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
        "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
    }
    return _final_eval(model, transform, cfg, device, save_dir,
                       best_global["src_val"], stats, cycle_logs, ds_name,
                       num_classes, bands, tgt_cube, patch_size,
                       manifest_data=_manifest_data,
                       best_state_for_diff=best_state)'''
assert old_call in s, "P1: _final_eval 调用未找到"
s = s.replace(old_call, new_call, 1)
print("✅ 4) run_tcst_v8 构造 manifest 并传给 _final_eval")

# ============ 5) warmup_only 分支也要传 manifest（它不走 cycle，没有 manifest）============
old_warmup_return = '''        return _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                            stats, [], ds_name, num_classes, bands, tgt_cube, patch_size)'''
new_warmup_return = '''        _manifest_data = {
            "warmup_manifest": warmup_ckpt_manifest,
            "cycle_manifest": [],
            "final_selected_source": "warmup",
            "final_selected_cycle": None,
            "final_selected_epoch": None,
            "final_selected_src_val": float(best_src_val),
            "warmup_best_src_val": float(best_src_val),
            "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
            "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
            "note": "warmup_only mode: no cycles executed",
        }
        return _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                            stats, [], ds_name, num_classes, bands, tgt_cube, patch_size,
                            manifest_data=_manifest_data,
                            best_state_for_diff=best_state)'''
assert old_warmup_return in s, "P1: warmup_only return 未找到"
s = s.replace(old_warmup_return, new_warmup_return, 1)
print("✅ 5) warmup_only 分支也传 manifest")

p.write_text(s)
ast.parse(s)
print("✅ 6) 语法正确")
PYEOF

echo ""
echo "===== 冒烟 A: Houston 1 epoch（应 ACCEPT，safe>0）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1manifest_smoke/houston'
yaml.dump(c, open('configs/p1m_houston.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1manifest_smoke/houston
python -u scripts/train.py --config configs/p1m_houston.yaml 2>&1 | grep -E "cyc|P1.3|Phase C" | head -5

echo ""
echo "===== 冒烟 B: HyRANK 1 epoch（应 REJECT，safe=0）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1manifest_smoke/hyrank'
yaml.dump(c, open('configs/p1m_hyrank.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1manifest_smoke/hyrank
python -u scripts/train.py --config configs/p1m_hyrank.yaml 2>&1 | grep -E "cyc|P1.3|Phase C" | head -5

echo ""
echo "===== 检查 manifest（应该修好了）====="
for ds in houston hyrank; do
    p="results_p1manifest_smoke/${ds}/checkpoint_manifest.json"
    if [ -f "$p" ]; then
        echo "--- $ds ---"
        python -c "
import json
d = json.load(open('$p'))
print(f\"  final_selected_source: {d['final_selected_source']}\")
print(f\"  final_selected_src_val: {d['final_selected_src_val']:.2f}\")
print(f\"  warmup_best_src_val: {d['warmup_best_src_val']:.2f}\")
print(f\"  param_diff: {d.get('param_diff_warmup_to_final', 'N/A')}\")
print(f\"  cycle_manifest: {len(d['cycle_manifest'])} entries\")
for e in d['cycle_manifest']:
    print(f\"    cyc={e['cycle']} ep={e['epoch']} src_val={e['src_val']:.2f} safe={e['safe_executed_count']}/{e['safe_total_batches']} is_best={e['is_best']}\")
"
    fi
done

echo ""
echo "✅ P1.1 + P1.3 manifest 作用域修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
