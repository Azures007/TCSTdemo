#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p2mmd_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/risk_controller.py')
s = p.read_text()

# ============ 1) backtracking_search 里加"逐类 margin 变化"诊断 ============
# 在计算 V/Vc 之后，加逐类 margin 变化统计
old_vc = '''            new_err = (correct0 & (~correct1))
            V = float(new_err.float().mean().item())
            y_t = torch.from_numpy(source_cal_labels).long()
            Vc = np.zeros(n_classes, dtype=np.float64)
            for c in range(n_classes):
                if cal_counts[c] > 0:
                    Vc[c] = float((new_err & (y_t == c)).float().sum().item()) / cal_counts[c]
            Vmax = float(Vc.max()) if Vc.size else 0.0'''
new_vc = '''            new_err = (correct0 & (~correct1))
            V = float(new_err.float().mean().item())
            y_t = torch.from_numpy(source_cal_labels).long()
            Vc = np.zeros(n_classes, dtype=np.float64)
            for c in range(n_classes):
                if cal_counts[c] > 0:
                    Vc[c] = float((new_err & (y_t == c)).float().sum().item()) / cal_counts[c]
            Vmax = float(Vc.max()) if Vc.size else 0.0

            # P2: 逐类 margin 变化（判断 MMD 降低是否伴随某些类恶化）
            true_logit1 = logits1.gather(1, y_t[:, None]).squeeze(1)
            other1 = logits1.clone(); other1.scatter_(1, y_t[:, None], -1e9)
            margin1 = true_logit1 - other1.max(1).values
            _margin_delta = (margin1 - margin0).numpy()  # 正=改善，负=恶化
            _per_class_margin = {}
            for c in range(n_classes):
                m = (y_t == c).numpy()
                if m.sum() > 0:
                    _per_class_margin[int(c)] = {
                        "mean_delta": float(_margin_delta[m].mean()),
                        "min_delta": float(_margin_delta[m].min()),
                        "frac_neg": float((_margin_delta[m] < 0).mean()),
                        "n": int(m.sum()),
                    }'''
assert old_vc in s, "P2: Vc 计算块未找到"
s = s.replace(old_vc, new_vc, 1)
print("✅ 1) 加逐类 margin 变化统计")

# ============ 2) diag 里加 _per_class_margin ============
old_diag = '''            diag = {"eta": float(eta), "V": V, "Vmax": Vmax, "Rsem": Rsem,
                    "swd_before": float(swd0), "swd_after": float(swd1),
                    "swd_drop": swd_drop, "swd_drop_rel": swd_drop_rel,
                    "Vc": Vc.tolist(),
                    "phi_norm": float(phi.norm()), "spectral": spectral_diag,
                    "spectral_ok": spec_ok, "spectral_reason": spec_reason}'''
new_diag = '''            diag = {"eta": float(eta), "V": V, "Vmax": Vmax, "Rsem": Rsem,
                    "swd_before": float(swd0), "swd_after": float(swd1),
                    "swd_drop": swd_drop, "swd_drop_rel": swd_drop_rel,
                    "Vc": Vc.tolist(),
                    "per_class_margin": _per_class_margin,
                    "phi_norm": float(phi.norm()), "spectral": spectral_diag,
                    "spectral_ok": spec_ok, "spectral_reason": spec_reason}'''
assert old_diag in s, "P2: diag 构造未找到"
s = s.replace(old_diag, new_diag, 1)
print("✅ 2) diag 里加 per_class_margin")

p.write_text(s)
ast.parse(s)
print("✅ 3) risk_controller.py 语法正确")
PYEOF

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 3) 在 cycle 里记录目标域无标签预测分布 ============
old_cycle_log = '''        cycle_logs.append({"cycle": cycle, "diag": diag,'''
new_cycle_log = '''        # P2: 记录目标域无标签预测分布（只用无标签数据，不碰 target GT）
        _tgt_pred_dist = None
        try:
            _pred_model = model
            _pred_model.eval()
            _tgt_probs_list = []
            with torch.no_grad():
                _n_eval = min(512, len(tgt_transport_p))
                _tgt_eval_idx = np.random.default_rng(cfg["output"]["seed"] + 999 + cycle).choice(
                    len(tgt_transport_p), _n_eval, replace=False)
                for _s in range(0, _n_eval, 128):
                    _batch_idx = _tgt_eval_idx[_s:_s+128]
                    _xb = torch.from_numpy(tgt_transport_p[_batch_idx]).to(device)
                    _lg = _pred_model(_xb).logits.float().cpu()
                    _tgt_probs_list.append(torch.softmax(_lg, dim=1))
                _tgt_probs = torch.cat(_tgt_probs_list, 0)
                _pred_counts = _tgt_probs.argmax(1).numpy()
                _dist = np.bincount(_pred_counts, minlength=num_classes).astype(np.float64)
                _dist = _dist / max(1.0, _dist.sum())
                # 预测熵（无标签辅助指标）
                _ent = -torch.sum(_tgt_probs * torch.log(_tgt_probs + 1e-8), dim=1).mean().item()
                _tgt_pred_dist = {
                    "pred_class_dist": _dist.tolist(),
                    "pred_entropy": float(_ent),
                    "n_samples": int(_n_eval),
                }
        except Exception as e:
            _log(f"  [P2] target pred dist failed: {e}")

        cycle_logs.append({"cycle": cycle, "diag": diag,'''
assert old_cycle_log in s, "P2: cycle_logs.append 未找到"
s = s.replace(old_cycle_log, new_cycle_log, 1)
print("✅ 4) cycle 里加目标域无标签预测分布")

# ============ 4) cycle_logs append 加 tgt_pred_dist ============
old_append = '''                            "safe_executed_total": int(_total_safe),
                            "safe_batches_total": int(_total_batches),'''
new_append = '''                            "safe_executed_total": int(_total_safe),
                            "safe_batches_total": int(_total_batches),
                            "tgt_pred_dist": _tgt_pred_dist,'''
assert old_append in s, "P2: cycle_logs append 字段未找到"
s = s.replace(old_append, new_append, 1)
print("✅ 5) cycle_logs 加 tgt_pred_dist")

p.write_text(s)
ast.parse(s)
print("✅ 6) trainer_v8.py 语法正确")
PYEOF

echo ""
echo "===== 冒烟: Houston 1 epoch（看 per_class_margin 和 tgt_pred_dist）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p2_smoke/houston'
yaml.dump(c, open('configs/p2_houston.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p2_smoke/houston
python -u scripts/train.py --config configs/p2_houston.yaml 2>&1 | grep -E "risk|P2|accepted|OA |AA " | head -15

echo ""
echo "===== 检查 training_log.jsonl 里的诊断字段 ====="
p="results_p2_smoke/houston/training_log.jsonl"
if [ -f "$p" ]; then
    python - << 'PYEOF'
import json
with open('results_p2_smoke/houston/training_log.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('stage') == 'cycle':
            tr = d.get('transport', {})
            acc = tr.get('accepted_candidate')
            if acc:
                print(f"  cycle={d.get('cycle')}")
                print(f"    accepted eta={acc.get('eta')}")
                print(f"    per_class_margin: {acc.get('per_class_margin')}")
                print(f"    tgt_pred_dist: {d.get('tgt_pred_dist')}")
            else:
                print(f"  cycle={d.get('cycle')}: no accepted candidate")
            break
PYEOF
fi

echo ""
echo "✅ P2 诊断日志补丁完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
