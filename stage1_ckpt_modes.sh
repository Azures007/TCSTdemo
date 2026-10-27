#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_ckpt_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ 备份 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ 1) 改造 _eval_source，支持 3 种模式 ============
old_eval = '''def _eval_source(model, patches, labels, device, use_cb_val=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            preds.append(model(xb).logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    y_true = np.asarray(labels)
    
    if not use_cb_val:
        correct = (preds == y_true).sum()
        return 100.0 * correct / max(1, len(patches))
    
    # CB-Val：各类别准确率的平均
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = (y_true == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == c).sum() / mask.sum())
    return 100.0 * float(np.mean(accs))'''

new_eval = '''def _eval_source(model, patches, labels, device, use_cb_val=False,
                 ckpt_mode="A", num_classes=None, target_patches=None):
    """
    ckpt_mode:
      "A" = accuracy (original), higher is better
      "B" = accuracy + target marginal entropy tie-breaker, higher is better
      "C" = class-balanced negative NLL, higher is better
    Returns a single float score (higher is always better).
    """
    model.eval()
    preds = []
    all_logits = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            lg = model(xb).logits.float().cpu()
            all_logits.append(lg)
            preds.append(lg.argmax(1).numpy())
    preds = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    logits = torch.cat(all_logits, 0) if all_logits else torch.empty(0)
    y_true = np.asarray(labels)

    # 基础 accuracy
    if not use_cb_val:
        acc = 100.0 * float((preds == y_true).sum()) / max(1, len(patches))
    else:
        classes = np.unique(y_true)
        accs = []
        for c in classes:
            mask = (y_true == c)
            if mask.sum() > 0:
                accs.append((preds[mask] == c).sum() / mask.sum())
        acc = 100.0 * float(np.mean(accs))

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

assert old_eval in s, "_eval_source 未找到"
s = s.replace(old_eval, new_eval, 1)

# ============ 2) 训练循环里传 ckpt_mode 和 target_patches ============
# 找到 warmup 中的调用
old_call_warm = '        src_val = _eval_source(model, src_val_p, src_val_y, device, use_cb_val=cfg["training"].get("use_cb_val", False))'
new_call_warm = '''        src_val = _eval_source(
            model, src_val_p, src_val_y, device,
            use_cb_val=cfg["training"].get("use_cb_val", False),
            ckpt_mode=ckpt_mode, num_classes=num_classes,
            target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None)'''
count_warm = s.count(old_call_warm)
s = s.replace(old_call_warm, new_call_warm)

# 找到 cycle 中的调用（有额外的缩进）
old_call_cyc = '            src_val = _eval_source(model, src_val_p, src_val_y, device, use_cb_val=cfg["training"].get("use_cb_val", False))'
new_call_cyc = '''            src_val = _eval_source(
                model, src_val_p, src_val_y, device,
                use_cb_val=cfg["training"].get("use_cb_val", False),
                ckpt_mode=ckpt_mode, num_classes=num_classes,
                target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None)'''
count_cyc = s.count(old_call_cyc)
s = s.replace(old_call_cyc, new_call_cyc)

# 在 run_tcst_v8 开头定义 ckpt_mode
old_def = '''    run_mode = cfg["training"].get("run_mode", "full")
    ds_name = cfg["dataset"]["name"]'''
new_def = '''    run_mode = cfg["training"].get("run_mode", "full")
    ds_name = cfg["dataset"]["name"]
    ckpt_mode = cfg["training"].get("ckpt_mode", "A")  # A/B/C
    _log(f"[ckpt] mode={ckpt_mode} (A=accuracy, B=acc+target_entropy, C=class_balanced_negNLL)")'''
assert old_def in s, "run_mode 位置未找到"
s = s.replace(old_def, new_def, 1)

p.write_text(s)
ast.parse(s)
print(f"✅ _eval_source 支持 3 模式, warmup 调用 {count_warm} 处, cycle 调用 {count_cyc} 处")
PYEOF

# ============ 3) 验证 ============
echo ""
echo "=== 验证 ==="
python - << 'PYEOF'
from pathlib import Path
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'ckpt_mode 变量定义': 'ckpt_mode = cfg["training"].get("ckpt_mode"' in tv,
    'A 模式: 返回 acc': 'if ckpt_mode == "A":\n        return acc' in tv,
    'B 模式: target entropy': 'acc + 0.1 * H_norm' in tv,
    'C 模式: class-balanced NLL': 'class_nlls.append' in tv and 'return -nll' in tv,
    'warmup 传 ckpt_mode': 'ckpt_mode=ckpt_mode, num_classes=num_classes' in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

# 轻量 smoke：只验证 A 模式 (C 模式需要真训练)
echo ""
echo "=== 轻量 smoke (A 模式，1 epoch) ==="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ckpt_smoke/A'
yaml.dump(c, open('configs/ckpt_smoke_A.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_ckpt_smoke/A
python -u scripts/train.py --config configs/ckpt_smoke_A.yaml 2>&1 | grep -E "\[ckpt\]|warm [0-9]+|restored|OA |AA " | head -10
echo ""
echo "✅ 阶段 1 完成（3 模式代码 + A 模式 smoke 通过）"
echo "   回滚: rm -rf core && mv $BACKUP core"
