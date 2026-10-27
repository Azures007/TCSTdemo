#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

# ============ 0) 备份 core ============
BACKUP="core_backup_stage3_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 已备份到 $BACKUP"
echo "   如果脚本崩溃，回滚: rm -rf core && mv $BACKUP core"

# ============ 1) 植入 EMA + 伪标签 ============
echo ""
echo "=== 植入 EMA teacher + 伪标签 self-training ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# --- 1a) model 初始化后加 EMA teacher + pseudo state ---
old1 = '''    model = _make_model(bands, num_classes, device)
    transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],'''
new1 = '''    model = _make_model(bands, num_classes, device)
    # === EMA teacher & pseudo-label state ===
    use_ema = bool(cfg["training"].get("use_ema", False))
    use_pseudo = bool(cfg["training"].get("use_pseudo_label", False))
    ema_decay = float(cfg["training"].get("ema_decay", 0.999))
    ema_teacher = copy.deepcopy(model) if use_ema else None
    if use_ema:
        for _p in ema_teacher.parameters():
            _p.requires_grad_(False)
        _log(f"[EMA] teacher enabled, decay={ema_decay}")
    pseudo_state = {"xb": None, "yb": None, "n_selected": 0}
    if use_pseudo:
        _log(f"[Pseudo] self-training enabled, threshold={cfg['training'].get('pseudo_threshold', 0.95)}, weight={cfg['training'].get('pseudo_weight', 0.5)}")
    transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],'''
assert old1 in s, "model 初始化位置未找到"
s = s.replace(old1, new1, 1)

# --- 1b) _train_step 尾部加 pseudo loss + EMA update ---
old2 = '''        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        return float(loss.detach())'''
new2 = '''        # === Pseudo-label loss ===
        if use_pseudo and pseudo_state["xb"] is not None:
            pb_size = xb.size(0)
            idx_p = np.random.choice(len(pseudo_state["xb"]), pb_size, replace=True)
            p_xb = torch.from_numpy(pseudo_state["xb"][idx_p]).to(device)
            p_yb = torch.from_numpy(pseudo_state["yb"][idx_p]).to(device)
            pseudo_w = float(cfg["training"].get("pseudo_weight", 0.5))
            loss_pseudo = pseudo_w * F.cross_entropy(
                model(p_xb).logits, p_yb,
                label_smoothing=label_smoothing_value)
            loss = loss + loss_pseudo
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        # === EMA update ===
        if use_ema and ema_teacher is not None:
            with torch.no_grad():
                for _ps, _pt in zip(model.parameters(), ema_teacher.parameters()):
                    _pt.data.mul_(ema_decay).add_(_ps.data, alpha=1.0 - ema_decay)
        return float(loss.detach())'''
assert old2 in s, "_train_step 尾部未找到"
s = s.replace(old2, new2, 1)

# --- 1c) 每个 cycle 开始时生成伪标签 ---
old3 = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()

        src_mmd_idx = _class_balanced_subsample(src_tr_y, 128,'''
new3 = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()

        # === Generate pseudo-labels from target transport patches ===
        if use_pseudo:
            _pred_model = ema_teacher if use_ema else model
            _pred_model.eval()
            _thr = float(cfg["training"].get("pseudo_threshold", 0.95))
            with torch.no_grad():
                _probs_list = []
                for _s in range(0, len(tgt_transport_p), 128):
                    _xb = torch.from_numpy(tgt_transport_p[_s:_s+128]).to(device)
                    _logits = _pred_model(_xb).logits
                    _probs_list.append(torch.softmax(_logits.float(), dim=1).cpu())
                _probs = torch.cat(_probs_list, 0)
                _conf, _pl = _probs.max(1)
                _mask = (_conf >= _thr).numpy()
                if _mask.any():
                    pseudo_state["xb"] = tgt_transport_p[_mask]
                    pseudo_state["yb"] = _pl[_mask].numpy().astype(np.int64)
                else:
                    pseudo_state["xb"] = None
                    pseudo_state["yb"] = None
                pseudo_state["n_selected"] = int(_mask.sum())
            _log(f"  [Pseudo] selected {pseudo_state['n_selected']} / {len(tgt_transport_p)} (thr={_thr:.2f}, teacher={'ema' if use_ema else 'student'})")

        src_mmd_idx = _class_balanced_subsample(src_tr_y, 128,'''
assert old3 in s, "cycle 循环开头未找到"
s = s.replace(old3, new3, 1)

p.write_text(s)
ast.parse(s)
print("✅ EMA + 伪标签 已植入，语法正确")
PYEOF

# ============ 2) 生成 9 个配置 ============
mkdir -p results_stage3/{houston,hyrank,shanghai}
python - << 'PYEOF'
import yaml, copy

BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_S5_ls005.yaml',
}
GROUPS = {
    'A_ema':    {'use_ema': True},
    'B_pseudo': {'use_pseudo_label': True},
    'C_both':   {'use_ema': True, 'use_pseudo_label': True},
}
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    for grp, flags in GROUPS.items():
        c = copy.deepcopy(base)
        for k, v in flags.items():
            c['training'][k] = v
        c['training']['ema_decay'] = 0.999
        c['training']['pseudo_threshold'] = 0.95
        c['training']['pseudo_weight'] = 0.5
        tag = f'{ds[:2].upper()}_{grp}'
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_stage3/{ds}/{grp}'
        yaml.dump(c, open(f'configs/{tag}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

# ============ 3) Smoke test: 1 epoch 确认代码能跑 ============
echo ""
echo "=== Smoke test (Houston EMA+pseudo, 1 epoch) ==="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/HO_C_both.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_stage3/houston/_smoke'
yaml.dump(c, open('configs/smoke_both.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_stage3/houston/_smoke
python -u scripts/train.py --config configs/smoke_both.yaml 2>&1 | tail -25
echo "✅ Smoke test 通过"

# ============ 4) 跑 9 个变体 ============
run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL -eL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|cyc[0-9]+ ep|\[EMA\]|\[Pseudo\]|\[risk\].*(ACCEPT|REJECT)|\[Phase|restored|OA |AA |Kappa|class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

# 组 A: EMA only
run_one "configs/HO_A_ema.yaml"    "A_ema"    "results_stage3/houston"
run_one "configs/HY_A_ema.yaml"    "A_ema"    "results_stage3/hyrank"
run_one "configs/SH_A_ema.yaml"    "A_ema"    "results_stage3/shanghai"

# 组 B: Pseudo only
run_one "configs/HO_B_pseudo.yaml" "B_pseudo" "results_stage3/houston"
run_one "configs/HY_B_pseudo.yaml" "B_pseudo" "results_stage3/hyrank"
run_one "configs/SH_B_pseudo.yaml" "B_pseudo" "results_stage3/shanghai"

# 组 C: EMA + Pseudo
run_one "configs/HO_C_both.yaml"   "C_both"   "results_stage3/houston"
run_one "configs/HY_C_both.yaml"   "C_both"   "results_stage3/hyrank"
run_one "configs/SH_C_both.yaml"   "C_both"   "results_stage3/shanghai"

# ============ 5) 汇总（同时打印到屏幕 + 写入文件）============
echo ""
echo "=================================================="
echo "  第三层消融汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_stage3/SUMMARY.txt
import json, os
BASE = {
    'houston':  (80.78, 77.27),
    'hyrank':   (62.25, 54.14),
    'shanghai': (77.42, 72.52),
}
GROUPS = ['A_ema', 'B_pseudo', 'C_both']
for ds, (oa0, aa0) in BASE.items():
    print(f"\n[{ds}]  基线（第二轮最优）: OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'group':<12} {'OA':>8} {'dOA':>8} {'AA':>8} {'dAA':>8}  per-class")
    print('-'*110)
    for grp in GROUPS:
        p = f'results_stage3/{ds}/{grp}/final_metrics.json'
        if not os.path.exists(p):
            print(f"{grp:<12} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        pc = [round(x,1) for x in m['class_accuracy']]
        print(f"{grp:<12} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}  {pc}")
PYEOF

# ============ 6) 结束（不关机）============
echo ""
echo "=================================================="
echo "  ✅ 第三层全部完成"
echo "  结束时间: $(date)"
echo "  结果目录: ~/autodl-tmp/TCSTdemo/results_stage3/"
echo "  汇总文件: ~/autodl-tmp/TCSTdemo/results_stage3/SUMMARY.txt"
echo "  core 备份: ~/autodl-tmp/TCSTdemo/$BACKUP"
echo "  如需回滚: rm -rf core && mv $BACKUP core"
echo "=================================================="
