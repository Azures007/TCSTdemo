#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_cc/{houston,hyrank,shanghai}

# ============ 1) 先诊断：打印当前 trainer_v8.py 的关键片段 ============
echo "=================================================="
echo "  诊断：当前 trainer_v8.py 关键片段"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
s = Path('core/trainer_v8.py').read_text()
lines = s.split('\n')

def show_around(keyword, before=2, after=6):
    for i, ln in enumerate(lines):
        if keyword in ln:
            lo, hi = max(0, i-before), min(len(lines), i+after)
            print(f"\n--- 行 {i+1} 附近（关键词: {keyword!r}）---")
            for j in range(lo, hi):
                print(f"  {j+1:4d}: {lines[j]}")
            return
    print(f"\n⚠️ 未找到关键词: {keyword!r}")

show_around('from core.spectral_transform import')
show_around('transform = SmoothSpectralTransform')
show_around('domain_gradient_mmd(transform')
show_around('src_for_mmd = ')
PYEOF

# ============ 2) 补 5a: import ============
echo ""
echo "=================================================="
echo "  补 5a: import ClassSmoothSpectralTransform"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

new_imp = 'from core.spectral_transform import SmoothSpectralTransform, ClassSmoothSpectralTransform'
if new_imp in s:
    print("✅ 5a 已存在，跳过")
else:
    old_imp = 'from core.spectral_transform import SmoothSpectralTransform'
    if old_imp in s:
        s = s.replace(old_imp, new_imp, 1)
        p.write_text(s)
        print("✅ 5a 完成：import 已替换")
    else:
        raise SystemExit("❌ 5a: 未找到 import 行")
ast.parse(p.read_text())
print("✅ 语法正确")
PYEOF

# ============ 3) 补 5b: use_class_cond + transform 选择 ============
echo ""
echo "=================================================="
echo "  补 5b: use_class_cond + transform 初始化分支"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

if 'use_class_cond' in s:
    print("✅ 5b 已存在，跳过")
else:
    old_tr = '''    transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],
                                        rho_a=cfg["tcst"]["rho_a"],
                                        rho_b=cfg["tcst"]["rho_b"]).to(device)'''
    new_tr = '''    use_class_cond = bool(cfg["tcst"].get("use_class_conditional", False))
    if use_class_cond:
        transform = ClassSmoothSpectralTransform(
            bands, num_classes=num_classes, K=cfg["tcst"]["K"],
            rho_a=cfg["tcst"]["rho_a"], rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] CLASS-CONDITIONAL: C={num_classes}, K={cfg['tcst']['K']}, dim={transform.dim}")
    else:
        transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],
                                            rho_a=cfg["tcst"]["rho_a"],
                                            rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] GLOBAL: K={cfg['tcst']['K']}, dim={transform.dim}")'''
    if old_tr in s:
        s = s.replace(old_tr, new_tr, 1)
        p.write_text(s)
        print("✅ 5b 完成：transform 初始化已分支")
    else:
        raise SystemExit("❌ 5b: transform 初始化 3 行未匹配，请把上面的诊断输出贴回来")
ast.parse(p.read_text())
print("✅ 语法正确")
PYEOF

# ============ 4) 补 5c: MMD 调用传 source_labels ============
echo ""
echo "=================================================="
echo "  补 5c: MMD 调用传 source_labels"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

if 'source_labels=src_mmd_labels' in s:
    print("✅ 5c 已存在，跳过")
else:
    # 4a) 在 src_for_mmd 定义后加 src_mmd_labels
    old_src = '        src_for_mmd = torch.from_numpy(src_tr_p[src_mmd_idx]).to(device)'
    new_src = (old_src + '\n'
               '        src_mmd_labels = torch.from_numpy(src_tr_y[src_mmd_idx]).long().to(device)')
    if old_src in s and 'src_mmd_labels = torch.from_numpy(src_tr_y' not in s:
        s = s.replace(old_src, new_src, 1)
        print("✅ 5c-1: src_mmd_labels 已添加")
    
    # 4b) 替换 MMD 调用（容错：带不带 bandwidths 都覆盖）
    import re
    patterns = [
        (r'g_D, mmd0 = domain_gradient_mmd\(transform, src_for_mmd, tgt_desc\)',
         'g_D, mmd0 = domain_gradient_mmd(transform, src_for_mmd, tgt_desc, source_labels=src_mmd_labels)'),
    ]
    replaced = False
    for pat, rep in patterns:
        if re.search(pat, s):
            s = re.sub(pat, rep, s, count=1)
            replaced = True
            print("✅ 5c-2: MMD 调用已替换")
            break
    
    if not replaced:
        # 兜底：找所有 domain_gradient_mmd(transform 调用，打印
        for i, ln in enumerate(s.split('\n')):
            if 'domain_gradient_mmd(transform' in ln:
                print(f"  找到调用（行 {i+1}）: {ln.strip()}")
        raise SystemExit("❌ 5c-2: MMD 调用未匹配，请把上面的诊断输出贴回来")
    
    p.write_text(s)
ast.parse(p.read_text())
print("✅ 语法正确")
PYEOF

# ============ 5) 最终验证 ============
echo ""
echo "=================================================="
echo "  最终验证 9 项"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
st = Path('core/spectral_transform.py').read_text()
tg = Path('core/task_geometry.py').read_text()
dc = Path('core/discrepancy.py').read_text()
rc = Path('core/risk_controller.py').read_text()
tv = Path('core/trainer_v8.py').read_text()

checks = {
    'spectral_transform: ClassSmoothSpectralTransform': 'class ClassSmoothSpectralTransform' in st,
    'spectral_transform: forward(labels/class_probs)': 'def forward(self, x, phi, labels=None, class_probs=None):' in st,
    'task_geometry: labels': 'x_phi = transform(x, phi, labels=labels_t)' in tg,
    'discrepancy: source_labels': 'source_labels is not None' in dc,
    'risk_controller: labels': 'labels=torch.from_numpy(source_cal_labels[:64])' in rc,
    'trainer_v8: import ClassSmoothSpectralTransform': 'SmoothSpectralTransform, ClassSmoothSpectralTransform' in tv,
    'trainer_v8: use_class_cond': 'use_class_cond' in tv,
    'trainer_v8: MMD source_labels': 'source_labels=src_mmd_labels' in tv,
    'trainer_v8: _train_step labels=yb': 'transform(xb, phi_star, labels=yb)' in tv,
}
ok = True
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
    if not v: ok = False
if not ok:
    raise SystemExit("❌ 有项未通过，请不要继续")
print("\n✅ 全部 9 项通过")
PYEOF

# ============ 6) Smoke test（这次才是真类条件） ============
echo ""
echo "=================================================="
echo "  Smoke test: Houston 类条件，1 epoch"
echo "  预期输出含 [transform] CLASS-CONDITIONAL: C=7, K=8, dim=112"
echo "=================================================="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['tcst']['use_class_conditional'] = True
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_cc/smoke'
yaml.dump(c, open('configs/smoke_cc.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_cc/smoke
python -u scripts/train.py --config configs/smoke_cc.yaml 2>&1 | tee results_cc/smoke.log | tail -30

# 关键检查：smoke 里有没有 CLASS-CONDITIONAL
if grep -q "CLASS-CONDITIONAL" results_cc/smoke.log; then
    echo "✅ Smoke test 确认走的是类条件 transform"
else
    echo "❌ Smoke 未出现 [transform] CLASS-CONDITIONAL，5b 没生效"
    exit 1
fi

# ============ 7) 生成 9 个配置 ============
echo ""
echo "=================================================="
echo "  生成 9 个配置"
echo "=================================================="
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_S5_ls005.yaml',
}
GROUPS = {
    'A_baseline': {'use_class_conditional': False},
    'B_cc010':    {'use_class_conditional': True, 'rho_a': 0.10, 'rho_b': 0.10},
    'C_cc015':    {'use_class_conditional': True, 'rho_a': 0.15, 'rho_b': 0.15},
}
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    for grp, flags in GROUPS.items():
        c = copy.deepcopy(base)
        for k, v in flags.items():
            c['tcst'][k] = v
        tag = f'{ds[:2].upper()}_{grp}'
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_cc/{ds}/{grp}'
        yaml.dump(c, open(f'configs/{tag}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

# ============ 8) 跑 9 变体 ============
run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL -eL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|cyc[0-9]+ ep|\[transform\]|\[risk\].*(ACCEPT|REJECT)|\[Phase|restored|OA |AA |Kappa|class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/HO_$grp.yaml" "$grp" "results_cc/houston"
done
for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/HY_$grp.yaml" "$grp" "results_cc/hyrank"
done
for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/SH_$grp.yaml" "$grp" "results_cc/shanghai"
done

# ============ 9) 汇总 ============
echo ""
echo "=================================================="
echo "  类条件 T-CST 消融汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_cc/SUMMARY.txt
import json, os
BASE = {
    'houston':  (80.78, 77.27),
    'hyrank':   (62.25, 54.14),
    'shanghai': (77.42, 72.52),
}
GROUPS = ['A_baseline', 'B_cc010', 'C_cc015']
for ds, (oa0, aa0) in BASE.items():
    print(f"\n[{ds}]  参考基线（v8.0.6）: OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'group':<14} {'OA':>8} {'dOA':>8} {'AA':>8} {'dAA':>8}  per-class")
    print('-'*115)
    for grp in GROUPS:
        p = f'results_cc/{ds}/{grp}/final_metrics.json'
        if not os.path.exists(p):
            print(f"{grp:<14} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        pc = [round(x,1) for x in m['class_accuracy']]
        print(f"{grp:<14} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}  {pc}")
PYEOF

# ============ 10) 结束 ============
echo ""
echo "=================================================="
echo "  ✅ 类条件 T-CST 全部完成"
echo "  结束: $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_cc/"
echo "=================================================="
