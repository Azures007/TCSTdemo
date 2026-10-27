#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_cc/{houston,hyrank,shanghai}

# ============ 1) 诊断当前补丁状态 ============
echo "=================================================="
echo "  诊断当前补丁状态"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
checks = {
    'spectral_transform.py: ClassSmoothSpectralTransform 已定义':
        'class ClassSmoothSpectralTransform' in Path('core/spectral_transform.py').read_text(),
    'spectral_transform.py: forward 支持 labels/class_probs':
        'def forward(self, x, phi, labels=None, class_probs=None):' in Path('core/spectral_transform.py').read_text(),
    'task_geometry.py: 传源域 labels':
        'x_phi = transform(x, phi, labels=labels_t)' in Path('core/task_geometry.py').read_text(),
    'discrepancy.py: 支持 source_labels':
        'source_labels is not None' in Path('core/discrepancy.py').read_text(),
    'risk_controller.py: 传源域 labels':
        'labels=torch.from_numpy(source_cal_labels[:64])' in Path('core/risk_controller.py').read_text(),
    'trainer_v8.py: 导入 ClassSmoothSpectralTransform':
        'from core.spectral_transform import SmoothSpectralTransform, ClassSmoothSpectralTransform'
        in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: use_class_cond 变量已定义':
        'use_class_cond' in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: MMD 调用传 source_labels':
        'source_labels=src_mmd_labels' in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: _train_step 传 labels=yb':
        'transform(xb, phi_star, labels=yb)' in Path('core/trainer_v8.py').read_text(),
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

# ============ 2) 补齐 5d（_train_step 里的 transform） ============
echo ""
echo "=================================================="
echo "  补齐 5d: _train_step transform 调用"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

if 'transform(xb, phi_star, labels=yb)' in s:
    print("✅ 5d 已存在，无需补丁")
else:
    # 尝试所有可能的写法（用户之前跑过 focal，最可能是 _ce 版本）
    candidates = [
        'loss_safe = _ce(model(transform(xb, phi_star)).logits, yb)',
        '''loss_safe = F.cross_entropy(model(transform(xb, phi_star)).logits, yb,
                                            label_smoothing=label_smoothing_value)''',
        'loss_safe = F.cross_entropy(model(transform(xb, phi_star)).logits, yb)',
    ]
    replaced = False
    for cand in candidates:
        if cand in s:
            new = cand.replace('transform(xb, phi_star)', 'transform(xb, phi_star, labels=yb)')
            s = s.replace(cand, new, 1)
            replaced = True
            print(f"✅ 匹配成功: {cand[:70]}")
            break
    if not replaced:
        # 兜底：直接替换所有未加 labels 的调用
        count = s.count('transform(xb, phi_star)')
        if count > 0:
            s = s.replace('transform(xb, phi_star)', 'transform(xb, phi_star, labels=yb)')
            print(f"✅ 兜底: 替换了 {count} 处 transform(xb, phi_star)")
        else:
            # 打印上下文供手动检查
            idx = s.find('transform(')
            if idx >= 0:
                print(f"⚠️ 未找到 transform(xb, phi_star)，当前上下文:\n{s[max(0,idx-100):idx+200]}")
            raise SystemExit("❌ 无法定位 _train_step 中的 transform 调用，请手动检查")
    p.write_text(s)

ast.parse(p.read_text())
print("✅ 语法正确")
PYEOF

# ============ 3) 再次验证所有 9 个检查项 ============
echo ""
echo "=================================================="
echo "  最终验证"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
checks = {
    'spectral_transform.py: ClassSmoothSpectralTransform':
        'class ClassSmoothSpectralTransform' in Path('core/spectral_transform.py').read_text(),
    'spectral_transform.py: forward labels':
        'def forward(self, x, phi, labels=None, class_probs=None):' in Path('core/spectral_transform.py').read_text(),
    'task_geometry.py: labels':
        'x_phi = transform(x, phi, labels=labels_t)' in Path('core/task_geometry.py').read_text(),
    'discrepancy.py: source_labels':
        'source_labels is not None' in Path('core/discrepancy.py').read_text(),
    'risk_controller.py: labels':
        'labels=torch.from_numpy(source_cal_labels[:64])' in Path('core/risk_controller.py').read_text(),
    'trainer_v8.py: import':
        'SmoothSpectralTransform, ClassSmoothSpectralTransform' in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: use_class_cond':
        'use_class_cond' in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: MMD labels':
        'source_labels=src_mmd_labels' in Path('core/trainer_v8.py').read_text(),
    'trainer_v8.py: _train_step labels':
        'transform(xb, phi_star, labels=yb)' in Path('core/trainer_v8.py').read_text(),
}
all_ok = True
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
    if not v: all_ok = False
if not all_ok:
    raise SystemExit("❌ 补丁不完整，请检查上面的 ❌ 项")
print("\n✅ 全部 9 项通过")
PYEOF

# ============ 4) Smoke test ============
echo ""
echo "=================================================="
echo "  Smoke test (Houston cc, 1 epoch)"
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
print("OK: configs/smoke_cc.yaml")
PYEOF
rm -rf results_cc/smoke
python -u scripts/train.py --config configs/smoke_cc.yaml 2>&1 | tee results_cc/smoke.log | tail -25
echo "✅ Smoke test 通过"

# ============ 5) 生成 9 个配置 ============
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

# ============ 6) 跑 9 变体 ============
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

# ============ 7) 汇总 ============
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

# ============ 8) 结束（不关机） ============
echo ""
echo "=================================================="
echo "  ✅ 类条件 T-CST 全部完成"
echo "  结束时间: $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_cc/"
echo "  汇总: ~/autodl-tmp/TCSTdemo/results_cc/SUMMARY.txt"
echo "=================================================="
