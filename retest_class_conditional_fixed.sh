#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_retest_cc_v2

echo "=================================================="
echo "  类条件 T-CST 重测（A1 修复后，src_val 不再饱和）"
echo "  开始: $(date)"
echo "=================================================="

# 生成 3 个类条件配置（和基线唯一差别: use_class_conditional=True）
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    c = copy.deepcopy(base)
    c['tcst']['use_class_conditional'] = True
    c['tcst']['rho_a'] = 0.10
    c['tcst']['rho_b'] = 0.10
    c['training']['ckpt_mode'] = 'A'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_retest_cc_v2/{ds}'
    yaml.dump(c, open(f'configs/retest_cc2_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个类条件配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_retest_cc_v2/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[transform\]|\[qp\]|warm [0-9]+/|restored|ACCEPT|OA |AA |class[0-9]+ " | tail -20
    echo "  DONE: $tag"
}

run_one "configs/retest_cc2_houston.yaml"  "houston"
run_one "configs/retest_cc2_hyrank.yaml"   "hyrank"
run_one "configs/retest_cc2_shanghai.yaml" "shanghai"

echo ""
echo "=================================================="
echo "  类条件 vs 基线（seed=1237, ckpt_mode=A）"
echo "=================================================="
python - << 'PYEOF' | tee results_retest_cc_v2/SUMMARY.txt
import json, os
BASE = {'houston': 76.12, 'hyrank': 48.95, 'shanghai': 59.19}
BASE_1237 = {'houston': 75.99, 'hyrank': 51.98, 'shanghai': 73.69}
for ds in ['houston', 'hyrank', 'shanghai']:
    p = f'results_retest_cc_v2/{ds}/final_metrics.json'
    print(f"\n[{ds}]")
    print(f"  基线 3-seed mean = {BASE[ds]:.2f}")
    print(f"  基线 seed=1237   = {BASE_1237[ds]:.2f}")
    if not os.path.exists(p):
        print("  类条件: (missing)"); continue
    m = json.load(open(p))['target_metrics']
    d_mean = m['OA'] - BASE[ds]
    d_1237 = m['OA'] - BASE_1237[ds]
    print(f"  类条件           OA={m['OA']:.2f}  AA={m['AA']:.2f}")
    print(f"    Δ vs 3-seed mean = {d_mean:+.2f}")
    print(f"    Δ vs seed=1237   = {d_1237:+.2f}")
    print(f"  per-class={[round(x,1) for x in m['class_accuracy']]}")
    if d_1237 > 2:
        print(f"  → ✅ 涨了 {d_1237:.2f}，值得跑 3-seed")
    elif d_1237 < -2:
        print(f"  → ❌ 跌了 {abs(d_1237):.2f}，类条件负迁移")
    else:
        print(f"  → ⚠️ 变化 <2，类条件无显著影响")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 类条件重测完成  $(date)"
echo "=================================================="
