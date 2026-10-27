#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_after_batch1

echo "=================================================="
echo "  第 1 批修复后：3 数据集 × 3 seed 重跑"
echo "  修复项: A2 (scheduler) + A5 (make_model) + A6 (TV) + A7 (fusion init) + A8 (clean loss) + A10 (QP) + A11 (MMD bw)"
echo "  开始: $(date)"
echo "=================================================="

# ============ 生成 9 个配置 ============
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_after_batch1/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/b1_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

# ============ 跑 9 次 ============
run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_after_batch1/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[backbone\]|\[qp\]|\[risk\].*(ACCEPT|REJECT)|\[Phase|restored|OA |AA |class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/b1_${ds}_s${seed}.yaml" "${ds}_s${seed}"
    done
done

# ============ 汇总 ============
echo ""
echo "=================================================="
echo "  第 1 批修复后 vs 旧基线"
echo "=================================================="
python - << 'PYEOF' | tee results_after_batch1/SUMMARY.txt
import json, os, statistics

# 旧基线（v8.0.6，未修复）
BASE_OLD = {
    'houston':  [(80.78, 77.27), (74.98, 64.56), (77.79, 70.43)],
    'hyrank':   [(62.25, 54.14), (43.40, 51.07), (56.33, 50.17)],
    'shanghai': [(78.27, 73.30), (29.45, 43.47), (67.13, 64.26)],
}
SEEDS = [1237, 42, 2024]

def stats(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

for ds in ['houston', 'hyrank', 'shanghai']:
    print(f"\n{'='*110}")
    print(f"[{ds}]")
    print(f"{'variant':<16} {'OA':>8} {'AA':>8}  per-class")
    print('-'*110)

    # 旧基线（v8.0.6）
    old_oa = [x[0] for x in BASE_OLD[ds]]
    old_aa = [x[1] for x in BASE_OLD[ds]]
    m_old, s_old = stats(old_oa)
    print(f"{'v8.0.6 (mean±std)':<16} {m_old:>5.2f}±{s_old:>4.2f} {statistics.mean(old_aa):>5.2f}±{statistics.stdev(old_aa):>4.2f}")

    # 第 1 批修复后
    new_oa, new_aa = [], []
    for seed in SEEDS:
        p = f'results_after_batch1/{ds}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  seed={seed}: (missing)"); continue
        m = json.load(open(p))['target_metrics']
        new_oa.append(m['OA']); new_aa.append(m['AA'])
        print(f"  seed={seed:<4}: OA={m['OA']:>7.2f}  AA={m['AA']:>7.2f}  {[round(x,1) for x in m['class_accuracy']]}")

    if len(new_oa) >= 2:
        m_new, s_new = stats(new_oa)
        print(f"{'batch1 (mean±std)':<16} {m_new:>5.2f}±{s_new:>4.2f} {statistics.mean(new_aa):>5.2f}±{statistics.stdev(new_aa):>4.2f}")
        print(f"  → Δmean={m_new-m_old:+.2f}  Δstd={s_new-s_old:+.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 第 1 批修复后重跑完成"
echo "  结束: $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_after_batch1/"
echo "=================================================="
