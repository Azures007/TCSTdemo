#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_100perclass

echo "=================================================="
echo "  100/类 × 3 seed × 3 数据集"
echo "  开始: $(date)"
echo "=================================================="

# ============ 1) 生成 9 个配置 ============
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
        c['training']['per_class_train_cap'] = 100
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_100perclass/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/100pc_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个 100/类 配置已生成")
PYEOF

# ============ 2) 跑 9 变体 ============
run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL -eL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "source-split|per-class train counts|warm [0-9]+/|restored|OA |AA |class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/100pc_${ds}_s${seed}.yaml" "${ds}_s${seed}" "results_100perclass"
    done
done

# ============ 3) 汇总 + 与 150/类 对比 ============
echo ""
echo "=================================================="
echo "  100/类 vs 150/类 对比"
echo "=================================================="
python - << 'PYEOF' | tee results_100perclass/SUMMARY.txt
import json, os, statistics

# 150/类 的 3-seed 结果（已知）
BASE_150 = {
    'houston':  [(80.78, 77.27), (74.98, 64.56), (77.79, 70.43)],
    'hyrank':   [(62.25, 54.14), (43.40, 51.07), (56.33, 50.17)],
    'shanghai': [(78.27, 73.30), (29.45, 43.47), (67.13, 64.26)],
}
SEEDS = [1237, 42, 2024]

def stats(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

for ds in ['houston', 'hyrank', 'shanghai']:
    oas_150 = [x[0] for x in BASE_150[ds]]
    aas_150 = [x[1] for x in BASE_150[ds]]
    m150, s150 = stats(oas_150)

    print(f"\n{'='*100}")
    print(f"[{ds}]")
    print(f"{'variant':<18} {'OA':>8} {'AA':>8}  per-class")
    print('-'*100)
    print(f"150/类 (3-seed mean±std): OA={m150:.2f}±{s150:.2f}  AA={statistics.mean(aas_150):.2f}±{statistics.stdev(aas_150):.2f}")

    oas_100, aas_100 = [], []
    for seed in SEEDS:
        p = f'results_100perclass/{ds}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  100/类 seed={seed}: (missing)"); continue
        m = json.load(open(p))['target_metrics']
        oas_100.append(m['OA']); aas_100.append(m['AA'])
        print(f"  100/类 seed={seed:<4}: OA={m['OA']:>7.2f}  AA={m['AA']:>7.2f}  {[round(x,1) for x in m['class_accuracy']]}")

    if len(oas_100) >= 2:
        m100, s100 = stats(oas_100)
        print(f"100/类 (3-seed mean±std): OA={m100:.2f}±{s100:.2f}  AA={statistics.mean(aas_100):.2f}±{statistics.stdev(aas_100):.2f}")
        print(f"  → Δmean={m100-m150:+.2f}  Δstd={s100-s150:+.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 100/类 全部完成"
echo "  结束: $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_100perclass/"
echo "=================================================="
