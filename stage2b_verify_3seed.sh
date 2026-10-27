#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ckpt_3seed_verify

echo "=================================================="
echo "  HyRANK C 模式 + Houston B 模式 3-seed 验证"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
TASKS = [
    ('hyrank',  'configs/hy_Y2_warm40.yaml',  'C', 42),
    ('hyrank',  'configs/hy_Y2_warm40.yaml',  'C', 2024),
    ('houston', 'configs/hou_H6_ls05.yaml',   'B', 42),
    ('houston', 'configs/hou_H6_ls05.yaml',   'B', 2024),
]
for ds, base_path, mode, seed in TASKS:
    base = yaml.safe_load(open(base_path))
    c = copy.deepcopy(base)
    c['training']['ckpt_mode'] = mode
    c['output']['seed'] = seed
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ckpt_3seed_verify/{ds}_{mode}_s{seed}'
    yaml.dump(c, open(f'configs/ckpt2_{ds}_{mode}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 4 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_ckpt_3seed_verify/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E "OA |AA |class[0-9]+ " | tail -14
    echo "  DONE: $tag"
}

for ds in hyrank houston; do
    if [ "$ds" = "hyrank" ]; then MODE=C; else MODE=B; fi
    for seed in 42 2024; do
        run_one "configs/ckpt2_${ds}_${MODE}_s${seed}.yaml" "${ds}_${MODE}_s${seed}"
    done
done

echo ""
echo "=================================================="
echo "  3-seed 验证汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_ckpt_3seed_verify/SUMMARY.txt
import json, os, statistics

# 已有的 seed=1237 结果（从 stage2 直接读）
SEED1237 = {
    'hyrank_C':  {'OA': 59.89, 'AA': 56.40, 'path': 'results_ckpt_seed1237/hyrank_C/final_metrics.json'},
    'houston_B': {'OA': 77.54, 'AA': 75.35, 'path': 'results_ckpt_seed1237/houston_B/final_metrics.json'},
}
# 旧基线（batch1+A1）
BASELINE = {
    'hyrank':  {'OA': 48.95, 'AA': 50.89, 'std': 10.95},
    'houston': {'OA': 76.12, 'AA': 74.66, 'std': 1.12},
}

for ds, mode, key in [('hyrank', 'C', 'hyrank_C'), ('houston', 'B', 'houston_B')]:
    print(f"\n{'='*90}")
    print(f"[{ds}] {mode} 模式 3-seed 验证")
    print(f"{'='*90}")
    oas, aas = [], []
    # seed=1237
    s1237 = SEED1237[key]
    oas.append(s1237['OA']); aas.append(s1237['AA'])
    print(f"  seed=1237: OA={s1237['OA']:.2f}  AA={s1237['AA']:.2f}")
    # seed=42, 2024
    for seed in [42, 2024]:
        p = f'results_ckpt_3seed_verify/{ds}_{mode}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  seed={seed}: (missing)"); continue
        m = json.load(open(p))['target_metrics']
        oas.append(m['OA']); aas.append(m['AA'])
        print(f"  seed={seed}: OA={m['OA']:.2f}  AA={m['AA']:.2f}  per-class={[round(x,1) for x in m['class_accuracy']]}")

    if len(oas) >= 2:
        mean_oa = statistics.mean(oas); std_oa = statistics.stdev(oas)
        mean_aa = statistics.mean(aas); std_aa = statistics.stdev(aas)
        base = BASELINE[ds]
        print(f"\n  {mode} 模式 3-seed:  OA={mean_oa:.2f}±{std_oa:.2f}  AA={mean_aa:.2f}±{std_aa:.2f}")
        print(f"  旧基线 3-seed:    OA={base['OA']:.2f}±{base['std']:.2f}  AA={base['AA']:.2f}")
        print(f"  Δmean={mean_oa-base['OA']:+.2f}  Δstd={std_oa-base['std']:+.2f}")

        # 判断
        if ds == 'hyrank' and mean_oa >= base['OA'] + 3 and std_oa <= base['std'] - 2:
            print(f"  → ✅ 锁定 {mode} 模式：HyRANK mean +{mean_oa-base['OA']:.2f}，std -{base['std']-std_oa:.2f}")
        elif ds == 'houston' and mean_oa >= base['OA'] + 1.0 and mean_aa >= base['AA'] - 0.5:
            print(f"  → ✅ 锁定 {mode} 模式：Houston mean +{mean_oa-base['OA']:.2f}，AA {mean_aa:.2f}")
        else:
            print(f"  → ⚠️ 不锁定：提升不显著或方差仍大")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 3-seed 验证完成  $(date)"
echo "=================================================="
