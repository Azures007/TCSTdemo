#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_baseline_v3c

echo "=================================================="
echo "  v3c 基线 (B1-B8 全修 + split v3c)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['training']['ckpt_mode'] = 'A'
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_baseline_v3c/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/v3c_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_baseline_v3c/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[split-v3\]|warm [0-9]+/|restored|ACCEPT|OA |AA |class[0-9]+ " | tail -15
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/v3c_${ds}_s${seed}.yaml" "${ds}_s${seed}"
    done
done

echo ""
echo "=================================================="
echo "  v3c 基线 vs v8.0.7 基线"
echo "=================================================="
python - << 'PYEOF' | tee results_baseline_v3c/SUMMARY.txt
import json, os, statistics
OLD = {
    'houston':  (76.12, 74.66),
    'hyrank':   (48.95, 50.89),
    'shanghai': (59.19, 60.05),
}
SEEDS = [1237, 42, 2024]
for ds in ['houston', 'hyrank', 'shanghai']:
    oas, aas = [], []
    print(f"\n{'='*100}")
    print(f"[{ds}]  旧基线: OA={OLD[ds][0]:.2f}  AA={OLD[ds][1]:.2f}")
    print(f"{'seed':<8} {'OA':>8} {'AA':>8}  per-class")
    print('-'*100)
    for seed in SEEDS:
        p = f'results_baseline_v3c/{ds}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  {seed}: (missing)"); continue
        m = json.load(open(p))['target_metrics']
        oas.append(m['OA']); aas.append(m['AA'])
        print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}  {[round(x,1) for x in m['class_accuracy']]}")
    if len(oas) >= 2:
        mean_oa, std_oa = statistics.mean(oas), statistics.stdev(oas)
        mean_aa, std_aa = statistics.mean(aas), statistics.stdev(aas)
        print(f"{'mean±std':<8} {mean_oa:>5.2f}±{std_oa:>4.2f} {mean_aa:>5.2f}±{std_aa:>4.2f}")
        print(f"  Δ vs 旧基线: OA {mean_oa-OLD[ds][0]:+.2f}  AA {mean_aa-OLD[ds][1]:+.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ v3c 基线重建完成  $(date)"
echo "=================================================="
