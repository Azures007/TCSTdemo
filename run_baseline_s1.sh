#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_baseline_s1

echo "=================================================="
echo "  S1 后完整 3×3 基线（warmup T_max=warmup_epochs）"
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
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_baseline_s1/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/s1_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_baseline_s1/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[scheduler\]|warm [0-9]+/|restored|OA |AA |class[0-9]+ " | tail -15
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/s1_${ds}_s${seed}.yaml" "${ds}_s${seed}"
    done
done

echo ""
echo "=================================================="
echo "  S1 后 vs v8.0.8"
echo "=================================================="
python - << 'PYEOF' | tee results_baseline_s1/SUMMARY.txt
import json, os, statistics
OLD = {
    'houston':  (76.12, 74.66),
    'hyrank':   (49.31, 48.30),
    'shanghai': (63.93, 65.14),
}
SEEDS = [1237, 42, 2024]
for ds in ['houston', 'hyrank', 'shanghai']:
    oas, aas = [], []
    print(f"\n{'='*100}")
    print(f"[{ds}]  v8.0.8: OA={OLD[ds][0]:.2f}  AA={OLD[ds][1]:.2f}")
    print(f"{'seed':<8} {'OA':>8} {'AA':>8}  per-class")
    print('-'*100)
    for seed in SEEDS:
        p = f'results_baseline_s1/{ds}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  {seed}: missing"); continue
        m = json.load(open(p))['target_metrics']
        oas.append(m['OA']); aas.append(m['AA'])
        print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}  {[round(x,1) for x in m['class_accuracy']]}")
    if len(oas) >= 2:
        moa, soa = statistics.mean(oas), statistics.stdev(oas)
        maa, saa = statistics.mean(aas), statistics.stdev(aas)
        print(f"{'mean±std':<8} {moa:>5.2f}±{soa:>4.2f} {maa:>5.2f}±{saa:>4.2f}")
        print(f"  Δ vs v8.0.8: OA {moa-OLD[ds][0]:+.2f}  AA {maa-OLD[ds][1]:+.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ S1 基线完成  $(date)"
echo "=================================================="
