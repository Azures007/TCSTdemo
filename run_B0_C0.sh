#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_B0_C0/{B0,C0}

echo "=================================================="
echo "  B0 (Source-only) + C0 (Source-only + Aug)"
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
    for variant in ['B0', 'C0']:
        for seed in SEEDS:
            c = copy.deepcopy(base)
            c['training']['ckpt_mode'] = 'A'
            c['training']['num_cycles'] = 0
            c['training']['run_mode'] = 'warmup_only'
            c['training']['use_warmup_augmentation'] = (variant == 'C0')
            c['training']['lambda_aug'] = 0.5 if variant == 'C0' else 0.0
            c['output']['seed'] = seed
            c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_B0_C0/{variant}/{ds}_s{seed}'
            yaml.dump(c, open(f'configs/{variant}_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 18 个配置已生成（B0: 9, C0: 9）")
PYEOF

run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E "warm [0-9]+/|aug=|restored|OA |AA " | tail -5
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/B0_${ds}_s${seed}.yaml" "${ds}_s${seed}" "results_B0_C0/B0"
    done
done

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/C0_${ds}_s${seed}.yaml" "${ds}_s${seed}" "results_B0_C0/C0"
    done
done

echo ""
echo "=================================================="
echo "  B0/C0 vs 当前 v8.1.0（T-CST 完整）"
echo "=================================================="
python - << 'PYEOF' | tee results_B0_C0/SUMMARY.txt
import json, os, statistics
SEEDS = [1237, 42, 2024]

# 当前 v8.1.0 的完整结果（T-CST with cycles）
FULL = {
    'houston':  [(75.99, 76.67), (75.07, 69.68), (77.30, 77.62)],
    'hyrank':   [(51.00, 50.69), (54.72, 48.36), (57.13, 44.02)],
    'shanghai': [(74.75, 72.73), (54.72, 60.51), (62.32, 62.17)],
}

for ds in ['houston', 'hyrank', 'shanghai']:
    print(f"\n{'='*90}")
    print(f"[{ds.upper()}]")
    print(f"{'='*90}")
    print(f"{'variant':<20} {'OA (mean±std)':>20} {'AA (mean±std)':>20}")
    print('-'*90)

    for variant, label in [('B0', 'B0 Source-only'), ('C0', 'C0 Source+Aug'), ('FULL', 'v8.1.0 T-CST cycles')]:
        oas, aas = [], []
        if variant == 'FULL':
            for oa, aa in FULL[ds]:
                oas.append(oa); aas.append(aa)
        else:
            for seed in SEEDS:
                p = f'results_B0_C0/{variant}/{ds}_s{seed}/final_metrics.json'
                if not os.path.exists(p): continue
                m = json.load(open(p))['target_metrics']
                oas.append(m['OA']); aas.append(m['AA'])
        if len(oas) >= 2:
            print(f"  {label:<18} {statistics.mean(oas):>8.2f}±{statistics.stdev(oas):<8.2f} "
                  f"{statistics.mean(aas):>8.2f}±{statistics.stdev(aas):<8.2f}")

    # 计算 Δ
    b0_oa, c0_oa, full_oa = [], [], []
    for seed in SEEDS:
        p_b0 = f'results_B0_C0/B0/{ds}_s{seed}/final_metrics.json'
        p_c0 = f'results_B0_C0/C0/{ds}_s{seed}/final_metrics.json'
        if os.path.exists(p_b0): b0_oa.append(json.load(open(p_b0))['target_metrics']['OA'])
        if os.path.exists(p_c0): c0_oa.append(json.load(open(p_c0))['target_metrics']['OA'])
    full_oa = [x[0] for x in FULL[ds]]

    print()
    if b0_oa and full_oa:
        print(f"  T-CST cycles 贡献 (v8.1.0 - B0):  {statistics.mean(full_oa)-statistics.mean(b0_oa):+.2f} OA")
    if b0_oa and c0_oa:
        print(f"  增强对 Source-only 的影响 (C0 - B0): {statistics.mean(c0_oa)-statistics.mean(b0_oa):+.2f} OA")

    print(f"\n  逐 seed:")
    for i, seed in enumerate(SEEDS):
        row = f"    seed={seed}: "
        p_b0 = f'results_B0_C0/B0/{ds}_s{seed}/final_metrics.json'
        p_c0 = f'results_B0_C0/C0/{ds}_s{seed}/final_metrics.json'
        if os.path.exists(p_b0):
            m = json.load(open(p_b0))['target_metrics']
            row += f"B0={m['OA']:.2f}  "
        if os.path.exists(p_c0):
            m = json.load(open(p_c0))['target_metrics']
            row += f"C0={m['OA']:.2f}  "
        row += f"FULL={FULL[ds][i][0]:.2f}"
        print(row)
PYEOF

echo ""
echo "=================================================="
echo "  ✅ B0/C0 完成  $(date)"
echo "=================================================="
