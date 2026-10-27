#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_aug_ablation/{houston,shanghai}

echo "=================================================="
echo "  增强消融：Houston/S→H × (关闭/开启) × 3 seed"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for aug in [False, True]:
        for seed in SEEDS:
            c = copy.deepcopy(base)
            c['training']['ckpt_mode'] = 'A'  # 统一 A（已决策）
            c['training']['use_warmup_augmentation'] = aug
            c['training']['lambda_aug'] = 0.5 if aug else 0.0
            c['output']['seed'] = seed
            tag = 'aug_on' if aug else 'aug_off'
            c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_aug_ablation/{ds}/{tag}_s{seed}'
            yaml.dump(c, open(f'configs/aug_{ds}_{tag}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 12 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|cyc[0-9]+ ep|OA |AA |class[0-9]+ " | tail -10
}

for ds in houston shanghai; do
    for aug in aug_off aug_on; do
        for seed in 1237 42 2024; do
            run_one "configs/aug_${ds}_${aug}_s${seed}.yaml" "${aug}_s${seed}" "results_aug_ablation/$ds"
        done
    done
done

echo ""
echo "=================================================="
echo "  增强消融汇总（统一 ckpt_mode=A）"
echo "=================================================="
python - << 'PYEOF' | tee results_aug_ablation/SUMMARY.txt
import json, os, statistics
SEEDS = [1237, 42, 2024]

for ds in ['houston', 'shanghai']:
    print(f"\n{'='*80}")
    print(f"[{ds}]  统一 ckpt_mode=A，对比增强开关")
    print(f"{'='*80}")
    print(f"{'config':<12} {'OA':>8} {'AA':>8}  per-class (mean across seeds)")
    print('-'*100)

    results = {}
    for aug in ['aug_off', 'aug_on']:
        oas, aas = [], []
        pcs = []
        for seed in SEEDS:
            p = f'results_aug_ablation/{ds}/{aug}_s{seed}/final_metrics.json'
            if not os.path.exists(p): continue
            m = json.load(open(p))['target_metrics']
            oas.append(m['OA']); aas.append(m['AA'])
            pcs.append(m['class_accuracy'])
        if len(oas) >= 2:
            moa, soa = statistics.mean(oas), statistics.stdev(oas)
            maa, saa = statistics.mean(aas), statistics.stdev(aas)
            # 逐类均值
            n_cls = len(pcs[0]) if pcs else 0
            avg_pc = [statistics.mean([pc[c] for pc in pcs]) for c in range(n_cls)] if pcs else []
            results[aug] = (moa, soa, maa, saa)
            print(f"  {aug:<10} {moa:>6.2f}±{soa:>4.2f} {maa:>6.2f}±{saa:>4.2f}  {[round(x,1) for x in avg_pc]}")

    if 'aug_off' in results and 'aug_on' in results:
        print()
        print(f"  Δ OA: {results['aug_on'][0] - results['aug_off'][0]:+.2f}")
        print(f"  Δ AA: {results['aug_on'][2] - results['aug_off'][2]:+.2f}")

    # 逐 seed 明细
    print(f"\n  逐 seed 明细:")
    for aug in ['aug_off', 'aug_on']:
        print(f"    [{aug}]")
        for seed in SEEDS:
            p = f'results_aug_ablation/{ds}/{aug}_s{seed}/final_metrics.json'
            if not os.path.exists(p): continue
            m = json.load(open(p))['target_metrics']
            print(f"      seed={seed}: OA={m['OA']:.2f}  AA={m['AA']:.2f}  {[round(x,1) for x in m['class_accuracy']]}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 增强消融完成  $(date)"
echo "=================================================="
