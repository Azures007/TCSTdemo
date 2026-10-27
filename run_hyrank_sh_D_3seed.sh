#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_D_3seed_all/{hyrank,shanghai}

echo "=================================================="
echo "  HyRANK + S→H D 模式 3-seed"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['training']['ckpt_mode'] = 'D'
        c['training']['ckpt_d_tolerance'] = 2.0
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_D_3seed_all/{ds}/s{seed}'
        yaml.dump(c, open(f'configs/dall_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 6 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_D_3seed_all/${tag%_s*}/log_${tag}.log" | \
        stdbuf -oL grep --line-buffered -E \
          "P1.2|Phase C|OA |AA |class[0-9]+ " | tail -12
}

for ds in hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/dall_${ds}_s${seed}.yaml" "${ds}_s${seed}"
    done
done

echo ""
echo "=================================================="
echo "  完整对比：三数据集 A vs D"
echo "=================================================="
python - << 'PYEOF' | tee results_D_3seed_all/SUMMARY.txt
import json, os, statistics
SEEDS = [1237, 42, 2024]

DATASETS = {
    'houston':  ('results_baseline_s1', 'results_houston_D_3seed', 's'),
    'hyrank':   ('results_baseline_s1', 'results_D_3seed_all/hyrank', 's'),
    'shanghai': ('results_baseline_s1', 'results_D_3seed_all/shanghai', 's'),
}

for ds in ['houston', 'hyrank', 'shanghai']:
    a_dir, d_dir, _ = DATASETS[ds]
    A, D = [], []
    for seed in SEEDS:
        pa = f'{a_dir}/{ds}_s{seed}/final_metrics.json'
        pd = f'{d_dir}/s{seed}/final_metrics.json'
        if os.path.exists(pa):
            m = json.load(open(pa))['target_metrics']
            A.append((m['OA'], m['AA']))
        if os.path.exists(pd):
            m = json.load(open(pd))['target_metrics']
            D.append((m['OA'], m['AA'], m['class_accuracy']))

    print(f"\n{'='*80}")
    print(f"[{ds.upper()}]")
    print(f"{'='*80}")

    if len(A) >= 2:
        aoa = [x[0] for x in A]; aaa = [x[1] for x in A]
        print(f"  A 模式: OA={statistics.mean(aoa):.2f}±{statistics.stdev(aoa):.2f}  "
              f"AA={statistics.mean(aaa):.2f}±{statistics.stdev(aaa):.2f}")
    if len(D) >= 2:
        doa = [x[0] for x in D]; daa = [x[1] for x in D]
        print(f"  D 模式: OA={statistics.mean(doa):.2f}±{statistics.stdev(doa):.2f}  "
              f"AA={statistics.mean(daa):.2f}±{statistics.stdev(daa):.2f}")
        if len(A) >= 2:
            print(f"  Δ:    ΔOA={statistics.mean(doa)-statistics.mean(aoa):+.2f}  "
                  f"ΔAA={statistics.mean(daa)-statistics.mean(aaa):+.2f}")
    print()
    print(f"  逐 seed:")
    for i, seed in enumerate(SEEDS):
        a_str = f"A={A[i][0]:.2f}/{A[i][1]:.2f}" if i < len(A) else "A=missing"
        d_str = f"D={D[i][0]:.2f}/{D[i][1]:.2f}" if i < len(D) else "D=missing"
        print(f"    seed={seed}: {a_str}  {d_str}")
        if i < len(D):
            print(f"      D per-class={[round(x,1) for x in D[i][2]]}")

# manifest 摘要
print(f"\n{'='*80}")
print("D 模式 checkpoint 选择摘要")
print(f"{'='*80}")
for ds in ['houston', 'hyrank', 'shanghai']:
    _, d_dir, _ = DATASETS[ds]
    for seed in SEEDS:
        p = f'{d_dir}/s{seed}/checkpoint_manifest.json'
        if not os.path.exists(p): continue
        d = json.load(open(p))
        sel = d.get('ckpt_d_selected')
        if sel:
            print(f"  {ds} seed={seed}: cycle {sel['cycle']} ep {sel['epoch']}, "
                  f"src_val={sel['src_val']:.2f}, nll={sel['src_nll']:.4f}")
        else:
            print(f"  {ds} seed={seed}: no qualified cycle (kept warmup)")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 全部 D 模式 3-seed 完成  $(date)"
echo "=================================================="
