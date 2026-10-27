#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_patch_ablation

echo "=================================================="
echo "  patch_size 消融 (9/7/5) × 3 随机 seed"
echo "  开始: $(date)"
echo "=================================================="

# ============ 1) 随机生成 3 个 seed ============
echo ""
echo "===== 随机生成 3 个 seed ====="
SEEDS=$(python - << 'PYEOF'
import secrets
seeds = sorted(set(secrets.randbelow(10000) + 1 for _ in range(3)))
while len(seeds) < 3:
    seeds.append(secrets.randbelow(10000) + 1)
print(' '.join(str(s) for s in seeds[:3]))
PYEOF
)
echo "随机 seed = $SEEDS"
echo "$SEEDS" > results_patch_ablation/random_seeds.txt

# ============ 2) 生成 27 个配置 ============
echo ""
echo "===== 生成 27 个配置 ====="
python - << PYEOF
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
PATCHES = [9, 7, 5]
SEEDS = [int(s) for s in "$SEEDS".split()]
n = 0
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for ps in PATCHES:
        for seed in SEEDS:
            c = copy.deepcopy(base)
            c['training']['patch_size'] = ps
            c['training']['ckpt_mode'] = 'A'
            c['output']['seed'] = seed
            c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_patch_ablation/{ds}_p{ps}_s{seed}'
            yaml.dump(c, open(f'configs/pa_{ds}_p{ps}_s{seed}.yaml','w'), sort_keys=False)
            n += 1
print(f"✅ {n} 个配置已生成")
PYEOF

# ============ 3) 跑 27 次 ============
run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_patch_ablation/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|restored|OA |AA |class[0-9]+ " | tail -12
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for ps in 9 7 5; do
        for seed in $SEEDS; do
            run_one "configs/pa_${ds}_p${ps}_s${seed}.yaml" "${ds}_p${ps}_s${seed}"
        done
    done
done

# ============ 4) 汇总 ============
echo ""
echo "=================================================="
echo "  patch 消融汇总"
echo "=================================================="
python - << PYEOF | tee results_patch_ablation/SUMMARY.txt
import json, os, statistics
SEEDS = [int(s) for s in "$SEEDS".split()]
PATCHES = [9, 7, 5]
DATASETS = ['houston', 'hyrank', 'shanghai']
# 当前基线（patch=11/13/11，seed=1237/42/2024）
BASELINE = {
    'houston':  (76.12, 74.66),
    'hyrank':   (49.31, 48.30),
    'shanghai': (63.93, 65.14),
}
print(f"\n随机 seed: {SEEDS}")
for ds in DATASETS:
    oa0, aa0 = BASELINE[ds]
    print(f"\n{'='*95}")
    print(f"[{ds}]  baseline (patch={11 if ds!='hyrank' else 13}): OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'patch':<8} {'OA (mean±std)':>16} {'AA (mean±std)':>16} {'ΔOA':>8} {'ΔAA':>8}")
    print('-'*95)
    for ps in PATCHES:
        oas, aas = [], []
        for seed in SEEDS:
            p = f'results_patch_ablation/{ds}_p{ps}_s{seed}/final_metrics.json'
            if not os.path.exists(p): continue
            m = json.load(open(p))['target_metrics']
            oas.append(m['OA']); aas.append(m['AA'])
        if len(oas) >= 2:
            moa, soa = statistics.mean(oas), statistics.stdev(oas)
            maa, saa = statistics.mean(aas), statistics.stdev(aas)
            print(f"  p{ps:<6} {moa:>7.2f}±{soa:>4.2f}    {maa:>7.2f}±{saa:>4.2f}  "
                  f"{moa-oa0:>+8.2f} {maa-aa0:>+8.2f}")
        elif len(oas) == 1:
            print(f"  p{ps:<6} {oas[0]:>7.2f} (1 seed)  {aas[0]:>7.2f}        "
                  f"{oas[0]-oa0:>+8.2f} {aas[0]-aa0:>+8.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ patch 消融完成  $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_patch_ablation/"
echo "  随机 seed: $(cat results_patch_ablation/random_seeds.txt)"
echo "=================================================="
