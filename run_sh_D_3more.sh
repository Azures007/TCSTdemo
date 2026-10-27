#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_D_3seed_all/shanghai_new3

echo "=================================================="
echo "  S→H D 模式：再跑 3 个随机 seed"
echo "  开始: $(date)"
echo "=================================================="

# 随机生成 3 个 seed（排除已有的 1237/42/2024）
NEW_SEEDS=$(python - << 'PYEOF'
import secrets
used = {1237, 42, 2024}
seeds = set()
while len(seeds) < 3:
    s = secrets.randbelow(10000) + 1
    if s not in used:
        seeds.add(s)
print(' '.join(str(s) for s in sorted(seeds)))
PYEOF
)
echo "新随机 seed = $NEW_SEEDS"
echo "$NEW_SEEDS" > results_D_3seed_all/shanghai_new3/random_seeds.txt

python - << PYEOF
import yaml, copy
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
NEW_SEEDS = [int(s) for s in "$NEW_SEEDS".split()]
for seed in NEW_SEEDS:
    c = copy.deepcopy(base)
    c['training']['ckpt_mode'] = 'D'
    c['training']['ckpt_d_tolerance'] = 2.0
    c['output']['seed'] = seed
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_D_3seed_all/shanghai_new3/s{seed}'
    yaml.dump(c, open(f'configs/dall_sh_s{seed}.yaml','w'), sort_keys=False)
print(f"✅ {len(NEW_SEEDS)} 个配置已生成")
PYEOF

for seed in $NEW_SEEDS; do
    echo ""
    echo "=================================================="
    echo "  RUN: S→H D seed=$seed   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config configs/dall_sh_s${seed}.yaml 2>&1 | \
        stdbuf -oL tee "results_D_3seed_all/shanghai_new3/log_s${seed}.log" | \
        stdbuf -oL grep --line-buffered -E \
          "P1.2|Phase C|OA |AA |class[0-9]+ " | tail -10
done

echo ""
echo "=================================================="
echo "  S→H D 模式 6-seed 合并"
echo "=================================================="
python - << PYEOF | tee results_D_3seed_all/shanghai_new3/SUMMARY.txt
import json, os, statistics

# 原 3 个 seed
OLD_SEEDS = [1237, 42, 2024]
OLD_DIR = 'results_D_3seed_all/shanghai'
# 新 3 个 seed
NEW_SEEDS = [int(s) for s in "$NEW_SEEDS".split()]
NEW_DIR = 'results_D_3seed_all/shanghai_new3'

# A 模式 6-seed 也需要，但 A 模式只跑了 3 个 seed（1237/42/2024），新 3 个 seed 需要也跑 A 模式做公平对比
# 先只报 D 模式 6-seed，A 模式 3-seed 作为对照

all_d = []
print("=" * 80)
print("S→H D 模式：所有 seed")
print("=" * 80)
print(f"{'seed':<8} {'OA':>8} {'AA':>8}  per-class")
for seed in OLD_SEEDS:
    p = f'{OLD_DIR}/s{seed}/final_metrics.json'
    if not os.path.exists(p): continue
    m = json.load(open(p))['target_metrics']
    all_d.append((seed, m['OA'], m['AA']))
    print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}  {[round(x,1) for x in m['class_accuracy']]}")

for seed in NEW_SEEDS:
    p = f'{NEW_DIR}/s{seed}/final_metrics.json'
    if not os.path.exists(p): continue
    m = json.load(open(p))['target_metrics']
    all_d.append((seed, m['OA'], m['AA']))
    print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}  {[round(x,1) for x in m['class_accuracy']]}")

if len(all_d) >= 2:
    oas = [x[1] for x in all_d]; aas = [x[2] for x in all_d]
    print()
    print(f"D 模式 {len(all_d)}-seed:  OA={statistics.mean(oas):.2f}±{statistics.stdev(oas):.2f}  "
          f"AA={statistics.mean(aas):.2f}±{statistics.stdev(aas):.2f}")
    print(f"  min={min(oas):.2f}  max={max(oas):.2f}")

# A 模式 3-seed 作对照
print()
print("=" * 80)
print("S→H A 模式（3 seed，作对照）")
print("=" * 80)
A = []
for seed in OLD_SEEDS:
    p = f'results_baseline_s1/shanghai_s{seed}/final_metrics.json'
    if not os.path.exists(p): continue
    m = json.load(open(p))['target_metrics']
    A.append((seed, m['OA'], m['AA']))
    print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}")
if len(A) >= 2:
    aoas = [x[1] for x in A]
    print(f"A 模式 3-seed: OA={statistics.mean(aoas):.2f}±{statistics.stdev(aoas):.2f}")

# manifest
print()
print("=" * 80)
print("D 模式 checkpoint 选择（所有 seed）")
print("=" * 80)
for seed in OLD_SEEDS + NEW_SEEDS:
    for d in [OLD_DIR, NEW_DIR]:
        p = f'{d}/s{seed}/checkpoint_manifest.json'
        if not os.path.exists(p): continue
        d_ = json.load(open(p))
        sel = d_.get('ckpt_d_selected')
        if sel:
            print(f"  seed={seed}: cycle {sel['cycle']} ep {sel['epoch']}, "
                  f"src_val={sel['src_val']:.2f}, nll={sel['src_nll']:.4f}")
        else:
            print(f"  seed={seed}: no qualified cycle (kept warmup)")
        break
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 完成  $(date)"
echo "  新随机 seed: $NEW_SEEDS"
echo "=================================================="
