#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_houston_D_3seed

echo "=================================================="
echo "  Houston D 模式 3-seed 确认"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
SEEDS = [1237, 42, 2024]
for seed in SEEDS:
    c = copy.deepcopy(base)
    c['training']['ckpt_mode'] = 'D'
    c['training']['ckpt_d_tolerance'] = 2.0
    c['output']['seed'] = seed
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_houston_D_3seed/s{seed}'
    yaml.dump(c, open(f'configs/houD_s{seed}.yaml','w'), sort_keys=False)
print("✅ 3 个配置已生成")
PYEOF

for seed in 1237 42 2024; do
    echo ""
    echo "=================================================="
    echo "  RUN: houston D seed=$seed   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config configs/houD_s${seed}.yaml 2>&1 | \
        stdbuf -oL tee "results_houston_D_3seed/log_s${seed}.log" | \
        stdbuf -oL grep --line-buffered -E \
          "P1.2|Phase C|OA |AA |class[0-9]+ " | tail -12
done

echo ""
echo "=================================================="
echo "  Houston A vs D 3-seed 对比"
echo "=================================================="
python - << 'PYEOF' | tee results_houston_D_3seed/SUMMARY.txt
import json, os, statistics
SEEDS = [1237, 42, 2024]

# A 模式（从 baseline_s1 读）
A_runs = []
print("=" * 80)
print("Houston A 模式（v8.0.9 baseline_s1）")
print("=" * 80)
print(f"{'seed':<8} {'OA':>8} {'AA':>8}")
for seed in SEEDS:
    p = f'results_baseline_s1/houston_s{seed}/final_metrics.json'
    if not os.path.exists(p):
        print(f"  {seed}: missing"); continue
    m = json.load(open(p))['target_metrics']
    A_runs.append((seed, m['OA'], m['AA']))
    print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}")
if len(A_runs) >= 2:
    A_oa = [x[1] for x in A_runs]
    A_aa = [x[2] for x in A_runs]
    print(f"  mean±std  OA={statistics.mean(A_oa):.2f}±{statistics.stdev(A_oa):.2f}  "
          f"AA={statistics.mean(A_aa):.2f}±{statistics.stdev(A_aa):.2f}")

# D 模式（从新结果读）
D_runs = []
print()
print("=" * 80)
print("Houston D 模式（ckpt_mode=D, tolerance=2.0）")
print("=" * 80)
print(f"{'seed':<8} {'OA':>8} {'AA':>8}  per-class")
for seed in SEEDS:
    p = f'results_houston_D_3seed/s{seed}/final_metrics.json'
    if not os.path.exists(p):
        print(f"  {seed}: missing"); continue
    m = json.load(open(p))['target_metrics']
    D_runs.append((seed, m['OA'], m['AA']))
    print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}  {[round(x,1) for x in m['class_accuracy']]}")
if len(D_runs) >= 2:
    D_oa = [x[1] for x in D_runs]
    D_aa = [x[2] for x in D_runs]
    print(f"  mean±std  OA={statistics.mean(D_oa):.2f}±{statistics.stdev(D_oa):.2f}  "
          f"AA={statistics.mean(D_aa):.2f}±{statistics.stdev(D_aa):.2f}")

# 差值
if len(A_runs) >= 2 and len(D_runs) >= 2:
    print()
    print("=" * 80)
    print("D vs A")
    print("=" * 80)
    print(f"  ΔOA_mean = {statistics.mean(D_oa) - statistics.mean(A_oa):+.2f}")
    print(f"  ΔAA_mean = {statistics.mean(D_aa) - statistics.mean(A_aa):+.2f}")

# manifest 检查：D 模式每个 seed 选了什么
print()
print("=" * 80)
print("D 模式 manifest 摘要")
print("=" * 80)
for seed in SEEDS:
    p = f'results_houston_D_3seed/s{seed}/checkpoint_manifest.json'
    if not os.path.exists(p): continue
    d = json.load(open(p))
    sel = d.get('ckpt_d_selected')
    if sel:
        print(f"  seed={seed}: selected cycle {sel['cycle']} ep {sel['epoch']}, "
              f"src_val={sel['src_val']:.2f}, nll={sel['src_nll']:.4f}")
    else:
        print(f"  seed={seed}: no qualified cycle, kept warmup")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ Houston D 3-seed 完成  $(date)"
echo "=================================================="
