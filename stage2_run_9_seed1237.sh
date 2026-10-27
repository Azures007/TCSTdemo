#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ckpt_seed1237

echo "=================================================="
echo "  3 模式 × 3 数据集 × seed=1237 (共 9 次)"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEED = 1237
MODES = ['A', 'B', 'C']
n = 0
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    for mode in MODES:
        c = copy.deepcopy(base)
        c['training']['ckpt_mode'] = mode
        c['output']['seed'] = SEED
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ckpt_seed1237/{ds}_{mode}'
        yaml.dump(c, open(f'configs/ckpt_{ds}_{mode}_s{SEED}.yaml','w'), sort_keys=False)
        n += 1
print(f"✅ {n} 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_ckpt_seed1237/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[ckpt\]|\[qp\]|\[risk\].*(ACCEPT|REJECT)|warm [0-9]+/|restored|OA |AA |class[0-9]+ " | tail -18
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for mode in A B C; do
        run_one "configs/ckpt_${ds}_${mode}_s1237.yaml" "${ds}_${mode}"
    done
done

echo ""
echo "=================================================="
echo "  9 次消融汇总 (seed=1237)"
echo "=================================================="
python - << 'PYEOF' | tee results_ckpt_seed1237/SUMMARY.txt
import json, os

# 当前基线（batch1 + A1 修复后，seed=1237）
BASE = {
    'houston':  {'OA': 75.99, 'AA': 76.67},
    'hyrank':   {'OA': 51.98, 'AA': 53.62},
    'shanghai': {'OA': 73.69, 'AA': 68.04},
}
MODES = ['A', 'B', 'C']
DATASETS = ['houston', 'hyrank', 'shanghai']

for ds in DATASETS:
    oa0 = BASE[ds]['OA']; aa0 = BASE[ds]['AA']
    print(f"\n{'='*110}")
    print(f"[{ds}]  seed=1237 基线 (batch1+A1): OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'mode':<6} {'OA':>8} {'ΔOA':>8} {'AA':>8} {'ΔAA':>8}  per-class")
    print('-'*110)
    for mode in MODES:
        p = f'results_ckpt_seed1237/{ds}_{mode}/final_metrics.json'
        if not os.path.exists(p):
            print(f"{mode:<6} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        pc = [round(x,1) for x in m['class_accuracy']]
        print(f"{mode:<6} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}  {pc}")

print(f"\n{'='*110}")
print("说明：")
print("  A = force_final_epoch=True（cycles 最后状态，T-CST 直接生效）")
print("  B = source val + target marginal entropy tie-breaker")
print("  C = class-balanced negative NLL（每类均等权重）")
print(f"{'='*110}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 9 次消融完成  $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_ckpt_seed1237/"
echo "=================================================="
