#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_d2_norm

echo "=================================================="
echo "  D2: BatchNorm vs LayerNorm（seed=1237）"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for norm in ['batch', 'layer']:
        c = copy.deepcopy(base)
        c['training']['projection_norm'] = norm
        c['training']['ckpt_mode'] = 'A'
        c['output']['seed'] = 1237
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_d2_norm/{ds}_{norm}'
        yaml.dump(c, open(f'configs/d2_{ds}_{norm}.yaml','w'), sort_keys=False)
print("✅ 6 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_d2_norm/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[backbone\]|warm [0-9]+/|restored|OA |AA |class[0-9]+ " | tail -14
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for norm in batch layer; do
        run_one "configs/d2_${ds}_${norm}.yaml" "${ds}_${norm}"
    done
done

echo ""
echo "=================================================="
echo "  D2 汇总 vs 回滚后基线（seed=1237）"
echo "=================================================="
python - << 'PYEOF' | tee results_d2_norm/SUMMARY.txt
import json, os
# 回滚后的 seed=1237 基线
BASE_1237 = {
    'houston':  (75.99, 76.67),
    'hyrank':   (51.30, 52.78),
    'shanghai': (74.75, 72.73),
}
for ds in ['houston', 'hyrank', 'shanghai']:
    oa0, aa0 = BASE_1237[ds]
    print(f"\n{'='*100}")
    print(f"[{ds}]  回滚后基线 seed=1237: OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'norm':<8} {'OA':>8} {'ΔOA':>8} {'AA':>8} {'ΔAA':>8}  per-class")
    print('-'*100)
    for norm in ['batch', 'layer']:
        p = f'results_d2_norm/{ds}_{norm}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  {norm:<6} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        pc = [round(x,1) for x in m['class_accuracy']]
        print(f"  {norm:<6} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}  {pc}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ D2 完成  $(date)"
echo "=================================================="
