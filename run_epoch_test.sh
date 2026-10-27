#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/epoch_test

echo "=================================================="
echo "  训练轮数翻倍测试 (S→H)"
echo "  当前: warmup 20, cycles 2"
echo "  测试: warmup 40, cycles 4"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
c = copy.deepcopy(base)

# 终极架构
c['training']['use_zscore'] = False
c['training']['amplitude_channel'] = True
c['training']['ckpt_mode'] = 'D'
c['training']['lambda_spec'] = 0.4
c['training']['ce_class_balance_power'] = 0.2
c['training']['use_pseudo'] = True
c['training']['pseudo_threshold'] = 0.8
c['training']['pseudo_per_class'] = 200
c['training']['pseudo_weight'] = 0.5
c['training']['patch_size'] = 5 # 用刚才找到的最优 Patch

# 翻倍轮数
c['training']['epochs_warmup'] = 40
c['training']['num_cycles'] = 4
c['training']['epochs_per_cycle'] = 5

c['output']['seed'] = 1237
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/epoch_test/sh_2x'
yaml.dump(c, open('configs/epoch_2x.yaml','w'), sort_keys=False)
print("✅ 翻倍配置已生成")
PYEOF

echo ""
echo "  RUN: S→H (warmup=40, cycles=4)   $(date +%H:%M:%S)"
python -u scripts/train.py --config "configs/epoch_2x.yaml" 2>&1 | \
    tee "results_ablation/epoch_test/log.txt" | \
    grep -E "\[Pseudo\]|Phase C|OA |AA |Kappa" | tail -8

echo ""
echo "=================================================="
echo "  轮数对比 (原基线 vs 翻倍)"
echo "=================================================="
python - << 'PYEOF'
import json, os
base_p = 'results_ablation/patch_reduction/shanghai_p5/final_metrics.json'
new_p = 'results_ablation/epoch_test/sh_2x/final_metrics.json'

oa_base = json.load(open(base_p))['target_metrics']['OA'] if os.path.exists(base_p) else 0
print(f"  原基线 (Patch=5, 20+10 epochs): OA = {oa_base:.2f}%")
if os.path.exists(new_p):
    m = json.load(open(new_p))['target_metrics']
    print(f"  翻倍版 (Patch=5, 40+20 epochs): OA = {m['OA']:.2f}%  (提升: {m['OA']-oa_base:+.2f}%)")
    print(f"  Per-Class: {[round(x, 1) for x in m['class_accuracy']]}")
PYEOF

