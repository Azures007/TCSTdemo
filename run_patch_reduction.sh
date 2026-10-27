#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/patch_reduction

echo "=================================================="
echo "  Patch Size 降维扫描 (针对 HyRANK 和 S→H)"
echo "  架构: L2 + Amp + D + Spec(0.4) + CB(0.2) + 伪标签"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

# 扫描矩阵
PATCH_SETTINGS = {
    'hyrank':   [11, 9, 7],
    'shanghai': [7, 5]
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for ps in PATCH_SETTINGS[ds]:
        c = copy.deepcopy(base)
        
        # 终极统一架构
        c['training']['use_zscore'] = False
        c['training']['amplitude_channel'] = True
        c['training']['ckpt_mode'] = 'D'
        c['training']['lambda_spec'] = 0.4
        c['training']['ce_class_balance_power'] = 0.2
        
        # 伪标签
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200
        c['training']['pseudo_weight'] = 0.5
        
        # 修改 Patch
        c['training']['patch_size'] = ps
        
        c['output']['seed'] = 1237
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/patch_reduction/{ds}_p{ps}'
        yaml.dump(c, open(f'configs/pr_{ds}_p{ps}.yaml','w'), sort_keys=False)
print("✅ 5 个降维配置已生成")
PYEOF

# 先跑 S→H 的 patch 降维
for ps in 7 5; do
    echo ""
    echo "=================================================="
    echo "  RUN: S→H / patch=$ps   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/pr_shanghai_p${ps}.yaml" 2>&1 | \
        tee "results_ablation/patch_reduction/log_sh_p${ps}.log" | \
        grep -E "\[Pseudo\]|OA |AA |Kappa" | tail -5
done

# 再跑 HyRANK 的 patch 降维
for ps in 11 9 7; do
    echo ""
    echo "=================================================="
    echo "  RUN: HyRANK / patch=$ps   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/pr_hyrank_p${ps}.yaml" 2>&1 | \
        tee "results_ablation/patch_reduction/log_hy_p${ps}.log" | \
        grep -E "\[Pseudo\]|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 Patch 降维扫描汇总"
echo "=================================================="
python - << 'PYEOF'
import json, os

print("=== S→H (原基线 Patch=11, OA=73.88%) ===")
for ps in [7, 5]:
    p = f'results_ablation/patch_reduction/shanghai_p{ps}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        print(f"  Patch={ps:<2}: OA={m['OA']:.2f}  AA={m['AA']:.2f}  Kappa={m['Kappa']:.4f}  {[round(x,1) for x in m['class_accuracy']]}")

print("\n=== HyRANK (原基线 Patch=13, OA=56.41%) ===")
for ps in [11, 9, 7]:
    p = f'results_ablation/patch_reduction/hyrank_p{ps}/final_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))['target_metrics']
        print(f"  Patch={ps:<2}: OA={m['OA']:.2f}  AA={m['AA']:.2f}  Kappa={m['Kappa']:.4f}  {[round(x,1) for x in m['class_accuracy']]}")
PYEOF
