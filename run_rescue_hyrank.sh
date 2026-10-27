#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/rescue_hyrank

echo "=================================================="
echo "  拯救 HyRANK：统一开启类均衡 + 降低光谱权重"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

# 两组测试参数： (光谱权重, 类别均衡)
SETTINGS = [
    ('spec0.1_cb0.5', 0.1, 0.5),
    ('spec0.3_cb0.5', 0.3, 0.5)
]

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for tag, lspec, cb_pow in SETTINGS:
        c = copy.deepcopy(base)
        # 统一架构设定
        c['training']['use_zscore'] = False
        c['training']['amplitude_channel'] = True
        c['training']['ckpt_mode'] = 'D'
        
        # 救场修改
        c['training']['lambda_spec'] = lspec
        c['training']['ce_class_balance_power'] = cb_pow
        
        c['output']['seed'] = 1237
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/rescue_hyrank/{tag}/{ds}'
        yaml.dump(c, open(f'configs/rescue_{tag}_{ds}.yaml','w'), sort_keys=False)
print("✅ 6 个救场配置已生成")
PYEOF

for tag in spec0.1_cb0.5 spec0.3_cb0.5; do
    for ds in houston hyrank shanghai; do
        echo ""
        echo "  RUN: ${tag} / ${ds}   $(date +%H:%M:%S)"
        python -u scripts/train.py --config "configs/rescue_${tag}_${ds}.yaml" 2>&1 | \
            tee "results_ablation/rescue_hyrank/log_${tag}_${ds}.log" | \
            grep -E "OA |AA |Kappa" | tail -3
    done
done

echo ""
echo "=================================================="
echo "  拯救结果汇总对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = ['spec0.1_cb0.5', 'spec0.3_cb0.5']
datasets = ['houston', 'hyrank', 'shanghai']

print(f"{'Setting':<16} | {'Houston':^10} | {'HyRANK':^10} | {'S→H':^10}")
print('-' * 55)
for v in variants:
    oas = []
    for ds in datasets:
        p = f'results_ablation/rescue_hyrank/{v}/{ds}/final_metrics.json'
        if os.path.exists(p):
            oas.append(json.load(open(p))['target_metrics']['OA'])
        else:
            oas.append(0.0)
    print(f"{v:<16} | {oas[0]:>10.2f} | {oas[1]:>10.2f} | {oas[2]:>10.2f}")
PYEOF
