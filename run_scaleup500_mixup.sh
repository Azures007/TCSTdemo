#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/scaleup500

echo "=================================================="
echo "  冲击上限: 500-shot + Mixup"
echo "  架构: L2 + Amp + D + Spec + 伪标签 + Mixup"
echo "  开始: $(date)"
echo "=================================================="

# 0) 检查并补丁 Mixup 代码
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

if 'use_mixup' not in s:
    print("⚠️ Mixup 代码丢失，正在重新打补丁...")
    old_clean = '''            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)'''
    new_clean = '''            # Mixup 增强
            _use_mixup = cfg["training"].get("use_mixup", False)
            if _use_mixup:
                _alpha = float(cfg["training"].get("mixup_alpha", 0.2))
                _lam = np.random.beta(_alpha, _alpha)
                _idx = torch.randperm(xb_raw.size(0), device=xb_raw.device)
                xb_raw = _lam * xb_raw + (1 - _lam) * xb_raw[_idx]
                yb_a, yb_b = yb, yb[_idx]
                
                out_clean = model(xb_raw)
                loss_clean = _lam * _ce(out_clean.logits, yb_a) + (1 - _lam) * _ce(out_clean.logits, yb_b)
                if _lambda_spec > 0:
                    loss_clean += _lambda_spec * (_lam * _ce(out_clean.spectral_logits, yb_a) + (1 - _lam) * _ce(out_clean.spectral_logits, yb_b))
            else:
                out_clean = model(xb_raw)
                loss_clean = _ce(out_clean.logits, yb)
                if _lambda_spec > 0:
                    loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)'''
    if old_clean in s:
        s = s.replace(old_clean, new_clean, 1)
        p.write_text(s)
        ast.parse(p.read_text())
        print("✅ Mixup 补丁已重新打入")
    else:
        print("❌ Mixup 补丁位置未找到，请手动检查 _train_step 里的 loss_clean")
else:
    print("✅ Mixup 代码已存在")
PYEOF

# 1) 生成配置 (基于各自最优 patch 和参数)
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    c = copy.deepcopy(base)
    
    # 统一 500-shot + Mixup
    c['training']['per_class_train_cap'] = 500
    c['training']['use_mixup'] = True
    c['training']['mixup_alpha'] = 0.2
    
    # 统一架构
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['ce_class_balance_power'] = 0.2
    
    # 特定数据集的最优设定
    if ds == 'houston':
        c['training']['lambda_spec'] = 0.5
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = False # 之前 Houston 伪标签没带来提升，先不开
    elif ds == 'hyrank':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 11
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200
    elif ds == 'shanghai':
        c['training']['lambda_spec'] = 0.4
        c['training']['patch_size'] = 5
        c['training']['use_pseudo'] = True
        c['training']['pseudo_threshold'] = 0.8
        c['training']['pseudo_per_class'] = 200

    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/scaleup500/{ds}'
    yaml.dump(c, open(f'configs/su500_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个 500-shot + Mixup 配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (500-shot + Mixup)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/su500_${ds}.yaml" 2>&1 | \
        tee "results_ablation/scaleup500/log_${ds}.log" | \
        grep -E "per-class train counts|\[Pseudo\]|Phase C|OA |AA |Kappa" | tail -6
done

echo ""
echo "=================================================="
echo "  🚀 500-shot + Mixup 提分效果对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston',  'results_ablation/final_boost/houston/final_metrics.json', 'results_ablation/scaleup500/houston/final_metrics.json'),
    ('HyRANK',   'results_ablation/patch_reduction/hyrank_p11/final_metrics.json', 'results_ablation/scaleup500/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/patch_reduction/shanghai_p5/final_metrics.json', 'results_ablation/scaleup500/shanghai/final_metrics.json')
]
print(f"{'Dataset':<10} | {'原最优 OA':^12} | {'500-shot OA':^12} | {'提升':^10} | Per-Class")
print('-' * 90)
for name, p_base, p_500 in variants:
    oa_base = json.load(open(p_base))['target_metrics']['OA'] if os.path.exists(p_base) else 0
    if os.path.exists(p_500):
        m = json.load(open(p_500))['target_metrics']
        oa_500 = m['OA']
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{name:<10} | {oa_base:>12.2f} | {oa_500:>12.2f} | {oa_500-oa_base:+.2f} | {pc}")
    else:
        print(f"{name:<10} | {oa_base:>12.2f} | (missing)")
PYEOF

