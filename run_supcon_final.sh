#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_supcon_final_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"
mkdir -p results_ablation/supcon_final

echo "=================================================="
echo "  1. 植入 SupCon Loss"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 1) 定义 _supcon_loss
supcon_func = '''
def _supcon_loss(features, labels, temperature=0.1):
    """Supervised Contrastive Loss (Khosla et al. 2020)"""
    device = features.device
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    loss = - (temperature * mean_log_prob_pos)
    return loss.mean()
'''
if '_supcon_loss' not in s:
    marker = '    def _focal_loss('
    s = s.replace(marker, supcon_func + '\n' + marker, 1)

# 2) 在 _train_step 中计算 SupCon
old_loss = '''            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)'''
new_loss = '''            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)
            # 注入 SupCon
            _lambda_sup = float(cfg["training"].get("lambda_supcon", 0.0))
            if _lambda_sup > 0:
                loss_clean += _lambda_sup * _supcon_loss(out_clean.features, yb)'''
assert old_loss in s, "loss_clean 位置未找到"
s = s.replace(old_loss, new_loss, 1)

p.write_text(s)
ast.parse(p.read_text())
print("✅ trainer_v8.py 已成功植入 SupCon Loss")
PYEOF

echo ""
echo "=================================================="
echo "  2. 在最优配置上开启 SupCon 训练 3 个数据集"
echo "=================================================="
python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

OPTIMAL_CFG = {
    'houston': {
        'use_zscore': False, 'amplitude_channel': True, 'ckpt_mode': 'D',
        'lambda_spec': 0.5, 'ce_class_balance_power': 0.0,
        'use_pseudo': False, 'patch_size': 11
    },
    'hyrank': {
        'use_zscore': False, 'amplitude_channel': True, 'ckpt_mode': 'D',
        'lambda_spec': 0.4, 'ce_class_balance_power': 0.2,
        'use_pseudo': True, 'pseudo_threshold': 0.7, 'pseudo_per_class': 200, 'patch_size': 11
    },
    'shanghai': {
        'use_zscore': False, 'amplitude_channel': True, 'ckpt_mode': 'D',
        'lambda_spec': 0.4, 'ce_class_balance_power': 0.2,
        'use_pseudo': True, 'pseudo_threshold': 0.8, 'pseudo_per_class': 200, 'patch_size': 5
    }
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    c = copy.deepcopy(base)
    
    for k, v in OPTIMAL_CFG[ds].items():
        c['training'][k] = v
        
    # 开启 SupCon，权重设置为 0.1
    c['training']['lambda_supcon'] = 0.1
    
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/supcon_final/{ds}'
    yaml.dump(c, open(f'configs/supcon_final_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个 SupCon 配置已生成")
PYEOF

for ds in houston hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (终极基线 + SupCon)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/supcon_final_${ds}.yaml" 2>&1 | \
        tee "results_ablation/supcon_final/log_${ds}.log" | \
        grep -E "CB-selected|P1.2|OA |AA |Kappa" | tail -5
done

echo ""
echo "=================================================="
echo "  🚀 SupCon 提分效果对比"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('Houston',  'results_final_best/surgical/houston/final_metrics.json', 'results_ablation/supcon_final/houston/final_metrics.json'),
    ('HyRANK',   'results_final_best/rescue_v3/hyrank/final_metrics.json', 'results_ablation/supcon_final/hyrank/final_metrics.json'),
    ('S→H',      'results_final_best/surgical/shanghai/final_metrics.json','results_ablation/supcon_final/shanghai/final_metrics.json')
]
print(f"{'Dataset':<10} | {'纯净基线 OA':^12} | {'+SupCon OA':^12} | {'提升':^10} | Per-Class")
print('-' * 95)
for name, p_base, p_sup in variants:
    oa_base = json.load(open(p_base))['target_metrics']['OA'] if os.path.exists(p_base) else 0
    if os.path.exists(p_sup):
        m = json.load(open(p_sup))['target_metrics']
        oa_sup = m['OA']
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{name:<10} | {oa_base:>12.2f} | {oa_sup:>12.2f} | {oa_sup-oa_base:+.2f} | {pc}")
    else:
        print(f"{name:<10} | {oa_base:>12.2f} | (missing)")
PYEOF

