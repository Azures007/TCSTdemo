#!/bin/bash
set -e
cd ~/autodl-tmp/TCSTdemo

echo "=================================================="
echo "  1. 修复 SupCon 的数值溢出问题"
echo "=================================================="

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 强制使用 float32 和 L2 归一化，防止 AMP 溢出导致 NaN 卡死
new_sup = '''def _supcon_loss(features, labels, temperature=0.1):
    """Supervised Contrastive Loss (Khosla et al. 2020) - Stable Version"""
    features = features.float()
    labels = labels.contiguous().view(-1, 1)
    features = torch.nn.functional.normalize(features, dim=1)
    
    device = features.device
    batch_size = features.shape[0]
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0
    )
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    mask_sum = mask.sum(1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
    loss = - (temperature * mean_log_prob_pos)
    return loss.mean()'''

import re
s = re.sub(r'def _supcon_loss\(.*?return loss\.mean\(\)', new_sup, s, flags=re.DOTALL)
p.write_text(s)
ast.parse(p.read_text())
print("✅ SupCon 已强制使用 float32 和 L2 归一化，修复数值溢出")
PYEOF

echo ""
echo "=================================================="
echo "  2. 重新尝试运行 Houston (带 SupCon)"
echo "=================================================="
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

# 直接跑刚才生成的 Houston SupCon 配置
python -u scripts/train.py --config "configs/supcon_final_houston.yaml" 2>&1 | \
    tee "results_ablation/supcon_final/log_houston_v2.log" | \
    grep -E "Phase C|OA |AA |Kappa" | tail -5

