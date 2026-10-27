#!/bin/bash
set -e
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

echo "=================================================="
echo "  1. 清理残留的僵尸进程 (释放 GPU)"
echo "=================================================="
# 强制杀掉所有 python 训练进程
pkill -f "scripts/train.py" || true
sleep 2
echo "✅ GPU 已释放"

echo ""
echo "=================================================="
echo "  2. 为 SupCon 加入防 NaN 极限保护"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 极限防护：如果在 autocast 下计算 supcon 出现 NaN，直接返回 0
new_sup = '''def _supcon_loss(features, labels, temperature=0.1):
    # 跳过 autocast，强制在纯 float32 下计算，防止矩阵乘法死锁
    with torch.amp.autocast('cuda', enabled=False):
        features = features.float()
        # 加上 1e-5 防止某些样本全为0导致归一化失败
        features = torch.nn.functional.normalize(features + 1e-5, dim=1)
        labels = labels.contiguous().view(-1, 1)
        
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
        final_loss = loss.mean()
        
        # 终极防爆：如果是 NaN，丢弃对比损失
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return final_loss'''

import re
s = re.sub(r'def _supcon_loss\(.*?return final_loss|def _supcon_loss\(.*?return loss\.mean\(\)', new_sup, s, flags=re.DOTALL)
p.write_text(s)
ast.parse(p.read_text())
print("✅ SupCon 已加入防 NaN 极限保护，并跳过 AMP 计算")
PYEOF

echo ""
echo "=================================================="
echo "  3. 重新跑 Houston (带防爆 SupCon)"
echo "=================================================="
# 我们把权重降到 0.05，避免梯度冲突
python - << 'PYEOF'
import yaml
p = 'configs/supcon_final_houston.yaml'
c = yaml.safe_load(open(p))
c['training']['lambda_supcon'] = 0.05 
yaml.dump(c, open(p, 'w'), sort_keys=False)
print("✅ 权重降为 0.05")
PYEOF

python -u scripts/train.py --config "configs/supcon_final_houston.yaml" 2>&1 | tee results_ablation/supcon_final/houston_fixed.log | tail -n 20
