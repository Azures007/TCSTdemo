#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_ablI3_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

echo "===== 1) 诊断 I1/I2/I3 开关状态 ====="
python - << 'PYEOF'
from pathlib import Path
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'I1 use_qp': 'use_qp' in tv,
    'I2 use_risk_search': 'use_risk_search' in tv,
    'I3 use_safe_training': 'use_safe_training' in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

echo ""
echo "===== 2) 补 I3 use_safe_training 开关 ====="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

if 'use_safe_training' in s:
    print("  ⏭️ I3 已存在，跳过")
else:
    old = '''            # 三路融合：clean + lambda_aug * aug + safe_w * safe
            _terms = [loss_clean]
            _weights = [1.0]
            if loss_aug is not None:
                _terms.append(loss_aug)
                _weights.append(_lambda_aug)
            if loss_safe is not None:
                _terms.append(loss_safe)
                _weights.append(safe_w)
            _wsum = sum(_weights)
            loss = sum(w * t for w, t in zip(_weights, _terms)) / _wsum'''
    new = '''            # 消融: use_safe_training=False 时用 hard replace（loss=CE(T(x),y)）
            # use_safe_training=True（默认）用加权融合
            _use_safe_training = bool(cfg["training"].get("use_safe_training", True))
            if loss_safe is not None and not _use_safe_training:
                # hard replace: 只用 safe loss（去掉 clean 锚点）
                _terms = [loss_safe]
                _weights = [1.0]
                if loss_aug is not None:
                    _terms.append(loss_aug)
                    _weights.append(_lambda_aug)
            else:
                # 默认: clean + lambda_aug * aug + safe_w * safe
                _terms = [loss_clean]
                _weights = [1.0]
                if loss_aug is not None:
                    _terms.append(loss_aug)
                    _weights.append(_lambda_aug)
                if loss_safe is not None:
                    _terms.append(loss_safe)
                    _weights.append(safe_w)
            _wsum = sum(_weights)
            loss = sum(w * t for w, t in zip(_weights, _terms)) / _wsum'''
    assert old in s, "I3: 三路融合块未找到"
    s = s.replace(old, new, 1)
    p.write_text(s)
    ast.parse(s)
    print("  ✅ I3 use_safe_training 开关已加")

# 验证
s = p.read_text()
print(f"  I1 use_qp: {'✅' if 'use_qp' in s else '❌'}")
print(f"  I2 use_risk_search: {'✅' if 'use_risk_search' in s else '❌'}")
print(f"  I3 use_safe_training: {'✅' if 'use_safe_training' in s else '❌'}")
PYEOF

echo ""
echo "===== 3) 冒烟 A: 默认（I1/I2/I3 全开）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablI3_smoke/default'
yaml.dump(c, open('configs/ablI3_default.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_ablI3_smoke/default
python -u scripts/train.py --config configs/ablI3_default.yaml 2>&1 | grep -E "qp|risk|OA " | head -4

echo ""
echo "===== 4) 冒烟 B: I1 no_qp ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['tcst']['use_qp'] = False
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablI3_smoke/no_qp'
yaml.dump(c, open('configs/ablI3_no_qp.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_ablI3_smoke/no_qp
python -u scripts/train.py --config configs/ablI3_no_qp.yaml 2>&1 | grep -E "qp|risk|OA " | head -4

echo ""
echo "===== 5) 冒烟 C: I2 no_risk ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['tcst']['use_risk_search'] = False
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablI3_smoke/no_risk'
yaml.dump(c, open('configs/ablI3_no_risk.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_ablI3_smoke/no_risk
python -u scripts/train.py --config configs/ablI3_no_risk.yaml 2>&1 | grep -E "qp|risk|OA " | head -4

echo ""
echo "===== 6) 冒烟 D: I3 hard_replace ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['use_safe_training'] = False
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablI3_smoke/hard_replace'
yaml.dump(c, open('configs/ablI3_hard_replace.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_ablI3_smoke/hard_replace
python -u scripts/train.py --config configs/ablI3_hard_replace.yaml 2>&1 | grep -E "qp|risk|OA " | head -4

echo ""
echo "✅ I1/I2/I3 开关全部就绪"
echo "   回滚: rm -rf core && mv $BACKUP core"
