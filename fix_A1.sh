#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_A1_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# A1 fix: 在混合模式下，对 val 坐标加隔离缓冲区
# 原来的逻辑：coords 打乱后按位置切分，val 和 train 可能中心相邻 < patch_size
# 新逻辑：如果混合模式，val 坐标必须距离所有 train 坐标 >= patch_size

old_mixed = '''            if remaining >= (target_train + target_val):
                n_tr = target_train
                n_val = target_val
            else:
                n_tr = min(target_train, max(1, int(remaining * 0.5)))
                n_val = max(1, remaining - n_tr)
            cn_coords = coords[:n_cn]
            ca_coords = coords[n_cn:n_cn + n_ca_pool]
            tr_coords = coords[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            va_coords = coords[n_cn + n_ca_pool + n_tr:n_cn + n_ca_pool + n_tr + n_val]'''

new_mixed = '''            if remaining >= (target_train + target_val):
                n_tr = target_train
                n_val = target_val
            else:
                n_tr = min(target_train, max(1, int(remaining * 0.5)))
                n_val = max(1, remaining - n_tr)
            cn_coords = coords[:n_cn]
            ca_coords = coords[n_cn:n_cn + n_ca_pool]
            tr_coords = coords[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            # A1 fix: exclude val coords within patch_size of any train coord
            _ps = int(cfg["training"]["patch_size"])
            _cand = coords[n_cn + n_ca_pool + n_tr:n_cn + n_ca_pool + n_tr + n_val + 100]
            if len(tr_coords) > 0 and len(_cand) > 0:
                _tr_set = tr_coords
                _keep = []
                for _v in _cand:
                    _d = np.abs(_tr_set - _v).max(axis=1)  # Chebyshev distance
                    if (_d >= _ps).all():
                        _keep.append(_v)
                    if len(_keep) >= n_val:
                        break
                va_coords = np.array(_keep, dtype=np.int64) if _keep else _cand[:n_val]
            else:
                va_coords = _cand[:n_val]'''

assert old_mixed in s, "A1: 混合模式代码未找到"
s = s.replace(old_mixed, new_mixed, 1)
p.write_text(s)
ast.parse(s)
print("✅ A1: 混合模式下 val 坐标加隔离缓冲区（Chebyshev 距离 >= patch_size）")
PYEOF

# 验证
python - << 'PYEOF'
from pathlib import Path
tv = Path('core/trainer_v8.py').read_text()
print(f"  A1 Chebyshev 距离: {'_d = np.abs(_tr_set - _v).max' in tv}")
print(f"  A1 缓冲区检查    : {'(_d >= _ps).all()' in tv}")
PYEOF

echo ""
echo "✅ A1 修复完成（这是最后一个大 bug）"
echo "   回滚: rm -rf core && mv $BACKUP core"
