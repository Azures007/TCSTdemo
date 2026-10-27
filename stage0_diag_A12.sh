#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo

echo "======================================================="
echo "  A15: 每类真实 train/constraint/cal_pool/val 数量"
echo "======================================================="
python - << 'PYEOF'
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
from core.data import load_scene
from core.trainer_v8 import _spatial_block_split

DATASETS = {
    'Houston': ('/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13.mat',
                '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13_7gt.mat',
                7, 11),
    'HyRANK':  ('/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni.mat',
                '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni_12gt.mat',
                12, 13),
    'S→H':     ('/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
                '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat',
                3, 11),
}

for name, (img, gt, nc, ps) in DATASETS.items():
    cube, gt_arr = load_scene(img, gt, None, nc, verbose=False)
    masks, stats, mixed = _spatial_block_split(
        gt_arr, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
        val_frac_from_train=0.10, seed=1237+11, grid_size=10,
        per_class_train_cap=150, per_class_val_target=150, patch_size=ps)
    print(f"\n{'='*80}\n[{name}] mixed_classes={mixed}\n{'='*80}")
    print(f"{'cls':<4} {'train':>8} {'constraint':>12} {'cal_pool':>10} {'val':>6}")
    for c in range(1, nc+1):
        vals = []
        for key in ['train', 'constraint', 'cal_pool', 'val']:
            m = masks[key] & (gt_arr == c)
            vals.append(int(m.sum()))
        print(f"  {c-1:<3} {vals[0]:>8} {vals[1]:>12} {vals[2]:>10} {vals[3]:>6}")
    print(f"  TOTAL: train={stats['train']}  constraint={stats['constraint']}  "
          f"cal_pool={stats['cal_pool']}  val={stats['val']}")
PYEOF

echo ""
echo "======================================================="
echo "  A12 修复: 日志记录每轮 phi_star 和 d_star"
echo "======================================================="
BACKUP="core_backup_A12_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ 备份 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# A12: cycle_logs 增加 phi_star 和 d_star 的统计
old_log = '''        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
new_log = '''        # A12 fix: persist phi_star / d_star stats for post-hoc audit
        _a12_phi_stats = None
        _a12_d_stats = None
        try:
            if 'phi_star' in dir() and phi_star is not None:
                _pf = phi_star.detach().cpu().numpy().flatten()
                _a12_phi_stats = {
                    "norm": float(np.linalg.norm(_pf)),
                    "abs_max": float(np.abs(_pf).max()) if _pf.size else 0.0,
                    "mean": float(_pf.mean()) if _pf.size else 0.0,
                    "std": float(_pf.std()) if _pf.size else 0.0,
                }
        except Exception:
            pass
        try:
            if 'qp' in dir() and isinstance(qp, dict) and qp.get('d') is not None:
                _d = np.asarray(qp['d']).flatten()
                _a12_d_stats = {
                    "norm": float(np.linalg.norm(_d)),
                    "abs_max": float(np.abs(_d).max()) if _d.size else 0.0,
                    "status": qp.get('status'),
                }
        except Exception:
            pass

        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
assert old_log in s, "A12: cycle_logs 未找到"
s = s.replace(old_log, new_log, 1)

# A12: training_log.jsonl 也写入
old_j = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
new_j = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
assert old_j in s, "A12: jsonl 写入未找到"
s = s.replace(old_j, new_j, 1)

p.write_text(s)
ast.parse(s)
print("✅ A12 修复: phi_star_stats 和 d_star_stats 已写入 cycle_logs 和 training_log.jsonl")
PYEOF

echo ""
echo "✅ 阶段 0 完成（A15 + A12）"
echo "   回滚 A12: rm -rf core && mv $BACKUP core"
