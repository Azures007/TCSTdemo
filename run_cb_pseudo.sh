#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_cbpseudo_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 替换伪标签生成逻辑，改为类别均衡筛选
old_pseudo = '''                _conf, _pl = _probs.max(1)
                _thr = float(cfg["training"].get("pseudo_threshold", 0.9))
                _mask = (_conf >= _thr).numpy()
                if _mask.any():
                    pseudo_state["xb"] = tgt_transport_p[_mask]
                    pseudo_state["yb"] = _pl[_mask].numpy().astype(np.int64)
                else:
                    pseudo_state["xb"] = None
                    pseudo_state["yb"] = None
                _log(f"  [Pseudo] selected {_mask.sum()} / {len(tgt_transport_p)} (thr={_thr})")'''

new_pseudo = '''                _conf, _pl = _probs.max(1)
                _thr = float(cfg["training"].get("pseudo_threshold", 0.8))
                _conf = _conf.numpy()
                _pl = _pl.numpy()
                
                # 类别均衡伪标签选择：每个类最多选 top_k 个高置信度样本
                _per_class = int(cfg["training"].get("pseudo_per_class", 150))
                _selected_idx = []
                for _c in range(num_classes):
                    _c_idx = np.where((_pl == _c) & (_conf >= _thr))[0]
                    if len(_c_idx) > 0:
                        # 按置信度降序排，取前 _per_class 个
                        _sorted_idx = _c_idx[np.argsort(-_conf[_c_idx])]
                        _selected_idx.extend(_sorted_idx[:_per_class].tolist())
                
                if len(_selected_idx) > 0:
                    _selected_idx = np.array(_selected_idx)
                    pseudo_state["xb"] = tgt_transport_p[_selected_idx]
                    pseudo_state["yb"] = _pl[_selected_idx].astype(np.int64)
                    _counts = np.bincount(pseudo_state["yb"], minlength=num_classes).tolist()
                    _log(f"  [Pseudo] CB-selected {len(_selected_idx)} samples, dist={_counts}")
                else:
                    pseudo_state["xb"] = None
                    pseudo_state["yb"] = None
                    _log(f"  [Pseudo] None selected (thr={_thr})")'''

assert old_pseudo in s, "伪标签生成逻辑未找到"
s = s.replace(old_pseudo, new_pseudo, 1)

p.write_text(s)
ast.parse(s)
print("✅ 已升级为类别均衡伪标签 (Class-Balanced Pseudo-Label)")
PYEOF

# 统一配置测试：thr=0.8, per_class=200
python - << 'PYEOF'
import yaml, copy
BASES = {
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml'
}

for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    c = copy.deepcopy(base)
    c['training']['use_zscore'] = False
    c['training']['amplitude_channel'] = True
    c['training']['ckpt_mode'] = 'D'
    c['training']['lambda_spec'] = 0.4
    c['training']['ce_class_balance_power'] = 0.2
    
    # 伪标签设定：降阈值，强均衡
    c['training']['use_pseudo'] = True
    c['training']['pseudo_threshold'] = 0.8
    c['training']['pseudo_per_class'] = 200
    c['training']['pseudo_weight'] = 0.5
    
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/cb_pseudo/{ds}'
    yaml.dump(c, open(f'configs/cbpseudo_{ds}.yaml','w'), sort_keys=False)
print("✅ 2 个类别均衡伪标签配置已生成")
PYEOF

mkdir -p results_ablation/cb_pseudo
for ds in hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (类别均衡伪标签)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/cbpseudo_${ds}.yaml" 2>&1 | \
        tee "results_ablation/cb_pseudo/log_${ds}.log" | \
        grep -E "\[Pseudo\]|OA |AA |Kappa" | tail -8
done

echo ""
echo "=================================================="
echo "  最终提分效果对比 (原基线 vs 均衡伪标签)"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('HyRANK',   'results_ablation/compromise/hyrank/final_metrics.json',   'results_ablation/cb_pseudo/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/compromise/shanghai/final_metrics.json', 'results_ablation/cb_pseudo/shanghai/final_metrics.json')
]
print(f"{'Dataset':<10} | {'原基线 OA':^10} | {'均衡伪标签 OA':^12} | {'提升':^10}")
print('-' * 55)
for name, p_base, p_pseudo in variants:
    oa_base = json.load(open(p_base))['target_metrics']['OA'] if os.path.exists(p_base) else 0
    if os.path.exists(p_pseudo):
        m = json.load(open(p_pseudo))['target_metrics']
        oa_pseudo = m['OA']
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{name:<10} | {oa_base:>10.2f} | {oa_pseudo:>12.2f} | {oa_pseudo-oa_base:+.2f} | {pc}")
    else:
        print(f"{name:<10} | {oa_base:>10.2f} | (missing)")
PYEOF

