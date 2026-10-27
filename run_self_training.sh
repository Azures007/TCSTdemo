#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_pseudo_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 1) 在 run_tcst_v8 初始化处加入伪标签状态
old_init = '''    _manifest_data = {}'''
# 找一下准确的位置，通常是 optimizer 初始化之前
old_flag = '''    use_rs_crw = bool(cfg["tcst"].get("use_rs_crw", False))'''
new_flag = '''    use_rs_crw = bool(cfg["tcst"].get("use_rs_crw", False))
    use_pseudo = bool(cfg["training"].get("use_pseudo", False))
    pseudo_state = {"xb": None, "yb": None}
    if use_pseudo:
        _log(f"[Pseudo] enabled, thr={cfg['training'].get('pseudo_threshold', 0.9)}, weight={cfg['training'].get('pseudo_weight', 0.5)}")'''
assert old_flag in s, "run_tcst_v8 初始化未找到"
s = s.replace(old_flag, new_flag, 1)

# 2) 在 cycle 循环开始时，预测无标签数据并生成伪标签
old_cycle_start = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()
        qp = None   # T3: 每 cycle 重置，防止跨 cycle 污染'''
new_cycle_start = '''    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\\n--- Cycle {cycle} ---")
        model.eval()
        qp = None
        
        # 生成伪标签
        if use_pseudo:
            with torch.no_grad():
                _probs_list = []
                for _s in range(0, len(tgt_transport_p), 128):
                    _xb = torch.from_numpy(tgt_transport_p[_s:_s+128]).to(device)
                    _lg = model(_xb).logits
                    _probs_list.append(torch.softmax(_lg.float(), dim=1).cpu())
                _probs = torch.cat(_probs_list, 0)
                _conf, _pl = _probs.max(1)
                _thr = float(cfg["training"].get("pseudo_threshold", 0.9))
                _mask = (_conf >= _thr).numpy()
                if _mask.any():
                    pseudo_state["xb"] = tgt_transport_p[_mask]
                    pseudo_state["yb"] = _pl[_mask].numpy().astype(np.int64)
                else:
                    pseudo_state["xb"] = None
                    pseudo_state["yb"] = None
                _log(f"  [Pseudo] selected {_mask.sum()} / {len(tgt_transport_p)} (thr={_thr})")'''
assert old_cycle_start in s, "cycle 开头未找到"
s = s.replace(old_cycle_start, new_cycle_start, 1)

# 3) 在 _train_step 的 loss 计算最后，加入伪标签 loss
old_loss = '''        scaler.scale(loss).backward()'''
new_loss = '''        # Pseudo-label loss
        if use_pseudo and pseudo_state["xb"] is not None:
            pb_size = xb.size(0)
            idx_p = np.random.choice(len(pseudo_state["xb"]), pb_size, replace=True)
            p_xb = torch.from_numpy(pseudo_state["xb"][idx_p]).to(device)
            p_yb = torch.from_numpy(pseudo_state["yb"][idx_p]).to(device)
            pseudo_w = float(cfg["training"].get("pseudo_weight", 0.5))
            
            out_pseudo = model(p_xb)
            loss_pseudo = _ce(out_pseudo.logits, p_yb)
            if _lambda_spec > 0:
                loss_pseudo += _lambda_spec * _ce(out_pseudo.spectral_logits, p_yb)
            loss = loss + pseudo_w * loss_pseudo

        scaler.scale(loss).backward()'''
assert old_loss in s, "_train_step 尾部未找到"
s = s.replace(old_loss, new_loss, 1)

p.write_text(s)
ast.parse(s)
print("✅ trainer_v8.py 已加入目标域伪标签自训练逻辑")
PYEOF

# 生成测试配置：统一架构 + 伪标签
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
    
    # 伪标签设定
    c['training']['use_pseudo'] = True
    c['training']['pseudo_threshold'] = 0.9
    c['training']['pseudo_weight'] = 0.5
    
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/pseudo/{ds}'
    yaml.dump(c, open(f'configs/pseudo_{ds}.yaml','w'), sort_keys=False)
print("✅ 2 个伪标签配置已生成")
PYEOF

mkdir -p results_ablation/pseudo
for ds in hyrank shanghai; do
    echo ""
    echo "=================================================="
    echo "  RUN: ${ds} (统一架构 + 伪标签)   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "configs/pseudo_${ds}.yaml" 2>&1 | \
        tee "results_ablation/pseudo/log_${ds}.log" | \
        grep -E "\[Pseudo\]|OA |AA |Kappa" | tail -8
done

echo ""
echo "=================================================="
echo "  伪标签提分效果对比 (原基线 vs +Pseudo)"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('HyRANK',   'results_ablation/compromise/hyrank/final_metrics.json',   'results_ablation/pseudo/hyrank/final_metrics.json'),
    ('S→H',      'results_ablation/compromise/shanghai/final_metrics.json', 'results_ablation/pseudo/shanghai/final_metrics.json')
]
print(f"{'Dataset':<10} | {'原基线 OA':^10} | {'+伪标签 OA':^10} | {'提升':^10}")
print('-' * 50)
for name, p_base, p_pseudo in variants:
    oa_base = json.load(open(p_base))['target_metrics']['OA'] if os.path.exists(p_base) else 0
    if os.path.exists(p_pseudo):
        m = json.load(open(p_pseudo))['target_metrics']
        oa_pseudo = m['OA']
        pc = [round(x, 1) for x in m['class_accuracy']]
        print(f"{name:<10} | {oa_base:>10.2f} | {oa_pseudo:>10.2f} | {oa_pseudo-oa_base:+.2f} | {pc}")
    else:
        print(f"{name:<10} | {oa_base:>10.2f} | (missing)")
PYEOF

