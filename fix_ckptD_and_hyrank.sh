#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo

echo "=================================================="
echo "  1. 检查并修复 ckpt_mode=D 逻辑"
echo "=================================================="

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 检查 P1.2 的核心代码是否存在
if 'P1.2 ckpt_mode=D' in s or '_d_best_state' in s:
    print("✅ ckpt_mode=D 逻辑似乎存在。检查是否被遗漏调用...")
else:
    print("❌ 发现 ckpt_mode=D 逻辑丢失，正在重新植入...")
    # 这一步很难用正则精准复原，所以我们直接把核心逻辑插在 force_final 判断之前。
    # 1. 初始化 _d_best_state
    if '_d_best_state' not in s:
        old_init = '    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict())}'
        new_init = '''    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict())}
    _d_tolerance = float(cfg["training"].get("ckpt_d_tolerance", 2.0))
    _d_base_src_val = float(best_src_val)
    _d_best_state = None
    _d_best_nll = float('inf')
    _d_best_meta = None'''
        assert old_init in s, "初始化位置未找到"
        s = s.replace(old_init, new_init, 1)

        # 2. 在 cycle 循环内部，判断 D 模式
        old_cycle_end = '''        cycle_logs.append({"cycle": cycle, "diag": diag,'''
        new_cycle_end = '''        # 添加 D 模式选择逻辑
        if ckpt_mode == "D" and src_val >= _d_base_src_val - _d_tolerance:
            if src_nll < _d_best_nll:
                _d_best_nll = src_nll
                _d_best_state = copy.deepcopy(model.state_dict())
                _d_best_meta = {"cycle": cycle, "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll)}
        
        cycle_logs.append({"cycle": cycle, "diag": diag,'''
        assert old_cycle_end in s, "cycle 末尾未找到"
        s = s.replace(old_cycle_end, new_cycle_end, 1)

        # 3. 在最终选择时，使用 D 模式选出的模型
        old_final = '''    force_final = bool(cfg["training"].get("force_final_epoch", False))'''
        new_final = '''    _d_selected = False
    if ckpt_mode == "D" and _d_best_state is not None:
        model.load_state_dict(_d_best_state)
        best_global["src_val"] = _d_best_meta["src_val"]
        _d_selected = True
        _log(f"\\n[P1.2] ckpt_mode=D: selected cycle {_d_best_meta['cycle']} ep {_d_best_meta['epoch']} (src_val={_d_best_meta['src_val']:.2f}, nll={_d_best_meta['src_nll']:.4f})")

    force_final = bool(cfg["training"].get("force_final_epoch", False))'''
        assert old_final in s, "force_final 位置未找到"
        s = s.replace(old_final, new_final, 1)
        
        # 4. 防止 _d_selected 被覆盖
        old_if = '''    if force_final:'''
        new_if = '''    if not _d_selected and force_final:'''
        assert old_if in s, "if force_final 位置未找到"
        s = s.replace(old_if, new_if, 1)

        old_else = '''    else:
        model.load_state_dict(best_global["state"])'''
        new_else = '''    elif not _d_selected:
        model.load_state_dict(best_global["state"])'''
        assert old_else in s, "else 位置未找到"
        s = s.replace(old_else, new_else, 1)

    p.write_text(s)
    ast.parse(p.read_text())
    print("✅ ckpt_mode=D 逻辑已成功补齐！")
PYEOF

echo ""
echo "=================================================="
echo "  2. 重新跑 HyRANK (带 D 模式选模)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
c = copy.deepcopy(base)
c['training']['use_zscore'] = False
c['training']['amplitude_channel'] = True
c['training']['ckpt_mode'] = 'D'
c['training']['lambda_spec'] = 0.4
c['training']['ce_class_balance_power'] = 0.2
c['training']['use_pseudo'] = True
c['training']['pseudo_threshold'] = 0.8
c['training']['pseudo_per_class'] = 200
c['training']['patch_size'] = 11
c['output']['seed'] = 1237
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_final_best/surgical/hyrank_v2'
yaml.dump(c, open('configs/surgical_hyrank_v2.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

python -u scripts/train.py --config configs/surgical_hyrank_v2.yaml 2>&1 | \
    tee results_final_best/surgical/log_hyrank_v2.log | \
    grep -E "CB-selected|P1.2|OA |AA |Kappa" | tail -6

