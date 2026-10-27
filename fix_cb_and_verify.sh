#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

echo "=================================================="
echo "  1. 正在重新植入 CB-selected 伪标签逻辑"
echo "=================================================="

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 检查是否已经有 CB 逻辑
if '_per_class = int(cfg["training"].get("pseudo_per_class"' in s:
    print("✅ CB 逻辑已存在，无需重复植入")
else:
    # 替换旧的普通伪标签逻辑
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
                _per_class = int(cfg["training"].get("pseudo_per_class", 200))
                _selected_idx = []
                for _c in range(num_classes):
                    _c_idx = np.where((_pl == _c) & (_conf >= _thr))[0]
                    if len(_c_idx) > 0:
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

    if old_pseudo in s:
        s = s.replace(old_pseudo, new_pseudo, 1)
        p.write_text(s)
        print("✅ CB 逻辑已成功植入！")
    else:
        # 尝试匹配有无 enable 日志的变体
        old_pseudo2 = '''                _conf, _pl = _probs.max(1)
                _thr = float(cfg["training"].get("pseudo_threshold", 0.9))
                _mask = (_conf >= _thr).numpy()'''
        if old_pseudo2 in s:
            # 用正则或更宽的替换
            import re
            s = re.sub(r'_conf, _pl = _probs\.max\(1\)[\s\S]*?_log\(f"  \[Pseudo\] selected', new_pseudo + '\n                #_log(f"  [Pseudo] selected', s, count=1)
            p.write_text(s)
            print("✅ CB 逻辑已通过正则成功植入！")
        else:
            print("❌ 没找到旧伪标签代码，请检查 trainer_v8.py 里的 [Pseudo] selected 附近")
            exit(1)

# 检查 Mixup 是否污染
if 'use_mixup' in s:
    print("⚠️ 发现 Mixup 代码，尝试清理...")
    # 清理 Mixup，恢复为纯净的 clean loss
    import re
    s = p.read_text()
    s = re.sub(r'            # Mixup 增强[\s\S]*?out_clean = model\(xb_raw\)\n            loss_clean = _ce\(out_clean\.logits, yb\)\n            if _lambda_spec > 0:\n                loss_clean \+= _lambda_spec \* _ce\(out_clean\.spectral_logits, yb\)', 
               '            out_clean = model(xb_raw)\n            loss_clean = _ce(out_clean.logits, yb)\n            if _lambda_spec > 0:\n                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)', s, count=1)
    p.write_text(s)
    print("✅ Mixup 已清理")

import ast
ast.parse(p.read_text())
print("✅ 语法正确")
PYEOF

echo ""
echo "=================================================="
echo "  2. 验证运行: S→H (Patch=5)"
echo "=================================================="
python - << 'PYEOF'
import yaml
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
base['training']['use_zscore'] = False
base['training']['amplitude_channel'] = True
base['training']['ckpt_mode'] = 'D'
base['training']['lambda_spec'] = 0.4
base['training']['ce_class_balance_power'] = 0.2
base['training']['use_pseudo'] = True
base['training']['pseudo_threshold'] = 0.8
base['training']['pseudo_per_class'] = 200
base['training']['patch_size'] = 5
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_final_best/sh_verify'
yaml.dump(base, open('configs/sh_verify.yaml','w'), sort_keys=False)
PYEOF

python -u scripts/train.py --config configs/sh_verify.yaml 2>&1 | grep -E "CB-selected|OA |AA |Kappa" | tail -5

