#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_zscore_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# 1) 修改 data.py，加入 zscore 开关（默认 legacy L2 以保持历史兼容）
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/data.py')
s = p.read_text()

old_norm = '''    norms = np.linalg.norm(cube, axis=-1, keepdims=True)
    cube = cube / np.maximum(norms, 1e-6)
    cube[~valid] = 0.0
    return cube, gt'''

new_norm = '''    # Z-score 标准化 vs 逐像素 L2 归一化
    if use_zscore:
        # 计算全局均值与标准差 (只在 valid 像素上)
        valid_pixels = cube[valid]
        if len(valid_pixels) > 0:
            mean = valid_pixels.mean(axis=0)
            std = valid_pixels.std(axis=0)
            std = np.maximum(std, 1e-6)
            cube = (cube - mean) / std
        cube[~valid] = 0.0
    else:
        # legacy L2 归一化
        norms = np.linalg.norm(cube, axis=-1, keepdims=True)
        cube = cube / np.maximum(norms, 1e-6)
        cube[~valid] = 0.0
    return cube, gt'''

# 1. 改 load_scene 签名
s = s.replace('def load_scene(image_path, gt_path, expected_bands, num_classes, verbose=True):',
              'def load_scene(image_path, gt_path, expected_bands, num_classes, verbose=True, use_zscore=False):')

# 2. 替换归一化逻辑
assert old_norm in s, "data.py 归一化代码未找到"
s = s.replace(old_norm, new_norm, 1)
p.write_text(s)
ast.parse(p.read_text())
print("✅ data.py 已支持 use_zscore 参数")

# 3) trainer_v8.py 传入 use_zscore
p2 = Path('core/trainer_v8.py')
s2 = p2.read_text()
old_call = '''    src_cube, src_gt = load_scene(cfg["dataset"]["source_image"],
                                  cfg["dataset"]["source_gt"],
                                  None, cfg["dataset"]["num_classes"], verbose=True)'''
new_call = '''    _use_zscore = bool(cfg["training"].get("use_zscore", False))
    src_cube, src_gt = load_scene(cfg["dataset"]["source_image"],
                                  cfg["dataset"]["source_gt"],
                                  None, cfg["dataset"]["num_classes"], verbose=True,
                                  use_zscore=_use_zscore)'''
assert old_call in s2, "trainer 的 src load_scene 未找到"
s2 = s2.replace(old_call, new_call, 1)

old_call2 = '''    tgt_cube, _ = load_scene(cfg["dataset"]["target_image"],
                             None,
                             None, cfg["dataset"]["num_classes"], verbose=True)'''
new_call2 = '''    tgt_cube, _ = load_scene(cfg["dataset"]["target_image"],
                             None,
                             None, cfg["dataset"]["num_classes"], verbose=True,
                             use_zscore=_use_zscore)'''
assert old_call2 in s2, "trainer 的 tgt load_scene 未找到"
s2 = s2.replace(old_call2, new_call2, 1)

p2.write_text(s2)
ast.parse(s2)
print("✅ trainer_v8.py 已传入 use_zscore")
PYEOF

# 4) 生成 S→H Z-score 配置并训练
echo "=================================================="
echo "  S→H: Z-score 标准化实验 (Patch=11, Mode=A)"
echo "  开始: $(date)"
echo "=================================================="
mkdir -p results_ablation/sh_zscore

python - << 'PYEOF'
import yaml
base = yaml.safe_load(open('configs/sh_V2_lr1e4.yaml'))
base['training']['use_zscore'] = True
base['output']['seed'] = 1237
base['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_ablation/sh_zscore/mode_A'
yaml.dump(base, open('configs/sh_zscore.yaml','w'), sort_keys=False)
print("✅ 配置已生成")
PYEOF

python -u scripts/train.py --config configs/sh_zscore.yaml 2>&1 | \
    tee results_ablation/sh_zscore/log.txt | \
    grep -E "\[risk\]|OA |AA |Kappa" | tail -8

echo ""
echo "=================================================="
echo "  结果对比 (S→H)"
echo "=================================================="
python - << 'PYEOF'
import json, os
variants = [
    ('L2 Norm, Patch=11, Mode=A', 'results_baseline_s1/shanghai_s1237/final_metrics.json'),
    ('Z-score, Patch=11, Mode=A', 'results_ablation/sh_zscore/mode_A/final_metrics.json')
]
print(f"{'Variant':<28} | {'OA':^8} | {'AA':^8} | {'Kappa':^8} | Per-Class")
print('-' * 90)
for name, p in variants:
    if not os.path.exists(p):
        print(f"{name:<28} | (missing)")
        continue
    m = json.load(open(p))['target_metrics']
    pc = [round(x, 1) for x in m['class_accuracy']]
    print(f"{name:<28} | {m['OA']:>8.2f} | {m['AA']:>8.2f} | {m['Kappa']:>8.4f} | {pc}")
PYEOF

