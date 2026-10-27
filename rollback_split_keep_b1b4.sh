#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo

BACKUP="core_backup_rollback_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# ============ 1) 从 v8.0.7 + B5/B6 修复前的备份恢复 _spatial_block_split ============
# 备份 core_backup_splitv3_20261023_144751 是 v3 替换前的版本
# 但那个版本已经含 B5/B6 的"扩大候选池"修复，不含 B7/B8 的 v3 特有逻辑
V3_BACKUP="core_backup_splitv3_20261023_144751"
if [ ! -d "$V3_BACKUP" ]; then
    echo "❌ 找不到 $V3_BACKUP"
    echo "   现有备份:"
    ls -d core_backup_* | head -20
    exit 1
fi

python - << 'PYEOF'
from pathlib import Path
import ast, re

# 从 v3 替换前的备份拿 _spatial_block_split（那是 v8.0.7 + B5/B6 修复的版本）
backup_path = Path('core_backup_splitv3_20261023_144751/trainer_v8.py')
if not backup_path.exists():
    raise SystemExit(f"❌ 备份里没有 trainer_v8.py: {backup_path}")

old_s = backup_path.read_text()

# 找 _spatial_block_split 函数（从 def 到下一个 \n\n\n@torch.no_grad()\ndef _sample_boundary_calibration）
start = old_s.find('def _spatial_block_split(')
end = old_s.find('\n\n\n@torch.no_grad()\ndef _sample_boundary_calibration(', start)
if start < 0 or end < 0:
    raise SystemExit("❌ 备份里找不到 _spatial_block_split 或下一个函数标记")

old_func = old_s[start:end]

# 写入当前的 trainer_v8.py
p = Path('core/trainer_v8.py')
cur = p.read_text()
cur_start = cur.find('def _spatial_block_split(')
cur_end = cur.find('\n\n\n@torch.no_grad()\ndef _sample_boundary_calibration(', cur_start)
if cur_start < 0 or cur_end < 0:
    raise SystemExit("❌ 当前文件找不到 _spatial_block_split")

new_s = cur[:cur_start] + old_func + cur[cur_end:]
p.write_text(new_s)
ast.parse(new_s)
print(f"✅ _spatial_block_split 已回滚（v3 → v8.0.7+B5/B6，函数长度 {len(old_func)} 字符）")
PYEOF

# ============ 2) 验证：B1-B4 还在，B7/B8 已消失 ============
echo ""
echo "===== 验证：B1-B4 保留 + split 回滚 ====="
python - << 'PYEOF'
from pathlib import Path
rc = Path('core/risk_controller.py').read_text()
tg = Path('core/task_geometry.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'B1 risk 传 labels': 'transform(xb, phi, labels=yb)' in rc,
    'B2 奇数保底': 'n_boundary = (n + 1) // 2' in tg,
    'B3 完整向量': 'phi_star_full' in tv and 'd_star_full' in tv,
    'B4 SWD 全样本': 'spectral_descriptor(torch.from_numpy(source_cal_patches).to(device))' in rc,
    'B7 已消失': '_remaining_tr[:target_train]' not in tv,
    'B8 已消失': '_effective_val < 50' not in tv,
    'split-v3 日志已消失': '[split-v3]' not in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

# ============ 3) 冒烟：3 数据集 split 统计（应该回到 v8.0.7 的分布） ============
echo ""
echo "===== 冒烟: 3 数据集 split 统计（应该是 v8.0.7 分布）====="
python - << 'PYEOF'
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'core')
import numpy as np
from core.data import load_scene
from core.trainer_v8 import _spatial_block_split

for name, img, gtp, nc, ps in [
    ('Houston', '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13.mat',
     '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13_7gt.mat', 7, 11),
    ('HyRANK', '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni.mat',
     '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni_12gt.mat', 12, 13),
    ('S→H', '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
     '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat', 3, 11),
]:
    _, gt = load_scene(img, gtp, None, nc, verbose=False)
    masks, stats, mixed = _spatial_block_split(
        gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
        val_frac_from_train=0.10, seed=1237+11, grid_size=10,
        per_class_train_cap=150, per_class_val_target=150, patch_size=ps)
    print(f"\n[{name}] train={stats['train']} val={stats['val']} mixed={mixed}")
    print(f"{'cls':<4} {'train':>8} {'val':>6}")
    for c in range(1, nc+1):
        vals = [int((masks[k] & (gt == c)).sum()) for k in ['train', 'val']]
        print(f"  {c-1:<3} {vals[0]:>8} {vals[1]:>6}")
PYEOF

# ============ 4) 重跑 3×3 基线 ============
echo ""
echo "=================================================="
echo "  回滚后 3×3 基线重跑"
echo "  开始: $(date)"
echo "=================================================="
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_baseline_rollback

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['training']['ckpt_mode'] = 'A'
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_baseline_rollback/{ds}_s{seed}'
        yaml.dump(c, open(f'configs/rb_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2"
    echo ""
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "results_baseline_rollback/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|restored|OA |AA " | tail -6
    echo "  DONE: $tag"
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/rb_${ds}_s${seed}.yaml" "${ds}_s${seed}"
    done
done

echo ""
echo "=================================================="
echo "  回滚后 vs v8.0.7 基线"
echo "=================================================="
python - << 'PYEOF' | tee results_baseline_rollback/SUMMARY.txt
import json, os, statistics
OLD = {
    'houston':  (76.12, 74.66),
    'hyrank':   (48.95, 50.89),
    'shanghai': (59.19, 60.05),
}
SEEDS = [1237, 42, 2024]
for ds in ['houston', 'hyrank', 'shanghai']:
    oas, aas = [], []
    print(f"\n{'='*100}")
    print(f"[{ds}]  v8.0.7: OA={OLD[ds][0]:.2f}  AA={OLD[ds][1]:.2f}")
    print(f"{'seed':<8} {'OA':>8} {'AA':>8}")
    print('-'*100)
    for seed in SEEDS:
        p = f'results_baseline_rollback/{ds}_s{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  {seed}: missing"); continue
        m = json.load(open(p))['target_metrics']
        oas.append(m['OA']); aas.append(m['AA'])
        print(f"  {seed:<6} {m['OA']:>8.2f} {m['AA']:>8.2f}")
    if len(oas) >= 2:
        moa, soa = statistics.mean(oas), statistics.stdev(oas)
        maa, saa = statistics.mean(aas), statistics.stdev(aas)
        print(f"{'mean±std':<8} {moa:>5.2f}±{soa:>4.2f} {maa:>5.2f}±{saa:>4.2f}")
        print(f"  Δ vs v8.0.7: OA {moa-OLD[ds][0]:+.2f}  AA {maa-OLD[ds][1]:+.2f}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 回滚后基线完成  $(date)"
echo "=================================================="
