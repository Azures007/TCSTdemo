#!/bin/bash
set -e
cd ~/autodl-tmp/TCSTdemo

echo "============================================================"
echo "  P0.0 环境审计"
echo "============================================================"
python - << 'PYEOF'
import sys
print(f"Python: {sys.version.split()[0]}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"PyTorch: MISSING ({e})")
for mod in ['numpy', 'scipy', 'h5py', 'yaml', 'cvxpy']:
    try:
        m = __import__(mod)
        print(f"{mod}: {getattr(m, '__version__', 'OK')}")
    except ImportError:
        print(f"{mod}: MISSING")
PYEOF

echo ""
echo "============================================================"
echo "  P0.0 数据文件存在性"
echo "============================================================"
DATA_ROOT="/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets"
for f in \
    "$DATA_ROOT/Houston/Houston13.mat" \
    "$DATA_ROOT/Houston/Houston13_7gt.mat" \
    "$DATA_ROOT/Houston/Houston18.mat" \
    "$DATA_ROOT/Houston/Houston18_7gt.mat" \
    "$DATA_ROOT/HyRANK/Dioni.mat" \
    "$DATA_ROOT/HyRANK/Dioni_12gt.mat" \
    "$DATA_ROOT/HyRANK/Loukia.mat" \
    "$DATA_ROOT/HyRANK/Loukia_12gt.mat" \
    "$DATA_ROOT/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat" \
    "$DATA_ROOT/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat" \
    "$DATA_ROOT/Shanghai_Hangzhou/Shanghai-Hangzhou/Hangzhou.mat" \
    "$DATA_ROOT/Shanghai_Hangzhou/Shanghai-Hangzhou/Hangzhou_3gt.mat"
do
    if [ -f "$f" ]; then
        sz=$(du -h "$f" | cut -f1)
        echo "  ✅ $f ($sz)"
    else
        echo "  ❌ MISSING: $f"
    fi
done

echo ""
echo "============================================================"
echo "  P0.1 数据审计（原始 .mat，不经 load_scene 归一化）"
echo "============================================================"
python - << 'PYEOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
import numpy as np
from core.data import _load_mat, _pick_3d, _pick_2d, _band_axis

DATA_ROOT = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets'

DATASETS = {
    'Houston': {
        'src_img': f'{DATA_ROOT}/Houston/Houston13.mat',
        'src_gt':  f'{DATA_ROOT}/Houston/Houston13_7gt.mat',
        'tgt_img': f'{DATA_ROOT}/Houston/Houston18.mat',
        'tgt_gt':  f'{DATA_ROOT}/Houston/Houston18_7gt.mat',
        'num_classes': 7,
    },
    'HyRANK': {
        'src_img': f'{DATA_ROOT}/HyRANK/Dioni.mat',
        'src_gt':  f'{DATA_ROOT}/HyRANK/Dioni_12gt.mat',
        'tgt_img': f'{DATA_ROOT}/HyRANK/Loukia.mat',
        'tgt_gt':  f'{DATA_ROOT}/HyRANK/Loukia_12gt.mat',
        'num_classes': 12,
    },
    'S→H': {
        'src_img': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
        'src_gt':  f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat',
        'tgt_img': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Hangzhou.mat',
        'tgt_gt':  f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Hangzhou_3gt.mat',
        'num_classes': 3,
    },
}

def read_raw(img_path, gt_path):
    d = _load_mat(img_path)
    cube, _ = _pick_3d(d)
    if cube is None:
        return None, None, None
    b_axis = _band_axis(cube, None)
    if b_axis != 2:
        cube = np.moveaxis(cube, b_axis, 2)
    cube = np.asarray(cube, dtype=np.float32)
    gd = _load_mat(gt_path)
    gt, _ = _pick_2d(gd)
    if gt is not None:
        gt = np.asarray(gt, dtype=np.int64)
    return cube, gt, b_axis

for name, spec in DATASETS.items():
    print(f"\n{'='*70}\n[{name}]  num_classes={spec['num_classes']}\n{'='*70}")

    src_cube, src_gt, src_b_axis = read_raw(spec['src_img'], spec['src_gt'])
    tgt_cube, tgt_gt, tgt_b_axis = read_raw(spec['tgt_img'], spec['tgt_gt'])

    if src_cube is None:
        print(f"  ❌ 源域读取失败: {spec['src_img']}")
        continue
    if tgt_cube is None:
        print(f"  ❌ 目标域读取失败: {spec['tgt_img']}")
        continue

    print(f"  源域: shape={src_cube.shape}  band_axis(from)={src_b_axis}")
    print(f"  目标: shape={tgt_cube.shape}  band_axis(from)={tgt_b_axis}")

    # NaN / Inf
    src_nan = np.isnan(src_cube).sum()
    src_inf = np.isinf(src_cube).sum()
    tgt_nan = np.isnan(tgt_cube).sum()
    tgt_inf = np.isinf(tgt_cube).sum()
    print(f"  源域 NaN={src_nan}  Inf={src_inf}    目标 NaN={tgt_nan}  Inf={tgt_inf}")

    # 归一化前的光谱幅值统计（前 1000 个有效像素）
    src_energy = np.linalg.norm(src_cube.reshape(-1, src_cube.shape[2]), axis=1)
    src_valid_energy = src_energy[src_energy > 0]
    if len(src_valid_energy) > 0:
        print(f"  源域 L2 norm (归一化前): min={src_valid_energy.min():.4f}  "
              f"median={np.median(src_valid_energy):.4f}  max={src_valid_energy.max():.4f}")

    # 标签审计
    if src_gt is not None:
        src_labels = sorted(np.unique(src_gt).tolist())
        print(f"  源域 label 唯一值: {src_labels}")
        print(f"  {'类':<8} {'源域像素':>10} {'目标域像素':>10}")
        for c in src_labels:
            if c == 0: continue
            sc = int((src_gt == c).sum())
            tc = int((tgt_gt == c).sum()) if tgt_gt is not None else 0
            print(f"    {c-1:<6} {sc:>10} {tc:>10}")

    if tgt_gt is not None:
        tgt_labels = sorted(np.unique(tgt_gt).tolist())
        print(f"  目标域 label 唯一值: {tgt_labels}")

PYEOF

echo ""
echo "============================================================"
echo "  P0.1 HyRANK class2/class5 源域真实像素数（重点核查）"
echo "============================================================"
python - << 'PYEOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
import numpy as np
from core.data import _load_mat, _pick_2d

gt_path = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni_12gt.mat'
gd = _load_mat(gt_path)
gt, _ = _pick_2d(gd)
gt = np.asarray(gt, dtype=np.int64)
print(f"HyRANK Dioni 源域 label 分布:")
print(f"{'class_idx':<12} {'mat_label':>10} {'pixels':>10}")
for lbl in sorted(np.unique(gt).tolist()):
    if lbl == 0: continue
    n = int((gt == lbl).sum())
    print(f"  {lbl-1:<10} {lbl:>10} {n:>10}")
print(f"\n总标注像素: {int((gt > 0).sum())}")
PYEOF
