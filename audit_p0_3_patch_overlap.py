"""P0.3: 独立 patch-overlap 审计（不修改训练代码，只审计现有 split）"""
import sys
from pathlib import Path
ROOT = Path('/root/autodl-tmp/TCSTdemo')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
from core.data import load_scene
from core.trainer_v8 import _spatial_block_split


def patch_footprint(center, patch_size):
    """返回 patch 覆盖的像素坐标集合 (r, c)"""
    r = patch_size // 2
    cr, cc = center
    return set((rr, cc2) for rr in range(cr - r, cr + r + 1)
                        for cc2 in range(cc - r, cc + r + 1))


def audit_overlap(masks, gt, patch_size, dataset_name):
    """检查三个集合对之间的 patch 重叠"""
    print(f"\n{'='*80}")
    print(f"[{dataset_name}]  patch_size={patch_size}  审计")
    print(f"{'='*80}")

    results = {}
    pairs = [('train', 'val'), ('constraint', 'val'), ('cal_pool', 'val'),
             ('train', 'constraint'), ('train', 'cal_pool')]

    for set_a, set_b in pairs:
        a_coords = np.argwhere(masks[set_a])
        b_coords = np.argwhere(masks[set_b])
        if len(a_coords) == 0 or len(b_coords) == 0:
            results[(set_a, set_b)] = {'n_a': len(a_coords), 'n_b': len(b_coords),
                                        'overlap_pairs': 0, 'a_overlapped': 0}
            continue

        # 构建 a 的 patch footprint 集合（Union）
        a_footprints = set()
        for (r, c) in a_coords:
            a_footprints |= patch_footprint((r, c), patch_size)

        # 检查 b 的每个 patch 是否与 a 有重叠
        b_overlapped = 0
        for (r, c) in b_coords:
            b_fp = patch_footprint((r, c), patch_size)
            if b_fp & a_footprints:
                b_overlapped += 1

        results[(set_a, set_b)] = {
            'n_a': len(a_coords),
            'n_b': len(b_coords),
            'a_overlapped': b_overlapped,
            'ratio': b_overlapped / max(1, len(b_coords)),
        }

    # 每类统计 train-val 重叠
    n_classes = int(gt.max())
    print(f"\n  {'pair':<20} {'|A|':>6} {'|B|':>6} {'B重叠A':>8} {'比例':>8}")
    print(f"  {'-'*60}")
    for (a, b), r in results.items():
        if 'ratio' in r:
            print(f"  {a+'↔'+b:<20} {r['n_a']:>6} {r['n_b']:>6} {r['a_overlapped']:>8} {r['ratio']*100:>7.2f}%")
        else:
            print(f"  {a+'↔'+b:<20} {r['n_a']:>6} {r['n_b']:>6} (空)")

    # 每类 train-val 重叠
    print(f"\n  每类 train ↔ val 重叠:")
    print(f"  {'class':<6} {'train':>6} {'val':>6} {'val重叠train':>12} {'比例':>8}")
    print(f"  {'-'*50}")
    for c in range(1, n_classes + 1):
        tr_mask = masks['train'] & (gt == c)
        va_mask = masks['val'] & (gt == c)
        tr_c = np.argwhere(tr_mask)
        va_c = np.argwhere(va_mask)
        if len(tr_c) == 0 or len(va_c) == 0:
            print(f"  {c-1:<6} {len(tr_c):>6} {len(va_c):>6} (空)")
            continue
        tr_fp = set()
        for (r, cc) in tr_c:
            tr_fp |= patch_footprint((r, cc), patch_size)
        overlapped = 0
        for (r, cc) in va_c:
            if patch_footprint((r, cc), patch_size) & tr_fp:
                overlapped += 1
        ratio = overlapped / len(va_c)
        flag = ' ⚠️' if ratio > 0.5 else ''
        print(f"  {c-1:<6} {len(tr_c):>6} {len(va_c):>6} {overlapped:>12} {ratio*100:>7.2f}%{flag}")

    return results


def main():
    DATA_ROOT = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets'
    DATASETS = {
        'Houston': {
            'gt': f'{DATA_ROOT}/Houston/Houston13_7gt.mat',
            'img': f'{DATA_ROOT}/Houston/Houston13.mat',
            'nc': 7, 'ps': 11,
        },
        'HyRANK': {
            'gt': f'{DATA_ROOT}/HyRANK/Dioni_12gt.mat',
            'img': f'{DATA_ROOT}/HyRANK/Dioni.mat',
            'nc': 12, 'ps': 13,
        },
        'S→H': {
            'gt': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat',
            'img': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
            'nc': 3, 'ps': 11,
        },
    }

    for name, spec in DATASETS.items():
        _, gt = load_scene(spec['img'], spec['gt'], None, spec['nc'], verbose=False)
        masks, stats, mixed = _spatial_block_split(
            gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
            val_frac_from_train=0.10, seed=1237 + 11, grid_size=10,
            per_class_train_cap=150, per_class_val_target=150,
            patch_size=spec['ps'])
        print(f"\n[{name}]  mixed_classes={mixed}  "
              f"train={stats['train']} val={stats['val']}")
        audit_overlap(masks, gt, spec['ps'], name)


if __name__ == '__main__':
    main()
