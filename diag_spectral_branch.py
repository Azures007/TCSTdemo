"""P2 第七节：光谱分支诊断（加载已有 checkpoint，不重训）"""
import sys
from pathlib import Path
ROOT = Path('/root/autodl-tmp/TCSTdemo')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
import torch
import yaml
from core.data import load_scene, build_patches, align_bands
from core.model import DualReliabilityEncoder
from core.trainer_v8 import _spatial_block_split
from core.metrics import classification_metrics, confusion_matrix


DATA_ROOT = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets'
DATASETS = {
    'houston': {
        'src_img': f'{DATA_ROOT}/Houston/Houston13.mat',
        'src_gt':  f'{DATA_ROOT}/Houston/Houston13_7gt.mat',
        'nc': 7, 'ps': 11, 'grid': 10,
        'ckpt_dir': 'results_baseline_s1/houston_s{seed}',
    },
    'hyrank': {
        'src_img': f'{DATA_ROOT}/HyRANK/Dioni.mat',
        'src_gt':  f'{DATA_ROOT}/HyRANK/Dioni_12gt.mat',
        'nc': 12, 'ps': 13, 'grid': 10,
        'ckpt_dir': 'results_baseline_s1/hyrank_s{seed}',
    },
    'shanghai': {
        'src_img': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
        'src_gt':  f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat',
        'nc': 3, 'ps': 11, 'grid': 10,
        'ckpt_dir': 'results_baseline_s1/shanghai_s{seed}',
    },
}
SEEDS = [1237, 42, 2024]


def eval_branch(model, patches, labels, num_classes, branch='fused', device='cpu', batch_size=256):
    """评估指定分支。branch ∈ {fused, context, spectral}"""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(patches), batch_size):
            xb = torch.from_numpy(patches[s:s+batch_size]).to(device)
            out = model(xb)
            if branch == 'context':
                lg = out.context_logits
            elif branch == 'spectral':
                lg = out.spectral_logits
            else:
                lg = out.logits
            preds.append(lg.argmax(1).cpu().numpy())
    preds = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    m = confusion_matrix(labels, preds, num_classes)
    return classification_metrics(m)


def eval_gate(model, patches, device='cpu', batch_size=256):
    """评估 spectral_reliability 分布"""
    model.eval()
    rels = []
    with torch.no_grad():
        for s in range(0, len(patches), batch_size):
            xb = torch.from_numpy(patches[s:s+batch_size]).to(device)
            out = model(xb)
            rels.append(out.spectral_reliability.cpu().numpy())
    rels = np.concatenate(rels) if rels else np.empty(0)
    return rels


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}\n")

    for ds, spec in DATASETS.items():
        print(f"{'='*90}\n[{ds.upper()}] 光谱分支诊断\n{'='*90}")

        # 加载源域数据
        src_cube, src_gt = load_scene(spec['src_img'], spec['src_gt'], None, spec['nc'], verbose=False)
        # 重建 val 划分（和训练时一致：seed=1237+11）
        masks, stats, mixed = _spatial_block_split(
            src_gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
            val_frac_from_train=0.10, seed=1237+11, grid_size=spec['grid'],
            per_class_train_cap=150, per_class_val_target=150, patch_size=spec['ps'])
        val_coords = np.argwhere(masks['val'])
        val_y = (src_gt[val_coords[:, 0], val_coords[:, 1]] - 1).astype(np.int64)
        val_p = build_patches(src_cube, val_coords, spec['ps'])
        print(f"  source val: {len(val_p)} patches  mixed_classes={mixed}")

        # 加载模型（从 seed=1237 的 checkpoint）
        ckpt_path = ROOT / spec['ckpt_dir'].format(seed=1237) / 'final_state.pt'
        if not ckpt_path.exists():
            print(f"  ❌ checkpoint missing: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = DualReliabilityEncoder(bands=src_cube.shape[2], num_classes=spec['nc'],
                                        projection_dim=128, projection_norm='batch',
                                        aspp_mode='adaptive', pooling_mode='guided',
                                        spectral_fusion_mode='reliable',
                                        amplitude_channel=False).to(device)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        # 三个分支在 source val 上的 OA/AA
        print(f"\n  {'branch':<15} {'OA':>8} {'AA':>8} {'Kappa':>8}")
        print(f"  {'-'*45}")
        results = {}
        for branch in ['context', 'spectral', 'fused']:
            m = eval_branch(model, val_p, val_y, spec['nc'], branch=branch, device=device)
            results[branch] = m
            print(f"  {branch:<13} {m['OA']:>8.2f} {m['AA']:>8.2f} {m['Kappa']:>8.4f}")

        # 空间 vs 光谱预测不一致比例
        ctx_pred = []
        spec_pred = []
        with torch.no_grad():
            for s in range(0, len(val_p), 256):
                xb = torch.from_numpy(val_p[s:s+256]).to(device)
                out = model(xb)
                ctx_pred.append(out.context_logits.argmax(1).cpu().numpy())
                spec_pred.append(out.spectral_logits.argmax(1).cpu().numpy())
        ctx_pred = np.concatenate(ctx_pred)
        spec_pred = np.concatenate(spec_pred)
        disagree = float((ctx_pred != spec_pred).mean()) * 100
        print(f"\n  context vs spectral 预测不一致: {disagree:.2f}%")

        # spectral_reliability 分布
        rels = eval_gate(model, val_p, device=device)
        if rels.size > 0:
            print(f"  spectral_reliability: mean={rels.mean():.4f}  "
                  f"std={rels.std():.4f}  min={rels.min():.4f}  max={rels.max():.4f}")
            # 分位数
            qs = np.percentile(rels, [5, 25, 50, 75, 95])
            print(f"  percentiles [5,25,50,75,95]: {[round(float(q),4) for q in qs]}")

        # 如果 spectral 分支 OA 显著低于 context → 光谱分支监督不足
        if results['spectral']['OA'] < results['context']['OA'] - 10:
            print(f"  ⚠️ spectral 比 context 低 {results['context']['OA']-results['spectral']['OA']:.2f} 分 → 光谱分支监督不足")
        elif results['spectral']['OA'] > results['context']['OA'] + 10:
            print(f"  ⚠️ spectral 比 context 高 {results['spectral']['OA']-results['context']['OA']:.2f} 分 → 光谱分支可能过度主导")
        else:
            print(f"  ✅ 两个分支 OA 接近（差 {abs(results['context']['OA']-results['spectral']['OA']):.2f} 分）")
        print()


if __name__ == '__main__':
    main()
