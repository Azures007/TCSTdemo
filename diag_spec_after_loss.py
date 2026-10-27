import sys
from pathlib import Path
ROOT = Path('/root/autodl-tmp/TCSTdemo')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
import torch
import yaml
from core.data import load_scene, build_patches
from core.model import DualReliabilityEncoder
from core.trainer_v8 import _spatial_block_split
from core.metrics import classification_metrics, confusion_matrix

DATA_ROOT = '/root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets'
DATASETS = {
    'houston': {'img': f'{DATA_ROOT}/Houston/Houston13.mat', 'gt': f'{DATA_ROOT}/Houston/Houston13_7gt.mat', 'nc': 7, 'ps': 11, 'grid': 10},
    'hyrank': {'img': f'{DATA_ROOT}/HyRANK/Dioni.mat', 'gt': f'{DATA_ROOT}/HyRANK/Dioni_12gt.mat', 'nc': 12, 'ps': 13, 'grid': 10},
    'shanghai': {'img': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat', 'gt': f'{DATA_ROOT}/Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat', 'nc': 3, 'ps': 11, 'grid': 10},
}

def eval_branch(model, patches, labels, num_classes, branch='fused', device='cpu'):
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            out = model(xb)
            if branch == 'context': lg = out.context_logits
            elif branch == 'spectral': lg = out.spectral_logits
            else: lg = out.logits
            preds.append(lg.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return classification_metrics(confusion_matrix(labels, preds, num_classes))

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds, spec in DATASETS.items():
        print(f"\n{'='*60}\n[{ds.upper()}] 光谱分支诊断（引入 Aux Loss 后）\n{'='*60}")
        
        # 加载源域数据，重建验证集
        src_cube, src_gt = load_scene(spec['img'], spec['gt'], None, spec['nc'], verbose=False)
        masks, _, _ = _spatial_block_split(src_gt, 1.0, 12, 12, 0.10, seed=1237+11, grid_size=spec['grid'], per_class_train_cap=150, per_class_val_target=150, patch_size=spec['ps'])
        val_c = np.argwhere(masks['val'])
        val_y = (src_gt[val_c[:, 0], val_c[:, 1]] - 1).astype(np.int64)
        val_p = build_patches(src_cube, val_c, spec['ps'])
        
        # 加载模型 (刚才跑出来的 Mode A + Spec 版本)
        ckpt_path = ROOT / f'results_ablation/unified_spec_modeA/{ds}/final_state.pt'
        if not ckpt_path.exists():
            print("  ❌ 找不到模型"); continue
            
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = DualReliabilityEncoder(
            bands=src_cube.shape[2], num_classes=spec['nc'], projection_dim=128,
            projection_norm='batch', aspp_mode='adaptive', pooling_mode='guided',
            spectral_fusion_mode='reliable', amplitude_channel=True  # 注意：这里开了 Amp
        ).to(device)
        model.load_state_dict(ckpt['model'], strict=False)
        
        # 分别评估
        for branch in ['context', 'spectral', 'fused']:
            m = eval_branch(model, val_p, val_y, spec['nc'], branch=branch, device=device)
            print(f"  {branch.capitalize():<10}: OA = {m['OA']:.2f}%   AA = {m['AA']:.2f}%")

if __name__ == '__main__': main()
