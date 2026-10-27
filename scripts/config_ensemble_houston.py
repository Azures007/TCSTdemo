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
from core.metrics import classification_metrics, confusion_matrix


def build_model_from_cfg(cfg, bands, device):
    """从 config 构建模型"""
    tr = cfg['training']
    ds = cfg['dataset']
    model = DualReliabilityEncoder(
        bands=bands,
        num_classes=ds['num_classes'],
        projection_dim=tr.get('projection_dim', 128),
        projection_norm=tr.get('projection_norm', 'batch'),
        aspp_mode=tr.get('aspp_mode', 'adaptive'),
        pooling_mode=tr.get('pooling_mode', 'guided'),
        spectral_fusion_mode=tr.get('spectral_fusion_mode', 'reliable'),
        amplitude_channel=tr.get('amplitude_channel', False),
    ).to(device)
    return model


def predict_probs_tta(model, patches, device):
    preds = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            logits = model(xb).logits
            logits = logits + model(torch.flip(xb, dims=(-1,))).logits
            logits = logits + model(torch.flip(xb, dims=(-2,))).logits
            logits = logits / 3.0
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ===== 1. 加载 config =====
    # 用 C3 的 config 作为数据/架构模板
    with open(ROOT / 'configs/houston_C3_rho15.yaml') as f:
        cfg = yaml.safe_load(f)

    print("Loading Houston data...")
    src_cube, src_gt = load_scene(
        cfg['dataset']['source_image'], cfg['dataset']['source_gt'],
        None, cfg['dataset']['num_classes'], verbose=False)
    tgt_cube, tgt_gt = load_scene(
        cfg['dataset']['target_image'], cfg['dataset']['target_gt'],
        None, cfg['dataset']['num_classes'], verbose=False)
    src_cube, tgt_cube, bands = align_bands(src_cube, tgt_cube)

    tgt_valid = np.linalg.norm(tgt_cube, axis=-1) > 0
    tgt_eval_c = np.argwhere((tgt_gt > 0) & tgt_valid)
    tgt_eval_y = (tgt_gt[tgt_eval_c[:, 0], tgt_eval_c[:, 1]] - 1).astype(np.int64)
    tgt_eval_p = build_patches(tgt_cube, tgt_eval_c, cfg['training']['patch_size'])
    print(f"Target eval: {len(tgt_eval_p)} samples, bands={bands}")

    # ===== 2. 加载三个模型并预测 =====
    ckpts = [
        ("C2 (K16)", "results_core_tuning/houston_C2_K16/final_state.pt"),
        ("C3 (rho15)", "results_core_tuning/houston_C3_rho15/final_state.pt"),
        ("叠加 (K16+rho15)", "results_core_tuning/houston_combo_K16_rho15/final_state.pt"),
    ]

    all_probs = []
    per_model_metrics = []
    for name, ckpt_path in ckpts:
        full_path = ROOT / ckpt_path
        if not full_path.exists():
            print(f"⚠️ MISSING: {full_path}")
            continue
        print(f"\nLoading {name} from {full_path}")

        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        model = build_model_from_cfg(cfg, bands, device)
        try:
            model.load_state_dict(ckpt['model'], strict=True)
        except Exception as e:
            print(f"  ⚠️ strict load failed: {e}, trying strict=False")
            model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        probs = predict_probs_tta(model, tgt_eval_p, device)
        all_probs.append(probs)

        preds = probs.argmax(1).numpy()
        m = confusion_matrix(tgt_eval_y, preds, cfg['dataset']['num_classes'])
        metrics = classification_metrics(m)
        per_model_metrics.append((name, metrics))
        print(f"  {name}: OA={metrics['OA']:.2f}  AA={metrics['AA']:.2f}  "
              f"Kappa={metrics['Kappa']:.4f}  MacroF1={metrics['MacroF1']:.2f}")
        print(f"  per-class: {[round(x,2) for x in metrics['class_accuracy']]}")

    if len(all_probs) < 2:
        print("\n⚠️ 有效模型不足 2 个，无法集成")
        return

    # ===== 3. 配置集成（概率平均）=====
    final_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    final_preds = final_probs.argmax(1).numpy()

    m = confusion_matrix(tgt_eval_y, final_preds, cfg['dataset']['num_classes'])
    metrics = classification_metrics(m)

    print(f"\n{'='*60}")
    print(f"CONFIG ENSEMBLE — Houston ({len(all_probs)} models)")
    print(f"{'='*60}")
    print(f"OA     = {metrics['OA']:.2f}")
    print(f"AA     = {metrics['AA']:.2f}")
    print(f"Kappa  = {metrics['Kappa']:.4f}")
    print(f"MacroF1= {metrics['MacroF1']:.2f}")
    print(f"Per-class: {[round(x,2) for x in metrics['class_accuracy']]}")
    print(f"{'='*60}")

    import json
    out = ROOT / 'results_core_tuning/houston_config_ensemble.json'
    with open(out, 'w') as f:
        json.dump({
            "per_model": [{"name": n, "metrics": m} for n, m in per_model_metrics],
            "ensemble": metrics
        }, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
