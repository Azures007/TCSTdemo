#!/usr/bin/env python3
"""Run HFCST inference on an image without opening target ground truth."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data_hfcst import load_cube
from core.model_hfcst import HFCSTModel


def resolve_path(value, root=ROOT):
    import os
    value = os.path.expandvars(str(value))
    if "$" in value:
        raise ValueError(f"Unexpanded environment variable in path: {value}")
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def extract_batch(padded, coords, patch_size):
    radius = patch_size // 2
    out = np.empty((len(coords), padded.shape[-1], patch_size, patch_size), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        patch = padded[r:r+patch_size, c:c+patch_size]
        out[i] = patch.transpose(2, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="Optional image override; GT is never loaded")
    ap.add_argument("--output", default=None, help="Output .mat path (default: beside checkpoint)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint_path = resolve_path(args.checkpoint)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    meta = ckpt.get("metadata", {})
    if "shared_source_scale" not in meta:
        raise ValueError("Checkpoint has no shared_source_scale; retrain with this HFCST version")
    image_path = resolve_path(args.image or cfg["dataset"]["target_image"])
    expected = cfg.get("dataset", {}).get("expected_bands", {}).get("target")
    cube, valid, audit = load_cube(image_path, expected_bands=expected)
    mapping = cfg.get("dataset", {}).get("band_mapping")
    if mapping:
        ti = mapping.get("target_indices")
        if not isinstance(ti, list) or not ti:
            raise ValueError("band_mapping must contain target_indices for inference")
        if min(ti) < 0 or max(ti) >= cube.shape[-1]:
            raise ValueError("target_indices outside image band range")
        cube = cube[..., np.asarray(ti, dtype=int)]
    if cube.shape[-1] != int(ckpt["bands"]):
        raise ValueError(f"Checkpoint expects {ckpt['bands']} bands, image has {cube.shape[-1]}")
    cube = cube / float(meta["shared_source_scale"])
    model = HFCSTModel(int(ckpt["bands"]), int(ckpt["num_classes"]),
                       feature_dim=int(ckpt.get("feature_dim", 64)),
                       max_gate=float(ckpt.get("max_gate", 0.5))).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    pred_map = np.zeros(cube.shape[:2], dtype=np.uint16)
    coords = np.argwhere(valid)
    patch_size = int(meta.get("patch_size", cfg.get("training", {}).get("patch_size", 11)))
    radius = patch_size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    with torch.no_grad():
        for start in range(0, len(coords), max(1, int(args.batch_size))):
            xy = coords[start:start+max(1, int(args.batch_size))]
            xb = torch.from_numpy(extract_batch(padded, xy, patch_size)).to(device=device, dtype=torch.float32)
            pred = model(xb).logits.argmax(1).cpu().numpy().astype(np.uint16) + 1
            pred_map[xy[:, 0], xy[:, 1]] = pred
    out_path = resolve_path(args.output) if args.output else checkpoint_path.parent / f"{image_path.stem}_prediction.mat"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_path, {"prediction": pred_map, "valid_mask": valid.astype(np.uint8),
                           "shared_source_scale": np.asarray([[meta["shared_source_scale"]]], dtype=np.float32)})
    np.save(out_path.with_suffix(".npy"), pred_map)
    print(f"[HFCST inference] device={device} image={image_path} valid_pixels={len(coords)}")
    print(f"[HFCST inference] saved map={out_path} and array={out_path.with_suffix('.npy')}")
    print("[HFCST inference] target GT was not loaded or used.")

if __name__ == "__main__":
    main()
