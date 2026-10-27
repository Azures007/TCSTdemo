"""Auto-detecting .mat loader + patch extraction."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


IMAGE_KEYS = ("ori_data", "data", "image", "img", "HSI", "hsi", "X", "cube")
GT_KEYS = ("map", "gt", "GT", "ground_truth", "label", "labels", "Y", "gt_map")


def _load_mat(path):
    try:
        return {k: v for k, v in sio.loadmat(str(path)).items() if not k.startswith("__")}
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k][:]) for k in f.keys()}


def _pick_3d(d):
    for k in IMAGE_KEYS:
        if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 3:
            return d[k], k
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v, k
    return None, None


def _pick_2d(d):
    for k in GT_KEYS:
        if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 2:
            return d[k], k
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v, k
    return None, None


def _band_axis(arr, expected_bands):
    """Find the axis that most likely is the spectral band axis."""
    if expected_bands is not None and expected_bands in arr.shape:
        return int(arr.shape.index(expected_bands))
    # heuristic: axis with size in [20, 500]
    candidates = [i for i, s in enumerate(arr.shape) if 20 <= s <= 500]
    if len(candidates) == 1:
        return candidates[0]
    # fallback: smallest axis
    return int(np.argmin(arr.shape))


def load_scene(image_path, gt_path, expected_bands, num_classes, verbose=True, use_zscore=False):
    image_path = Path(image_path)
    d = _load_mat(image_path)
    cube, key = _pick_3d(d)
    if cube is None:
        raise KeyError(f"{image_path}: no 3-D array found. keys={list(d.keys())}")
    b_axis = _band_axis(cube, expected_bands)
    if b_axis != 2:
        cube = np.moveaxis(cube, b_axis, 2)
    cube = np.asarray(cube, dtype=np.float32)
    if verbose:
        print(f"[load] {image_path.name} key={key} raw_shape={cube.shape} "
              f"band_axis_from={b_axis} num_classes={num_classes}")

    gt = None
    if gt_path is not None:
        gd = _load_mat(Path(gt_path))
        gt_arr, gkey = _pick_2d(gd)
        if gt_arr is None:
            raise KeyError(f"{gt_path}: no 2-D array found. keys={list(gd.keys())}")
        gt = np.asarray(gt_arr, dtype=np.int64)
        if gt.shape != cube.shape[:2]:
            raise ValueError(f"image {cube.shape[:2]} vs gt {gt.shape}")
        if verbose:
            print(f"[load] {Path(gt_path).name} key={gkey} shape={gt.shape} "
                  f"classes={sorted(np.unique(gt).tolist())}")

    finite = np.isfinite(cube).all(-1)
    energy = np.linalg.norm(np.nan_to_num(cube, copy=False), axis=-1)
    valid = finite & (energy > 0)
    cube[~np.isfinite(cube)] = 0.0
    scale = float(np.abs(cube[valid]).max()) if valid.any() else 1.0
    cube = cube / max(scale, 1e-6)
    # Z-score 标准化 vs 逐像素 L2 归一化
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
    return cube, gt


def align_bands(src_cube, tgt_cube, src_stride=1):
    """Match band counts by optional stride on source then uniform on smaller."""
    if src_stride > 1:
        src_cube = src_cube[:, :, ::src_stride]
    bs, bt = src_cube.shape[2], tgt_cube.shape[2]
    target_bands = min(bs, bt)
    if bs != target_bands:
        idx = np.linspace(0, bs - 1, target_bands).round().astype(int)
        src_cube = src_cube[:, :, idx]
    if bt != target_bands:
        idx = np.linspace(0, bt - 1, target_bands).round().astype(int)
        tgt_cube = tgt_cube[:, :, idx]
    return src_cube, tgt_cube, target_bands


def build_patches(image, coords, patch_size):
    r = patch_size // 2
    pad = ((r, r), (r, r), (0, 0))
    img = np.pad(image, pad, mode="reflect")
    patches = np.empty((len(coords), image.shape[2], patch_size, patch_size),
                       dtype=np.float32)
    for i, (row, col) in enumerate(coords):
        rr, cc = row + r, col + r
        patches[i] = img[rr-r:rr+r+1, cc-r:cc+r+1].transpose(2, 0, 1)
    return patches


def stratified_split(gt, ratios, seed):
    """ratios: dict with keys 'train','constraint','calibration','test'."""
    rng = np.random.default_rng(seed)
    masks = {k: np.zeros_like(gt, dtype=bool) for k in ratios}
    for c in [int(i) for i in np.unique(gt) if i > 0]:
        coords = np.argwhere(gt == c)
        rng.shuffle(coords)
        n = len(coords)
        n_tr = max(1, int(round(n * ratios["train"])))
        n_cn = max(1, int(round(n * ratios["constraint"])))
        n_ca = max(1, int(round(n * ratios["calibration"])))
        n_te = n - n_tr - n_cn - n_ca
        if n_te < 0:
            n_te = 0
        slices = {"train": coords[:n_tr],
                  "constraint": coords[n_tr:n_tr+n_cn],
                  "calibration": coords[n_tr+n_cn:n_tr+n_cn+n_ca],
                  "test": coords[n_tr+n_cn+n_ca:n_tr+n_cn+n_ca+n_te]}
        for k, arr in slices.items():
            if len(arr):
                masks[k][arr[:, 0], arr[:, 1]] = True
    return masks


def sample_unlabeled(image, target_mask, n, seed):
    valid = np.argwhere(target_mask)
    rng = np.random.default_rng(seed)
    if len(valid) > n:
        idx = rng.choice(len(valid), size=n, replace=False)
        valid = valid[idx]
    return valid
