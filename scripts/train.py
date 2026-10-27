#!/usr/bin/env python3
"""Universal entry: python scripts/train.py --config configs/houston.yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from core.trainer_v8 import run_tcst_v8


def apply_cli_overrides(cfg, seed=None, save_dir=None):
    if seed is not None:
        cfg["output"]["seed"] = seed
    if save_dir is not None:
        cfg["output"]["save_dir"] = save_dir
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    apply_cli_overrides(cfg, seed=args.seed, save_dir=args.save_dir)
    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    run_tcst_v8(cfg, device)


if __name__ == "__main__":
    main()
