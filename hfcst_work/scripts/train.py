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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    run_tcst_v8(cfg, device)


if __name__ == "__main__":
    main()
