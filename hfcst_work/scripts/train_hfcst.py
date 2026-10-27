#!/usr/bin/env python3
"""Train HFCST-v9: python -u scripts/train_hfcst.py --config configs_hfcst/houston.yaml"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.trainer_hfcst import run_hfcst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="HFCST YAML config")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    args = ap.parse_args()
    path = Path(args.config)
    if not path.is_absolute(): path = (ROOT / path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict): raise ValueError(f"Invalid YAML config: {path}")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}", flush=True)
    run_hfcst(cfg, device, config_path=path)

if __name__ == "__main__":
    main()
