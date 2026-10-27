#!/usr/bin/env python3
"""Print a full report from a results directory or json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def show(path: Path):
    d = json.loads(path.read_text())
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    print(f"Dataset: {d.get('dataset')}   bands={d.get('bands')}   "
          f"num_classes={d.get('num_classes')}")
    cycles = d.get("cycles", [])
    for c in cycles:
        tr = c.get("diag", {})
        if "qp_status" in tr:
            print(f"  Cycle {c['cycle']}: QP={tr['qp_status']}  "
                  f"||d*||={tr.get('d_norm', 'NA')}")
        if "candidates" in tr and tr["candidates"]:
            last = tr["candidates"][-1]
            print(f"  Cycle {c['cycle']}: last_cand eta={last['eta']:.3f} "
                  f"V={last['V']:.4f} Rsem={last['Rsem']:.4f} "
                  f"SWDdrop={last['swd_drop']:.4f}")
    m = d.get("target_metrics")
    if m:
        print(f"\nTarget OA={m['OA']:.2f}  AA={m['AA']:.2f}  "
              f"Kappa={m['Kappa']:.4f}  MacroF1={m['MacroF1']:.2f}")
        print(f"Per-class: {[round(x,2) for x in m['class_accuracy']]}")
        cm = m.get("confusion_matrix", [])
        if cm:
            print("\nConfusion matrix:")
            for row in cm:
                print("  " + " ".join(f"{v:4d}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="results directory or a specific final_metrics.json")
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.rglob("final_metrics.json"))
        if not files:
            print(f"No final_metrics.json under {p}")
            return
        for f in files:
            show(f)
    else:
        show(p)


if __name__ == "__main__":
    main()
