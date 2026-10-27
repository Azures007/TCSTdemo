#!/usr/bin/env python3
"""Print a full report across all TCST runs under results/."""
import json
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"No results dir: {root}")
        return

    runs = []
    for jp in sorted(root.rglob("final_metrics.json")):
        try:
            runs.append((jp, json.loads(jp.read_text())))
        except Exception:
            pass

    print(f"\n{'='*110}")
    print(f"{'run':<38} {'OA':>7} {'AA':>7} {'Kappa':>8} {'MacroF1':>9} "
          f"{'best_score':>11} {'cycles':>7}")
    print('-'*110)
    for jp, d in runs:
        m = d.get("target_metrics")
        if not m:
            continue
        tag = str(jp.parent.relative_to(root))
        score = d.get("best_score", float('nan'))
        ncyc = len(d.get("cycles", []))
        print(f"{tag:<38} {m['OA']:>7.2f} {m['AA']:>7.2f} {m['Kappa']:>8.4f} "
              f"{m['MacroF1']:>9.2f} {score:>11.2f} {ncyc:>7}")

    if args.detail:
        for jp, d in runs:
            m = d.get("target_metrics")
            if not m:
                continue
            tag = str(jp.parent.relative_to(root))
            print(f"\n{'='*110}\n{tag}\n{'='*110}")
            print(f"OA={m['OA']:.2f}  AA={m['AA']:.2f}  "
                  f"Kappa={m['Kappa']:.4f}  MacroF1={m['MacroF1']:.2f}")
            print(f"per-class: {[round(x,2) for x in m['class_accuracy']]}")
            for c in d.get("cycles", []):
                diag = c.get("diag", {})
                cands = diag.get("candidates", [])
                if cands:
                    last = cands[-1]
                    print(f"  cycle {c['cycle']}: score={c.get('score'):.2f} "
                          f"eta={last['eta']:.3f} "
                          f"tgt_flip={last.get('target_flip', 'NA')} "
                          f"swd_drop={last['swd_drop']:+.2e} "
                          f"phi_norm={last.get('phi_norm', 0):.3f}")


if __name__ == "__main__":
    main()
