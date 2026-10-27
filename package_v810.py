from pathlib import Path
import yaml, subprocess, os, tarfile

ROOT = Path('/root/autodl-tmp/TCSTdemo')
FINAL = ROOT / 'configs_FINAL_v810'
FINAL.mkdir(exist_ok=True)

# 最终配置：统一 ckpt_mode=A, use_warmup_augmentation=false
BASES = {
    'houston':  ('configs/hou_H6_ls05.yaml',    76.12, 74.66),
    'hyrank':   ('configs/hy_Y2_warm40.yaml',   54.28, 47.69),
    'shanghai': ('configs/sh_V2_lr1e4.yaml',    63.93, 65.14),
}
for ds, (bp, oa, aa) in BASES.items():
    c = yaml.safe_load(open(ROOT / bp))
    c['training']['ckpt_mode'] = 'A'
    c['training']['use_warmup_augmentation'] = False
    c['training']['lambda_aug'] = 0.0
    c['training']['projection_norm'] = 'batch'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = str(ROOT / f'results_FINAL_v810/{ds}')
    yaml.dump(c, open(FINAL / f'{ds}_final.yaml', 'w', encoding='utf-8'),
              sort_keys=False, allow_unicode=True)
    print(f"✅ {ds}: OA={oa:.2f} AA={aa:.2f}")

readme = """# T-CST v8.1.0 (Strict Final, P0/P1/P2 complete)

## Final Results (3 seeds: 1237, 42, 2024, unified ckpt_mode=A, no warmup augmentation)
| Dataset | OA (mean±std) | AA (mean±std) | Best single |
|---|---|---|---|
| Houston13->18 | 76.12 ± 1.12 | 74.66 ± 4.34 | 77.30 (seed=2024) |
| Dioni->Loukia | 54.28 ± 3.09 | 47.69 ± 3.38 | 57.13 (seed=2024) |
| Shanghai->Hangzhou | 63.93 ± 10.11 | 65.14 ± 6.63 | 74.75 (seed=1237) |

## What is new vs v8.0.9

### P0: data / split / logging audit
- P0.1: data audit reveals HyRANK class1=204, class5=361 pixels (NOT <30 as README v8.0.9 claimed).
- P0.2: all three datasets have identical source/target band counts; align_bands never triggers.
- P0.3: patch-overlap audit reveals Houston has 100% same-class train/val overlap, HyRANK has 4/12 classes at 100%, S->H has 0% same-class overlap.
- P0.3 fix: added `strict_mixed_isolation` switch (default False = v8.0.9 behavior; True disables fallback but Houston 7/7 classes get val=0).
- P0.4: replaced hardcoded "10% source train labels" with dynamic ratio; added per-class train vs total labeled comparison.

### P1: checkpoint selection and adaptation verification
- P1.1: saves full checkpoint manifest (warmup + cycle entries with src_val, src_nll, safe_executed_count, is_best, param_diff).
- P1.2: added ckpt_mode=D (qualified set + NLL ranking). Results:
  - Houston: D 78.33±1.19 (+2.21 vs A)
  - HyRANK: D 56.90±5.30 (+2.62 vs A, but std 3.09->5.30)
  - S->H: D 50.63±12.63 (6-seed, -13.30 vs A, min 28.90)
  - **Decision: keep ckpt_mode=A unified; D is reported as ablation.**
- P1.3: per-cycle safe execution verified:
  - Houston: safe=60/60 (T-CST accepted 4/4, all batches executed)
  - HyRANK: safe=0/51 (T-CST rejected 6/6, only pure CE)
  - S->H: safe=0/15 (T-CST rejected 2/2, only pure CE)
- P1 augmentation decoupling: added `use_warmup_augmentation` (default False). Ablation:
  - Houston: -0.97 OA, -3.20 AA (harmful)
  - S->H: +1.13 OA but std 10.11->15.85 (unstable)
  - **Decision: keep disabled.**

### P2: MMD negative transfer diagnosis
- per_class_margin: at each ACCEPT, records mean margin delta per class.
- tgt_pred_dist: records target unlabeled prediction distribution and entropy per cycle.
- Key findings:
  - **Houston**: T-CST improves class0/class5 margins, but target predictions concentrate on class5 (0.37-0.79), class3 almost 0.
  - **HyRANK**: pure CE cycles push predictions to class6 (0.24-0.50), class1/class5 almost 0 (though they have 204/361 source pixels).
  - **S->H**: extreme instability. Predictions swing between Water/Land/Plant across seeds and cycles:
    - seed 1237: Water 0.58 -> Plant 0.90
    - seed 42: Land 0.61 -> Water 0.58
    - seed 2024: Land 0.55 -> Plant 0.47
    - This is the direct cause of S->H std=10.11.

## Root causes (final)
1. **Houston 76.12**: warmup saturates at 100% src_val. D-mode shows T-CST can give +2.21 if checkpoint selection bypasses saturation, but D hurts S->H by -13.30, so unified A is chosen.
2. **HyRANK 54.28**: T-CST always rejected (swd_rel ~3e-3 < 0.01). Pure CE cycles help src_val (77->79) but lose rare classes (class1/class5 -> 0% target predictions).
3. **S->H 63.93±10.11**: mode collapse across seeds. Target predictions swing between Water/Land/Plant. T-CST always rejected. No checkpoint selection strategy fixes this.

## Reproduce
python -u scripts/train.py --config configs_FINAL_v810/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v810/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v810/shanghai_final.yaml
"""
(ROOT / 'README_V810.md').write_text(readme, encoding='utf-8')
print("✅ README_V810.md written")

os.chdir(ROOT)
if (ROOT / 'TCST_v8.1.0_Strict_Final.tar.gz').exists():
    (ROOT / 'TCST_v8.1.0_Strict_Final.tar.gz').unlink()

subprocess.run([
    'tar', '-czf', 'TCST_v8.1.0_Strict_Final.tar.gz',
    '--exclude=results*', '--exclude=*.pyc', '--exclude=__pycache__',
    '--exclude=*.log', '--exclude=.git',
    '--exclude=core_backup*', '--exclude=configs_backup*',
    'core/', 'configs/', 'configs_FINAL_v810/', 'scripts/', 'tests/',
    'docs/', 'requirements.txt', 'README_V810.md',
], check=True)
size = (ROOT / 'TCST_v8.1.0_Strict_Final.tar.gz').stat().st_size
print(f"\n✅ 打包完成: TCST_v8.1.0_Strict_Final.tar.gz ({size/1024:.1f} KB)")
