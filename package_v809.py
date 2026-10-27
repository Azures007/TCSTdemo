from pathlib import Path
import yaml, subprocess, os, tarfile

ROOT = Path('/root/autodl-tmp/TCSTdemo')
FINAL = ROOT / 'configs_FINAL_v809'
FINAL.mkdir(exist_ok=True)

# 三数据集统一：ckpt_mode=A, projection_norm=batch, warmup scheduler T_max=warmup_epochs
BASES = {
    'houston':  ('configs/hou_H6_ls05.yaml',    76.12, 74.66),
    'hyrank':   ('configs/hy_Y2_warm40.yaml',   54.28, 47.69),
    'shanghai': ('configs/sh_V2_lr1e4.yaml',    63.93, 65.14),
}
for ds, (bp, oa, aa) in BASES.items():
    c = yaml.safe_load(open(ROOT / bp))
    c['training']['ckpt_mode'] = 'A'
    c['training']['projection_norm'] = 'batch'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = str(ROOT / f'results_FINAL_v809/{ds}')
    yaml.dump(c, open(FINAL / f'{ds}_final.yaml', 'w', encoding='utf-8'),
              sort_keys=False, allow_unicode=True)
    print(f"✅ {ds}: OA={oa:.2f} AA={aa:.2f}")

readme = """# T-CST v8.0.9 (Strict Final, S1 scheduler fix, 3-seed mean±std)

## Final Results (3 seeds: 1237, 42, 2024)
| Dataset | OA (mean±std) | AA (mean±std) | Best single seed |
|---|---|---|---|
| Houston13->18 | 76.12 ± 1.12 | 74.66 ± 4.34 | 77.30 (seed=2024) |
| Dioni->Loukia | 54.28 ± 3.09 | 47.69 ± 3.38 | 57.13 (seed=2024) |
| Shanghai->Hangzhou | 63.93 ± 10.11 | 65.14 ± 6.63 | 74.75 (seed=1237) |

## Key change from v8.0.8: S1 scheduler fix
Warmup scheduler T_max changed from total_epochs (warmup+cycles) to warmup_epochs.
- Before: HyRANK T_max=70, lr at epoch 40 is 1.24e-4 (not at min)
- After: HyRANK T_max=40, lr at epoch 40 reaches 1e-6 (min_lr)
- Effect: HyRANK 49.31 -> 54.28 (+4.97, seed 42 +8.65, seed 2024 +6.57)
- Houston/S->H use lr_scheduler=none, so S1 does not affect them (verified: bit-identical to v8.0.8)

## P0/P1 fixes (T1-T9) kept but neutral
- T1: SWD unequal-length (quantile interpolation)
- T2: jsonl records phi_star_full/d_star_full
- T3: qp=None per cycle
- T4: cycles scheduler T_max=remaining_epochs
- T5: aug_class_mask merged with aug_classes
- T6: Risk Search thresholds configurable (gain_tv_max, offset_tv_max, rel_change_max, rsem_delta)
- T7: MMD target re-sampled per cycle
- T8: Clean-only BN (Safe forward with frozen BN stats)
- T9: Independent domain calibration for SWD

These are all code-correct but do not change results on Houston/S->H (src_val saturated -> cycles discarded) or HyRANK (T-CST still 0/6 reject).

## What was NOT changed
- split = v8.0.7 mixed-mode (v3 block split rejected: helps S->H +16 but destroys HyRANK -25)
- ckpt_mode=A (B and C tested but unstable)
- projection_norm=batch (LayerNorm tested and rejected: HyRANK -9.78, S->H -11.40)
- No target GT, no pseudo-labels, no EMA, single model

## Known Limitations
- Houston: src_val saturates at 100, cycles discarded
- HyRANK: class2/class5 have <30 source pixels (physical limit)
- S->H: 3-class mode collapse, std 10.11

## Reproduce
python -u scripts/train.py --config configs_FINAL_v809/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v809/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v809/shanghai_final.yaml
"""
(ROOT / 'README_V809.md').write_text(readme, encoding='utf-8')
print("✅ README_V809.md written")

os.chdir(ROOT)
if (ROOT / 'TCST_v8.0.9_Strict_Final.tar.gz').exists():
    (ROOT / 'TCST_v8.0.9_Strict_Final.tar.gz').unlink()

subprocess.run([
    'tar', '-czf', 'TCST_v8.0.9_Strict_Final.tar.gz',
    '--exclude=results*', '--exclude=*.pyc', '--exclude=__pycache__',
    '--exclude=*.log', '--exclude=.git',
    '--exclude=core_backup*', '--exclude=configs_backup*',
    'core/', 'configs/', 'configs_FINAL_v809/', 'scripts/', 'tests/',
    'docs/', 'requirements.txt', 'README_V809.md',
], check=True)
size = (ROOT / 'TCST_v8.0.9_Strict_Final.tar.gz').stat().st_size
print(f"\n✅ 打包完成: TCST_v8.0.9_Strict_Final.tar.gz ({size/1024:.1f} KB)")

with tarfile.open(ROOT / 'TCST_v8.0.9_Strict_Final.tar.gz') as tf:
    names = tf.getnames()
    print(f"   .py: {len([n for n in names if n.endswith('.py')])}")
    print(f"   .yaml: {len([n for n in names if n.endswith('.yaml')])}")
