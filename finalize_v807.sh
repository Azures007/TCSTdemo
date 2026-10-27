#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

echo "=================================================="
echo "  T-CST v8.0.7 最终收尾：诊断 → 条件重测 → 封版"
echo "  开始: $(date)"
echo "=================================================="

# ============ Step 1: 诊断 ============
echo ""
echo "[1/4] 诊断当前状态"

HAS_CC=0
if grep -q "ClassSmoothSpectralTransform" core/trainer_v8.py 2>/dev/null; then
    HAS_CC=1
    echo "  ✅ 类条件代码在 core/trainer_v8.py"
else
    echo "  ⚠️ 类条件代码不在（可能被某次回滚覆盖）"
fi

echo ""
echo "--- HyRANK warm src_val 轨迹（最后 5 个 epoch）---"
grep "warm.*src_val" results_ckpt_seed1237/hyrank_A/log_hyrank_A.log 2>/dev/null | tail -5 || echo "  无日志"

echo ""
echo "--- Houston warm src_val 轨迹（最后 5 个 epoch）---"
grep "warm.*src_val" results_ckpt_seed1237/houston_A/log_houston_A.log 2>/dev/null | tail -5 || echo "  无日志"

echo ""
echo "--- S→H warm src_val 轨迹（最后 5 个 epoch）---"
grep "warm.*src_val" results_ckpt_seed1237/shanghai_A/log_shanghai_A.log 2>/dev/null | tail -5 || echo "  无日志"

# 提取最后 src_val
HYRANK_LAST=$(grep "warm.*src_val" results_ckpt_seed1237/hyrank_A/log_hyrank_A.log 2>/dev/null | tail -1 | sed 's/.*src_val=\([0-9.]*\).*/\1/' || echo "100.0")
echo ""
echo "HyRANK 最后 warm src_val = ${HYRANK_LAST:-100.0}"

# ============ Step 2: 判断 ============
echo ""
echo "[2/4] 判断是否重测类条件 T-CST"

SKIP_RETEST=1
if [ "$HAS_CC" = "0" ]; then
    echo "  → 跳过：类条件代码不在"
elif python -c "import sys; v=float('${HYRANK_LAST:-100.0}'); sys.exit(0 if v > 97 else 1)" 2>/dev/null; then
    echo "  → 跳过：HyRANK src_val=${HYRANK_LAST} > 97，饱和"
    echo "    ckpt_mode=A 下 cycles checkpoint 永远被丢弃，类条件重测会得到逐位相同的结果"
else
    echo "  → 执行重测：HyRANK src_val=${HYRANK_LAST} 未饱和"
    SKIP_RETEST=0
fi

# ============ Step 3: 条件重测 ============
if [ "$SKIP_RETEST" = "0" ]; then
    echo ""
    echo "[3/4] 重测类条件 T-CST（3 数据集 × seed=1237）"
    mkdir -p results_retest_cc

    python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    c = copy.deepcopy(base)
    c['tcst']['use_class_conditional'] = True
    c['tcst']['rho_a'] = 0.10
    c['tcst']['rho_b'] = 0.10
    c['training']['ckpt_mode'] = 'A'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_retest_cc/{ds}'
    yaml.dump(c, open(f'configs/retest_cc_{ds}.yaml','w'), sort_keys=False)
print("✅ 3 个类条件配置已生成")
PYEOF

    for ds in houston hyrank shanghai; do
        echo ""
        echo "=== RUN: $ds 类条件 ==="
        python -u scripts/train.py --config "configs/retest_cc_${ds}.yaml" 2>&1 | \
            tee "results_retest_cc/log_${ds}.log" | \
            grep -E "\[transform\]|\[qp\]|OA |AA |class[0-9]+ " | tail -16
    done

    echo ""
    echo "=== 类条件 vs 基线 ==="
    python - << 'PYEOF' | tee results_retest_cc/SUMMARY.txt
import json, os
BASE = {'houston': 76.12, 'hyrank': 48.95, 'shanghai': 59.19}
for ds in ['houston','hyrank','shanghai']:
    p = f'results_retest_cc/{ds}/final_metrics.json'
    print(f"\n[{ds}] baseline={BASE[ds]:.2f}")
    if not os.path.exists(p):
        print("  (missing)"); continue
    m = json.load(open(p))['target_metrics']
    delta = m['OA'] - BASE[ds]
    print(f"  类条件: OA={m['OA']:.2f} ({delta:+.2f})  AA={m['AA']:.2f}")
    print(f"  per-class={[round(x,1) for x in m['class_accuracy']]}")
    if delta > 2:
        print(f"  → ✅ 涨了 {delta:.2f}，值得跑 3-seed 确认")
    elif delta < -2:
        print(f"  → ❌ 跌了 {abs(delta):.2f}，类条件负迁移")
    else:
        print(f"  → ⚠️ 变化 <2，类条件无显著影响")
PYEOF

    echo ""
    echo "  ⚠️ 如果上面显示任一数据集涨 >2，请手动跑 3-seed 确认："
    echo "     bash ~/autodl-tmp/TCSTdemo/stage2b_verify_3seed.sh  # 需要改造为类条件版本"
else
    echo ""
    echo "[3/4] 跳过重测（见上面的原因）"
fi

# ============ Step 4: 封版 ============
echo ""
echo "[4/4] 封版 v8.0.7"

python - << 'PYEOF'
from pathlib import Path
import yaml, subprocess, os

ROOT = Path('/root/autodl-tmp/TCSTdemo')
FINAL = ROOT / 'configs_FINAL_v807'
FINAL.mkdir(exist_ok=True)

BASES = {
    'houston':  ('configs/hou_H6_ls05.yaml',    76.12, 74.66),
    'hyrank':   ('configs/hy_Y2_warm40.yaml',   48.95, 50.89),
    'shanghai': ('configs/sh_V2_lr1e4.yaml',    59.19, 60.05),
}
for ds, (base_path, oa, aa) in BASES.items():
    c = yaml.safe_load(open(ROOT / base_path))
    c['training']['ckpt_mode'] = 'A'
    c['output']['seed'] = 1237
    c['output']['save_dir'] = str(ROOT / f'results_FINAL_v807/{ds}')
    yaml.dump(c, open(FINAL / f'{ds}_final.yaml', 'w', encoding='utf-8'),
              sort_keys=False, allow_unicode=True)
    print(f"✅ {ds}: ckpt_mode=A, {oa:.2f}±{aa:.2f}")

readme = """# T-CST v8.0.7 (Strict Final, unified ckpt_mode=A, 3-seed mean±std)

## Final Results (seeds: 1237, 42, 2024)
| Dataset | OA (mean±std) | AA (mean±std) | Best single seed |
|---|---|---|---|
| Houston13->18 | 76.12 ± 1.12 | 74.66 ± 4.34 | 77.30 (seed=2024) |
| Dioni->Loukia | 48.95 ± 10.95 | 50.89 ± 3.49 | 58.06 (seed=2024) |
| Shanghai->Hangzhou | 59.19 ± 20.61 | 60.05 ± 8.77 | 73.69 (seed=1237) |

## Checkpoint selection: unified ckpt_mode=A
All three datasets use ckpt_mode=A (source val accuracy).
B (acc+entropy): Houston +1.55 but S->H -15.09 (rejected).
C (class-balanced NLL): HyRANK seed=1237 +7.91 but 3-seed mean 48.88±15.78 (rejected).

## Bugs Fixed (vs v8.0.6)
1. A1: Spatial isolation buffer with val<5 fallback
2. A2: Scheduler re-bound after optimizer reset
3. A5: _make_model reads architecture from cfg
4. A6: Spectral TV thresholds tightened to 0.36/0.18
5. A7: Fusion gate init preserved after _init()
6. A8: Clean Loss uses raw source (not augmented)
7. A10: QP accepts optimal_inaccurate; residual uses xi.value
8. A11: MMD bandwidth uses joint source+target median
9. A12: training_log.jsonl records phi_star_stats/d_star_stats

## Known Limitations
- Houston: source val saturates at ~95%; T-CST cycles discarded by best_src_val
- HyRANK: class2/class5 have <30 source pixels (train 9-10 not 150)
- S->H: 3-class mode collapse; std 20.61 across seeds

## Reproduce
python -u scripts/train.py --config configs_FINAL_v807/houston_final.yaml
python -u scripts/train.py --config configs_FINAL_v807/hyrank_final.yaml
python -u scripts/train.py --config configs_FINAL_v807/shanghai_final.yaml
"""
(ROOT / 'README_V807.md').write_text(readme, encoding='utf-8')
print("✅ README_V807.md written")

os.chdir(ROOT)
if (ROOT / 'TCST_v8.0.7_Strict_Final.tar.gz').exists():
    (ROOT / 'TCST_v8.0.7_Strict_Final.tar.gz').unlink()

subprocess.run([
    'tar', '-czf', 'TCST_v8.0.7_Strict_Final.tar.gz',
    '--exclude=results*', '--exclude=*.pyc', '--exclude=__pycache__',
    '--exclude=*.log', '--exclude=.git',
    '--exclude=core_backup*', '--exclude=configs_backup*',
    'core/', 'configs/', 'configs_FINAL_v807/', 'scripts/', 'tests/',
    'docs/', 'requirements.txt', 'README_V807.md',
], check=True)
print(f"\n✅ 打包完成 ({os.path.getsize('TCST_v8.0.7_Strict_Final.tar.gz')/1024:.1f} KB)")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ T-CST v8.0.7 封版完成"
echo "  结束: $(date)"
echo "  最终数字: 76.12±1.12 / 48.95±10.95 / 59.19±20.61"
echo "  包: ~/autodl-tmp/TCSTdemo/TCST_v8.0.7_Strict_Final.tar.gz"
echo "=================================================="
