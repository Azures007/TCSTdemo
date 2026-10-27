#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_ablation/innovation_s1237

echo "=================================================="
echo "  I1/I2/I3 消融（3 变体 × 3 数据集 × seed=1237）"
echo "  开始: $(date)"
echo "=================================================="

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEED = 1237
variants = {
    'I1_no_qp':        {'tcst': {'use_qp': False}},
    'I2_no_risk':      {'tcst': {'use_risk_search': False}},
    'I3_hard_replace': {'training': {'use_safe_training': False}},
}
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for vname, changes in variants.items():
        c = copy.deepcopy(base)
        for sec, kv in changes.items():
            for k, v in kv.items():
                c[sec][k] = v
        c['output']['seed'] = SEED
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_ablation/innovation_s1237/{vname}/{ds}'
        yaml.dump(c, open(f'configs/innov_{vname}_{ds}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    mkdir -p "$outdir"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "\[qp\]|\[risk\]|ACCEPT|REJECT|restored|OA |AA " | tail -10
    echo "  DONE: $tag"
}

for vname in I1_no_qp I2_no_risk I3_hard_replace; do
    for ds in houston hyrank shanghai; do
        run_one "configs/innov_${vname}_${ds}.yaml" "${vname}_${ds}" "results_ablation/innovation_s1237/${vname}"
    done
done

echo ""
echo "=================================================="
echo "  I1/I2/I3 消融汇总（seed=1237）"
echo "=================================================="
python - << 'PYEOF' | tee results_ablation/innovation_s1237/SUMMARY.txt
import json, os

# 基线 R2（v8.1.0 统一 ckpt_mode=A, use_warmup_augmentation=False, seed=1237）
BASELINE = {'houston': 75.99, 'hyrank': 51.00, 'shanghai': 74.75}

print(f"{'variant':<22} {'Houston':>12} {'HyRANK':>12} {'S→H':>12}")
print('-' * 68)
print(f"  {'R2 (full T-CST)':<20} "
      f"{BASELINE['houston']:>10.2f}  {BASELINE['hyrank']:>10.2f}  {BASELINE['shanghai']:>10.2f}")
print('-' * 68)

for vname in ['I1_no_qp', 'I2_no_risk', 'I3_hard_replace']:
    row = {}
    for ds in ['houston', 'hyrank', 'shanghai']:
        p = f'results_ablation/innovation_s1237/{vname}/{ds}/final_metrics.json'
        if os.path.exists(p):
            m = json.load(open(p))['target_metrics']
            row[ds] = m['OA']
    if row:
        print(f"  {vname:<20} "
              f"{row.get('houston', 0):>10.2f}  {row.get('hyrank', 0):>10.2f}  {row.get('shanghai', 0):>10.2f}")

print()
print("  Δ vs R2 (seed=1237):")
for vname in ['I1_no_qp', 'I2_no_risk', 'I3_hard_replace']:
    deltas = []
    for ds in ['houston', 'hyrank', 'shanghai']:
        p = f'results_ablation/innovation_s1237/{vname}/{ds}/final_metrics.json'
        if os.path.exists(p):
            m = json.load(open(p))['target_metrics']
            deltas.append((ds, m['OA'] - BASELINE[ds]))
    if deltas:
        print(f"  {vname:<20} " + "  ".join(f"{ds}: {d:+.2f}" for ds, d in deltas))

print()
print("  ACCEPT 统计（重点：I2 应该全 ACCEPT，I1 可能 ACCEPT 更多）:")
for vname in ['I1_no_qp', 'I2_no_risk', 'I3_hard_replace']:
    for ds in ['houston', 'hyrank', 'shanghai']:
        p = f'results_ablation/innovation_s1237/{vname}/{ds}/training_log.jsonl'
        if not os.path.exists(p): continue
        n_acc, n_total = 0, 0
        for line in open(p):
            d = json.loads(line)
            if d.get('stage') != 'cycle': continue
            n_total += 1
            tr = d.get('transport', {})
            if tr.get('accepted_candidate') or d.get('skipped_risk_search'):
                n_acc += 1
        print(f"    {vname}/{ds}: {n_acc}/{n_total} ACCEPT")

# 逐类准确率明细（关键：看 I1/I2 让哪些类恶化）
print()
print("  逐类准确率（只列有 final_metrics 的）:")
for vname in ['I1_no_qp', 'I2_no_risk', 'I3_hard_replace']:
    for ds in ['houston', 'hyrank', 'shanghai']:
        p = f'results_ablation/innovation_s1237/{vname}/{ds}/final_metrics.json'
        if not os.path.exists(p): continue
        m = json.load(open(p))['target_metrics']
        print(f"    {vname}/{ds}: {[round(x,1) for x in m['class_accuracy']]}")
PYEOF

echo ""
echo "=================================================="
echo "  ✅ 完成  $(date)"
echo "=================================================="
