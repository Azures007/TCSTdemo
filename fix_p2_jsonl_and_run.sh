#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p2jsonl_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# ============ 1) 修 jsonl 写入 ============
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 找 jsonl 写入块，加 tgt_pred_dist
old_jsonl = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "phi_star_full": _a12_phi_full,
                                "d_star_full": _a12_d_full,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
new_jsonl = '''            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "phi_star_full": _a12_phi_full,
                                "d_star_full": _a12_d_full,
                                "tgt_pred_dist": _tgt_pred_dist,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\\n")'''
assert old_jsonl in s, "P2 jsonl 写入块未找到"
s = s.replace(old_jsonl, new_jsonl, 1)
p.write_text(s)
ast.parse(s)
print("✅ jsonl 写入加 tgt_pred_dist")
PYEOF

# ============ 2) 快速冒烟确认 ============
echo ""
echo "===== 冒烟: Houston 1 epoch 确认 jsonl 有 tgt_pred_dist ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p2jsonl_smoke/houston'
yaml.dump(c, open('configs/p2jsonl_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p2jsonl_smoke/houston
python -u scripts/train.py --config configs/p2jsonl_smoke.yaml 2>&1 | tail -3

echo ""
echo "--- jsonl 里的 tgt_pred_dist ---"
python - << 'PYEOF'
import json
with open('results_p2jsonl_smoke/houston/training_log.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('stage') == 'cycle':
            print(f"  cycle={d.get('cycle')}")
            print(f"  tgt_pred_dist: {d.get('tgt_pred_dist')}")
            break
PYEOF

# ============ 3) 跑完整 3 数据集 × 3 seed P2 诊断 ============
echo ""
echo "=================================================="
echo "  P2 完整诊断：3 数据集 × 3 seed"
echo "  开始: $(date)"
echo "=================================================="
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_p2_full/{houston,hyrank,shanghai}

python - << 'PYEOF'
import yaml, copy
BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_V2_lr1e4.yaml',
}
SEEDS = [1237, 42, 2024]
for ds, bp in BASES.items():
    base = yaml.safe_load(open(bp))
    for seed in SEEDS:
        c = copy.deepcopy(base)
        c['training']['ckpt_mode'] = 'A'
        c['output']['seed'] = seed
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_p2_full/{ds}/s{seed}'
        yaml.dump(c, open(f'configs/p2_{ds}_s{seed}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "  RUN: $tag   $(date +%H:%M:%S)"
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E "warm [0-9]+/|cyc[0-9]+ ep|OA |AA " | tail -6
}

for ds in houston hyrank shanghai; do
    for seed in 1237 42 2024; do
        run_one "configs/p2_${ds}_s${seed}.yaml" "${ds}_s${seed}" "results_p2_full/$ds"
    done
done

# ============ 4) 汇总 P2 诊断 ============
echo ""
echo "=================================================="
echo "  P2 诊断汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_p2_full/SUMMARY.txt
import json, os, statistics
SEEDS = [1237, 42, 2024]

for ds in ['houston', 'hyrank', 'shanghai']:
    print(f"\n{'='*90}")
    print(f"[{ds.upper()}]  P2 诊断")
    print(f"{'='*90}")

    # 先检查最终 OA
    oas = []
    for seed in SEEDS:
        p = f'results_p2_full/{ds}/s{seed}/final_metrics.json'
        if os.path.exists(p):
            m = json.load(open(p))['target_metrics']
            oas.append(m['OA'])
    if oas:
        print(f"  Final OA: {statistics.mean(oas):.2f}±{statistics.stdev(oas) if len(oas) > 1 else 0:.2f}")

    # 从 jsonl 读 per_class_margin 和 tgt_pred_dist
    for seed in SEEDS:
        p = f'results_p2_full/{ds}/s{seed}/training_log.jsonl'
        if not os.path.exists(p): continue
        print(f"\n  [seed={seed}]")
        cycles_seen = 0
        for line in open(p):
            d = json.loads(line)
            if d.get('stage') != 'cycle': continue
            cycles_seen += 1
            tr = d.get('transport', {})
            acc = tr.get('accepted_candidate')
            tgt = d.get('tgt_pred_dist')
            if acc is None:
                print(f"    cycle {d['cycle']}: REJECT (no accepted candidate)")
            else:
                pcm = acc.get('per_class_margin', {})
                if pcm:
                    # 找出恶化的类
                    neg_classes = [c for c, s_ in pcm.items() if s_.get('mean_delta', 0) < -0.1]
                    pos_classes = [c for c, s_ in pcm.items() if s_.get('mean_delta', 0) > 0.1]
                    print(f"    cycle {d['cycle']}: ACCEPT eta={acc.get('eta'):.2f}  "
                          f"margin_worse={neg_classes}  margin_better={pos_classes}")
                if tgt:
                    dist = tgt.get('pred_class_dist', [])
                    ent = tgt.get('pred_entropy', 0)
                    print(f"      tgt_pred_dist={[round(x,3) for x in dist]}  entropy={ent:.3f}")
            if cycles_seen >= 3:  # 只打印前 3 个 cycle
                print(f"      ... (共 {sum(1 for _ in open(p) if json.loads(_).get('stage') == 'cycle')} 个 cycle)")
                break
PYEOF

echo ""
echo "=================================================="
echo "  ✅ P2 完整诊断完成  $(date)"
echo "=================================================="
