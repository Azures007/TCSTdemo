#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_stabilize/sh

python - << 'PYEOF'
import yaml, copy
base = yaml.safe_load(open('configs/sh_S5_ls005.yaml'))

# V1: 关 AMP（fp16 溢出可能是坍塌元凶）
c = copy.deepcopy(base)
c['training']['amp'] = False
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_stabilize/sh/V1_noamp'
yaml.dump(c, open('configs/sh_V1_noamp.yaml','w'), sort_keys=False)

# V2: 降 lr 到 1e-4（更慢收敛，避免早塌）
c = copy.deepcopy(base)
c['training']['lr'] = 1e-4
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_stabilize/sh/V2_lr1e4'
yaml.dump(c, open('configs/sh_V2_lr1e4.yaml','w'), sort_keys=False)

# V3: 加大 batch + 关 AMP + 加 warmup epochs
c = copy.deepcopy(base)
c['training']['amp'] = False
c['training']['batch_size'] = 128
c['training']['epochs_warmup'] = 30
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_stabilize/sh/V3_bigbatch'
yaml.dump(c, open('configs/sh_V3_bigbatch.yaml','w'), sort_keys=False)

print("OK: 3 个稳定化配置")
PYEOF

# 每个变体跑 3 个 seed
for v in V1_noamp V2_lr1e4 V3_bigbatch; do
    for seed in 1237 42 2024; do
        python - << PYEOF
import yaml
c = yaml.safe_load(open('configs/sh_$v.yaml'))
c['output']['seed'] = $seed
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_stabilize/sh/${v}_seed$seed'
yaml.dump(c, open('configs/sh_${v}_s$seed.yaml','w'), sort_keys=False)
PYEOF
        echo ""
        echo "=== RUN: $v / seed=$seed ==="
        python -u scripts/train.py --config "configs/sh_${v}_s$seed.yaml" 2>&1 | \
            stdbuf -oL grep --line-buffered -E "OA |AA |class[0-9]+ " | tail -6
    done
done

echo ""
echo "=== 稳定化对比（目标：std < 5） ==="
python - << 'PYEOF'
import json, os, statistics
VARIANTS = ['V1_noamp', 'V2_lr1e4', 'V3_bigbatch']
SEEDS = [1237, 42, 2024]
for v in VARIANTS:
    oas = []
    print(f"\n[{v}]")
    for seed in SEEDS:
        p = f'results_stabilize/sh/{v}_seed{seed}/final_metrics.json'
        if not os.path.exists(p):
            print(f"  seed={seed}: (missing)"); continue
        m = json.load(open(p))['target_metrics']
        oas.append(m['OA'])
        print(f"  seed={seed}: OA={m['OA']:.2f}  AA={m['AA']:.2f}  per-class={[round(x,1) for x in m['class_accuracy']]}")
    if len(oas) >= 2:
        print(f"  → mean={statistics.mean(oas):.2f}  std={statistics.stdev(oas):.2f}  min={min(oas):.2f}  max={max(oas):.2f}")
PYEOF
