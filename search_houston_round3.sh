#!/bin/bash
set -e
cd /root/autodl-tmp/TCSTdemo

run_one () {
  local tag=$1
  local cfg="configs_reg/houston_${tag}.yaml"
  local out="results_reg/houston_${tag}"
  local log="results_reg/houston_${tag}.log"
  echo ""
  echo "=================================================="
  echo " RUN: ${tag}   $(date +%H:%M:%S)"
  echo "=================================================="
  python -u scripts/train.py --config "$cfg" 2>&1 | tee "$log" | \
    grep --line-buffered -E "restored best warmup|selected cycle|TARGET RESULT|OA |AA |Kappa|Traceback|Error"
}

# ===== 第 1 组：dropout 0.30（继续压 base）=====
sed -i 's/ResidualBlock2D(64, 96, dropout=0.20)/ResidualBlock2D(64, 96, dropout=0.30)/' core/model.py
sed -i 's/ResidualBlock2D(96, 128, dropout=0.20)/ResidualBlock2D(96, 128, dropout=0.30)/' core/model.py
sed -i 's/ResidualBlock2D(128, 128, dropout=0.20)/ResidualBlock2D(128, 128, dropout=0.30)/' core/model.py
grep -n "ResidualBlock2D(64, 96\|ResidualBlock2D(96, 128\|ResidualBlock2D(128, 128" core/model.py

python - <<'PY'
import yaml
c = yaml.safe_load(open('configs_FINAL/houston_final.yaml'))
c['training']['ckpt_mode'] = 'D'
c['output']['seed'] = 1237
c['output']['save_dir'] = 'results_reg/houston_dropout0.30'
yaml.dump(c, open('configs_reg/houston_dropout0.30.yaml','w'), sort_keys=False)
PY
run_one dropout0.30

# ===== 第 2/3 组：dropout 0.20 + ckpt_d_tolerance 4.0 / 6.0 =====
# 回到 dropout 0.20
sed -i 's/ResidualBlock2D(64, 96, dropout=0.30)/ResidualBlock2D(64, 96, dropout=0.20)/' core/model.py
sed -i 's/ResidualBlock2D(96, 128, dropout=0.30)/ResidualBlock2D(96, 128, dropout=0.20)/' core/model.py
sed -i 's/ResidualBlock2D(128, 128, dropout=0.30)/ResidualBlock2D(128, 128, dropout=0.20)/' core/model.py

for tol in 4.0 6.0; do
  tag="dropout0.20_tol${tol}"
  python - <<PY
import yaml
c = yaml.safe_load(open('configs_FINAL/houston_final.yaml'))
c['training']['ckpt_mode'] = 'D'
c['training']['ckpt_d_tolerance'] = float('${tol}')
c['output']['seed'] = 1237
c['output']['save_dir'] = 'results_reg/houston_${tag}'
yaml.dump(c, open('configs_reg/houston_${tag}.yaml','w'), sort_keys=False)
PY
  run_one "$tag"
done

# ===== 第 4 组：dropout 0.20 + aug 加强 + warm 20 =====
tag="dropout0.20_aug0.15_w20"
python - <<'PY'
import yaml
c = yaml.safe_load(open('configs_FINAL/houston_final.yaml'))
c['training']['ckpt_mode'] = 'D'
c['training']['aug_scale_range'] = 0.15
c['training']['aug_noise_std'] = 0.03
c['training']['epochs_warmup'] = 20
c['output']['seed'] = 1237
c['output']['save_dir'] = 'results_reg/houston_dropout0.20_aug0.15_w20'
yaml.dump(c, open('configs_reg/houston_dropout0.20_aug0.15_w20.yaml','w'), sort_keys=False)
PY
run_one "$tag"

echo ""
echo "=================================================="
echo " 汇总"
echo "=================================================="
python - <<'PY'
import json, glob, re, os
rows = []
for p in sorted(glob.glob('results_reg/houston_*/final_metrics.json')):
    tag = os.path.basename(os.path.dirname(p))
    try:
        m = json.load(open(p))['target_metrics']
    except Exception:
        continue
    logfile = os.path.join('results_reg', tag + '.log')
    base = None
    if os.path.exists(logfile):
        txt = open(logfile, errors='ignore').read()
        mm = re.findall(r'restored best warmup \(src_val=([0-9.]+)\)', txt)
        if mm: base = float(mm[-1])
    rows.append((m['OA'], m['AA'], m['Kappa'], base, tag))
rows.sort(reverse=True)
print(f"{'OA':>6} {'AA':>6} {'Kappa':>7} {'base':>7}  tag")
print('-'*70)
for oa,aa,k,base,tag in rows:
    bs = f"{base:6.2f}" if base is not None else "  ?  "
    print(f"{oa:6.2f} {aa:6.2f} {k:7.4f} {bs}  {tag}")
PY
