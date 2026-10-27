#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p results_stage2/{houston,hyrank,shanghai}

echo "=================================================="
echo "  T-CST 第二轮：12 变体（3 数据集 × 4）"
echo "  开始时间: $(date)"
echo "=================================================="

# ============ 1) 生成 12 个配置 ============
echo ""
echo "=== 生成配置 ==="
python - << 'PYEOF'
import yaml, copy

# Houston
base = yaml.safe_load(open('configs_FINAL_v805/houston_final.yaml'))
variants = {
    'H5_ff_true_tau008': {'training': {'force_final_epoch': True}, 'tcst': {'tau_swd_rel': 0.008}},
    'H6_ls05':           {'training': {'label_smoothing': 0.05}},
    'H7_warm20':         {'training': {'epochs_warmup': 20}},
    'H8_tau006':         {'tcst': {'tau_swd_rel': 0.006}},
}
for name, ch in variants.items():
    c = copy.deepcopy(base)
    for sec, kv in ch.items():
        for k, v in kv.items(): c[sec][k] = v
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_stage2/houston/{name}'
    yaml.dump(c, open(f'configs/hou_{name}.yaml','w'), sort_keys=False)

# HyRANK (基于第一层最优 Y2_warm40)
base = yaml.safe_load(open('configs/hy_Y2_warm40.yaml'))
variants = {
    'Y5_warm50':  {'training': {'epochs_warmup': 50}},
    'Y6_lr2e4':   {'training': {'lr': 2e-4}},
    'Y7_p13':     {'training': {'patch_size': 13}},
    'Y8_cb03':    {'training': {'ce_class_balance_power': 0.3}},
}
for name, ch in variants.items():
    c = copy.deepcopy(base)
    for sec, kv in ch.items():
        for k, v in kv.items(): c[sec][k] = v
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_stage2/hyrank/{name}'
    yaml.dump(c, open(f'configs/hy_{name}.yaml','w'), sort_keys=False)

# S→H (基于第一层最优 S2_ls001)
base = yaml.safe_load(open('configs/sh_S2_ls001.yaml'))
variants = {
    'S5_ls005':  {'training': {'label_smoothing': 0.005}},
    'S6_ls015':  {'training': {'label_smoothing': 0.015}},
    'S7_bs16':   {'training': {'batch_size': 16}},
    'S8_warm30': {'training': {'epochs_warmup': 30}},
}
for name, ch in variants.items():
    c = copy.deepcopy(base)
    for sec, kv in ch.items():
        for k, v in kv.items(): c[sec][k] = v
    c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_stage2/shanghai/{name}'
    yaml.dump(c, open(f'configs/sh_{name}.yaml','w'), sort_keys=False)

print("OK: 12 个配置已生成（持久化在 ~/autodl-tmp/TCSTdemo/configs/）")
PYEOF

# ============ 2) 单变体运行函数 ============
run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL -eL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|cyc[0-9]+ ep|\[risk\].*(ACCEPT|REJECT)|\[Phase|restored|OA |AA |Kappa|class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

# ============ 3) 跑 12 个变体 ============
for v in H5_ff_true_tau008 H6_ls05 H7_warm20 H8_tau006; do
    run_one "configs/hou_$v.yaml" "$v" "results_stage2/houston"
done

for v in Y5_warm50 Y6_lr2e4 Y7_p13 Y8_cb03; do
    run_one "configs/hy_$v.yaml" "$v" "results_stage2/hyrank"
done

for v in S5_ls005 S6_ls015 S7_bs16 S8_warm30; do
    run_one "configs/sh_$v.yaml" "$v" "results_stage2/shanghai"
done

# ============ 4) 汇总（写入持久化文件）============
echo ""
echo "=================================================="
echo "  第二轮汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_stage2/SUMMARY.txt
import json, os
BASE = {
    'houston':  (79.37, 78.55),
    'hyrank':   (62.25, 54.14),
    'shanghai': (77.20, 72.23),
}
RUNS = {
    'houston':  ['H5_ff_true_tau008','H6_ls05','H7_warm20','H8_tau006'],
    'hyrank':   ['Y5_warm50','Y6_lr2e4','Y7_p13','Y8_cb03'],
    'shanghai': ['S5_ls005','S6_ls015','S7_bs16','S8_warm30'],
}
for ds, variants in RUNS.items():
    oa0, aa0 = BASE[ds]
    print(f"\n[{ds}]  第一层最优: OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'variant':<22} {'OA':>8} {'ΔOA':>8} {'AA':>8} {'ΔAA':>8}")
    print('-'*58)
    for v in variants:
        p = f'results_stage2/{ds}/{v}/final_metrics.json'
        if not os.path.exists(p): print(f"{v:<22} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        print(f"{v:<22} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}")
PYEOF

# ============ 5) 同步 + 关机 ============
echo ""
echo "=================================================="
echo "  所有变体完成，开始同步磁盘并关机"
echo "  结束时间: $(date)"
echo "  结果目录: ~/autodl-tmp/TCSTdemo/results_stage2/"
echo "  汇总文件: ~/autodl-tmp/TCSTdemo/results_stage2/SUMMARY.txt"
echo "=================================================="

# 强制 flush 到磁盘（tee 的缓冲 + page cache）
sync
sync
sync

echo "磁盘已同步，5 秒后关机..."
sleep 5

# autodl 容器里 shutdown 可能失败，fallback 到 poweroff
shutdown -h now 2>/dev/null || poweroff 2>/dev/null || {
    echo "⚠️ 自动关机失败（容器环境限制）。请手动在 autodl 控制台关机。"
    echo "   结果已全部保存到 ~/autodl-tmp/TCSTdemo/results_stage2/"
    exit 1
}
