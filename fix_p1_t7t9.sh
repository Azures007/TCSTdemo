#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_p1_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

def patch_independent(name, path, old, new, verify_str):
    """每个补丁独立读写，不共享变量。"""
    p = Path(path)
    s = p.read_text()
    if verify_str in s:
        print(f"  ⏭️ {name}: 已存在")
        return True
    if old not in s:
        print(f"  ❌ {name}: 未匹配")
        return False
    s2 = s.replace(old, new, 1)
    try:
        ast.parse(s2)
    except SyntaxError as e:
        print(f"  ❌ {name}: 语法错误 {e}")
        return False
    p.write_text(s2)
    print(f"  ✅ {name}")
    return True

# ============ T7: MMD target 每 cycle 重新抽样 ============
print("===== T7: MMD target 重抽 =====")
patch_independent(
    "T7 MMD target 重抽",
    "core/trainer_v8.py",
    '''        tgt_desc = spectral_descriptor(
            torch.from_numpy(tgt_transport_p[:128]).to(device))''',
    '''        # T7: 每 cycle 从 transport pool 随机抽 128，而非固定前 128
        _tgt_rng = np.random.default_rng(cfg["output"]["seed"] + 500 + cycle)
        _n_tgt = min(128, len(tgt_transport_p))
        _tgt_idx = _tgt_rng.choice(len(tgt_transport_p), _n_tgt, replace=False)
        tgt_desc = spectral_descriptor(
            torch.from_numpy(tgt_transport_p[_tgt_idx]).to(device))''',
    '_tgt_rng.choice(len(tgt_transport_p)'
)

# ============ T8: Clean-only BN（Safe 分支用 eval 模式更新 BN）============
print("\n===== T8: Clean-only BN =====")
# 8a) 加 context manager（在文件顶部 import 之后）
patch_independent(
    "T8a context manager",
    "core/trainer_v8.py",
    "from __future__ import annotations",
    '''from __future__ import annotations

from contextlib import contextmanager as _cm

@_cm
def _freeze_bn(model):
    """T8: Safe 分支 forward 时冻结 BN 统计（eval 模式），梯度仍通过 BN 仿射参数。"""
    bn_modules = [m for m in model.modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    states = [m.training for m in bn_modules]
    try:
        for m in bn_modules:
            m.eval()
        yield
    finally:
        for m, st in zip(bn_modules, states):
            m.train(st)
''',
    'def _freeze_bn(model):'
)

# 8b) _train_step 的 Safe 分支用 _freeze_bn
patch_independent(
    "T8b Safe 分支冻结 BN",
    "core/trainer_v8.py",
    '''                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)
                loss_safe = _ce(model(xb_safe).logits, yb)''',
    '''                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)
                # T8: Safe 分支 forward 时冻结 BN 统计（避免 Clean/Safe 双更新）
                with _freeze_bn(model):
                    loss_safe = _ce(model(xb_safe).logits, yb)''',
    'with _freeze_bn(model):'
)

# ============ T9: SWD 独立 Domain Calibration ============
print("\n===== T9: SWD 独立 Domain Calibration =====")
# 9a) backtracking_search 签名加 domain_cal_patches
patch_independent(
    "T9a bs 签名加 domain_cal_patches",
    "core/risk_controller.py",
    '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05):''',
    '''                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05,
                        domain_cal_patches=None):''',
    'domain_cal_patches=None):'
)

# 9b) SWD 源域用 domain_cal_patches（如果提供）
patch_independent(
    "T9b SWD 用 domain cal",
    "core/risk_controller.py",
    '''        src_desc0 = spectral_descriptor(torch.from_numpy(source_cal_patches).to(device))
        tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches).to(device))''',
    '''        # T9: SWD 用独立的 domain calibration（如果提供），而非 BA-CS 边界样本
        _src_swd = domain_cal_patches if domain_cal_patches is not None else source_cal_patches
        src_desc0 = spectral_descriptor(torch.from_numpy(_src_swd).to(device))
        tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches).to(device))''',
    '_src_swd = domain_cal_patches'
)

patch_independent(
    "T9c SWD after 用 domain cal",
    "core/risk_controller.py",
    '''            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(source_cal_patches).to(device), phi,
                          labels=torch.from_numpy(source_cal_labels).long().to(device)))''',
    '''            _src_swd1 = domain_cal_patches if domain_cal_patches is not None else source_cal_patches
            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(_src_swd1).to(device), phi,
                          labels=None))''',
    '_src_swd1 = domain_cal_patches'
)

# 9d) run_tcst_v8 准备 src_domain_cal_p 并传入
patch_independent(
    "T9d run_tcst_v8 准备 domain cal",
    "core/trainer_v8.py",
    '''        ba_cs_ca_p, ba_cs_ca_y, ba_cs_counts = _sample_boundary_calibration(''',
    '''        # T9: 独立 domain calibration（随机抽源域训练样本，和 BA-CS 边界样本分开）
        _dom_rng = np.random.default_rng(cfg["output"]["seed"] + 777)
        _dom_idx = _dom_rng.choice(len(src_tr_p), min(128, len(src_tr_p)), replace=False)
        src_domain_cal_p = src_tr_p[_dom_idx]
        ba_cs_ca_p, ba_cs_ca_y, ba_cs_counts = _sample_boundary_calibration(''',
    'src_domain_cal_p = src_tr_p[_dom_idx]'
)

patch_independent(
    "T9e trainer 传 domain cal",
    "core/trainer_v8.py",
    '''                    tau_swd_rel=cfg["tcst"].get("tau_swd_rel", 0.01),
                    device=device,
                    swd_seed=cfg["output"]["seed"] + cycle,
                    gain_tv_max=float(cfg["tcst"].get("gain_tv_max", 0.36)),
                    offset_tv_max=float(cfg["tcst"].get("offset_tv_max", 0.18)),
                    rel_change_max=float(cfg["tcst"].get("rel_change_max", 0.15)),
                    rsem_delta=float(cfg["tcst"].get("rsem_delta", 0.05)))''',
    '''                    tau_swd_rel=cfg["tcst"].get("tau_swd_rel", 0.01),
                    device=device,
                    swd_seed=cfg["output"]["seed"] + cycle,
                    gain_tv_max=float(cfg["tcst"].get("gain_tv_max", 0.36)),
                    offset_tv_max=float(cfg["tcst"].get("offset_tv_max", 0.18)),
                    rel_change_max=float(cfg["tcst"].get("rel_change_max", 0.15)),
                    rsem_delta=float(cfg["tcst"].get("rsem_delta", 0.05)),
                    domain_cal_patches=src_domain_cal_p)''',
    'domain_cal_patches=src_domain_cal_p)'
)

print("\n===== 补丁阶段完成 =====")
PYEOF

echo ""
echo "===== 验证 T7/T8/T9 ====="
python - << 'PYEOF'
from pathlib import Path
rc = Path('core/risk_controller.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
checks = {
    'T7 MMD 重抽': '_tgt_rng.choice(len(tgt_transport_p)' in tv,
    'T8a context manager': 'def _freeze_bn(model):' in tv,
    'T8b Safe 用 _freeze_bn': 'with _freeze_bn(model):' in tv,
    'T9a bs 签名': 'domain_cal_patches=None):' in rc,
    'T9b SWD 用 domain cal': '_src_swd = domain_cal_patches' in rc,
    'T9c SWD after': '_src_swd1 = domain_cal_patches' in rc,
    'T9d run_tcst_v8 准备': 'src_domain_cal_p = src_tr_p[_dom_idx]' in tv,
    'T9e trainer 传 domain cal': 'domain_cal_patches=src_domain_cal_p)' in tv,
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
PYEOF

echo ""
echo "===== 语法检查 ====="
python - << 'PYEOF'
import ast
from pathlib import Path
for f in ['core/risk_controller.py', 'core/trainer_v8.py']:
    ast.parse(Path(f).read_text())
    print(f"  ✅ {f}")
PYEOF

echo ""
echo "===== 冒烟测试（Houston 1 epoch）====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_p1_smoke/smoke'
yaml.dump(c, open('configs/p1_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_p1_smoke/smoke
python -u scripts/train.py --config configs/p1_smoke.yaml 2>&1 | tail -30

echo ""
echo "✅ P1 (T7-T9) 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
