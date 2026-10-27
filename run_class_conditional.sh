#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

# ============ 0) 备份 core ============
BACKUP="core_backup_cc_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 已备份到 $BACKUP"
echo "   回滚命令: rm -rf core && mv $BACKUP core"

# ============ 1) 补丁 1: spectral_transform.py ============
echo ""
echo "=== 补丁 1/5: spectral_transform.py 加 ClassSmoothSpectralTransform ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/spectral_transform.py')
s = p.read_text()

# 1a) SmoothSpectralTransform.forward 加 labels/class_probs（向后兼容，忽略）
old_fwd = '''    def forward(self, x, phi):
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        a = self.gain(phi[: self.K]).view(1, -1, 1, 1)
        b = self.offset(phi[self.K:]).view(1, -1, 1, 1)
        return x * a + b'''
new_fwd = '''    def forward(self, x, phi, labels=None, class_probs=None):
        # labels/class_probs are ignored for global transform (backward compat)
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        a = self.gain(phi[: self.K]).view(1, -1, 1, 1)
        b = self.offset(phi[self.K:]).view(1, -1, 1, 1)
        return x * a + b'''
assert old_fwd in s, "SmoothSpectralTransform.forward 未找到"
s = s.replace(old_fwd, new_fwd, 1)

# 1b) SmoothSpectralTransform.diagnostics 加 labels（忽略）
old_diag = '''    @torch.no_grad()
    def diagnostics(self, x, phi):'''
new_diag = '''    @torch.no_grad()
    def diagnostics(self, x, phi, labels=None):'''
assert old_diag in s, "SmoothSpectralTransform.diagnostics 未找到"
s = s.replace(old_diag, new_diag, 1)

# 1c) 文件末尾追加 ClassSmoothSpectralTransform
class_code = '''

class ClassSmoothSpectralTransform(nn.Module):
    """Per-class smooth spectral transform: each class c has its own (a_c, b_c).

    Source domain: labels=c gives hard assignment (a_c, b_c).
    Target domain: class_probs (N, C) gives soft mixture (sum_c p_c * a_c).
    phi is [C*2K] flat: alpha[C,K] then beta[C,K].
    """

    def __init__(self, bands, num_classes, K=8, rho_a=0.10, rho_b=0.10):
        super().__init__()
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.K = int(K)
        self.rho_a = float(rho_a)
        self.rho_b = float(rho_b)
        self.register_buffer("U", build_dct_basis(self.bands, self.K), persistent=True)

    @property
    def dim(self):
        return self.num_classes * 2 * self.K

    def gain_all(self, alpha):
        # alpha: [C, K]; U: [B, K];  -> [C, B]
        return torch.exp(self.rho_a * torch.tanh(self.U @ alpha.t()).t())

    def offset_all(self, beta):
        return self.rho_b * torch.tanh(self.U @ beta.t()).t()

    def _split(self, phi):
        C, K = self.num_classes, self.K
        alpha = phi[: C * K].view(C, K)
        beta = phi[C * K:].view(C, K)
        return alpha, beta

    def forward(self, x, phi, labels=None, class_probs=None):
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        alpha, beta = self._split(phi)
        a = self.gain_all(alpha)   # [C, B]
        b = self.offset_all(beta)  # [C, B]
        if labels is not None:
            labels = labels.long()
            a_n = a[labels]          # [N, B]
            b_n = b[labels]
        elif class_probs is not None:
            a_n = class_probs @ a    # [N, B]
            b_n = class_probs @ b
        else:
            a_n = a.mean(0, keepdim=True).expand(x.size(0), -1)
            b_n = b.mean(0, keepdim=True).expand(x.size(0), -1)
        a_n = a_n.view(-1, self.bands, 1, 1)
        b_n = b_n.view(-1, self.bands, 1, 1)
        return x * a_n + b_n

    @torch.no_grad()
    def diagnostics(self, x, phi, labels=None):
        xt = self.forward(x, phi, labels=labels)
        diff = xt - x
        rel = float(diff.norm() / x.norm().clamp_min(1e-8))
        alpha, beta = self._split(phi)
        a = self.gain_all(alpha)   # [C, B]
        b = self.offset_all(beta)
        if a.size(1) > 1:
            gain_tv_per = (a[:, 1:] - a[:, :-1]).abs().mean(dim=1)
            offset_tv_per = (b[:, 1:] - b[:, :-1]).abs().mean(dim=1)
            gain_tv = float(gain_tv_per.max())
            offset_tv = float(offset_tv_per.max())
        else:
            gain_tv = offset_tv = 0.0
        return {
            "relative_spectral_change": rel,
            "gain_min": float(a.min()), "gain_max": float(a.max()),
            "offset_min": float(b.min()), "offset_max": float(b.max()),
            "gain_tv": gain_tv, "offset_tv": offset_tv,
        }
'''
if 'class ClassSmoothSpectralTransform' not in s:
    s = s + class_code
p.write_text(s)
ast.parse(s)
print("✅ spectral_transform.py: ClassSmoothSpectralTransform 已添加")
PYEOF

# ============ 2) 补丁 2: task_geometry.py ============
echo ""
echo "=== 补丁 2/5: task_geometry.py 传源域标签 ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/task_geometry.py')
s = p.read_text()

old = '''                phi = torch.zeros(dim, device=device, requires_grad=True)
                x_phi = transform(x, phi)'''
new = '''                phi = torch.zeros(dim, device=device, requires_grad=True)
                labels_t = torch.tensor([y], device=device, dtype=torch.long)
                x_phi = transform(x, phi, labels=labels_t)'''
assert old in s, "compute_margin_gradients transform 调用未找到"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ task_geometry.py: 已传 labels")
PYEOF

# ============ 3) 补丁 3: discrepancy.py ============
echo ""
echo "=== 补丁 3/5: discrepancy.py domain_gradient_mmd 支持 source_labels ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/discrepancy.py')
s = p.read_text()

old_sig = '''def domain_gradient_mmd(transform, source_images, target_descriptor,
                        bandwidths=None):'''
new_sig = '''def domain_gradient_mmd(transform, source_images, target_descriptor,
                        bandwidths=None, source_labels=None):'''
assert old_sig in s, "domain_gradient_mmd 签名未找到"
s = s.replace(old_sig, new_sig, 1)

old_call = '''    phi = torch.zeros(transform.dim, device=source_images.device, requires_grad=True)
    d_s = spectral_descriptor(transform(source_images, phi))
    loss = mk_mmd2(d_s, target_descriptor, bandwidths)'''
new_call = '''    phi = torch.zeros(transform.dim, device=source_images.device, requires_grad=True)
    if source_labels is not None:
        d_s = spectral_descriptor(transform(source_images, phi, labels=source_labels))
    else:
        d_s = spectral_descriptor(transform(source_images, phi))
    loss = mk_mmd2(d_s, target_descriptor, bandwidths)'''
assert old_call in s, "domain_gradient_mmd 调用未找到"
s = s.replace(old_call, new_call, 1)
p.write_text(s)
ast.parse(s)
print("✅ discrepancy.py: 已支持 source_labels")
PYEOF

# ============ 4) 补丁 4: risk_controller.py ============
echo ""
echo "=== 补丁 4/5: risk_controller.py 源域 labels 传递 ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/risk_controller.py')
s = p.read_text()

# 4a) src_desc1 调用
old1 = '''            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(source_cal_patches[:64]).to(device), phi))'''
new1 = '''            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(source_cal_patches[:64]).to(device), phi,
                          labels=torch.from_numpy(source_cal_labels[:64]).long().to(device)))'''
assert old1 in s, "src_desc1 调用未找到"
s = s.replace(old1, new1, 1)

# 4b) spectral_diag
old2 = '''            spectral_diag = transform.diagnostics(
                torch.from_numpy(source_cal_patches[:32]).to(device), phi)'''
new2 = '''            spectral_diag = transform.diagnostics(
                torch.from_numpy(source_cal_patches[:32]).to(device), phi,
                labels=torch.from_numpy(source_cal_labels[:32]).long().to(device))'''
assert old2 in s, "diagnostics 调用未找到"
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("✅ risk_controller.py: 已传源域 labels")
PYEOF

# ============ 5) 补丁 5: trainer_v8.py ============
echo ""
echo "=== 补丁 5/5: trainer_v8.py 支持类条件选择 ==="
python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# 5a) 导入 ClassSmoothSpectralTransform
old_imp = 'from core.spectral_transform import SmoothSpectralTransform'
new_imp = 'from core.spectral_transform import SmoothSpectralTransform, ClassSmoothSpectralTransform'
assert old_imp in s, "import 未找到"
s = s.replace(old_imp, new_imp, 1)

# 5b) transform 选择
old_tr = '''    transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],
                                        rho_a=cfg["tcst"]["rho_a"],
                                        rho_b=cfg["tcst"]["rho_b"]).to(device)'''
new_tr = '''    use_class_cond = bool(cfg["tcst"].get("use_class_conditional", False))
    if use_class_cond:
        transform = ClassSmoothSpectralTransform(
            bands, num_classes=num_classes, K=cfg["tcst"]["K"],
            rho_a=cfg["tcst"]["rho_a"], rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] CLASS-CONDITIONAL: C={num_classes}, K={cfg['tcst']['K']}, dim={transform.dim}")
    else:
        transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],
                                            rho_a=cfg["tcst"]["rho_a"],
                                            rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] GLOBAL: K={cfg['tcst']['K']}, dim={transform.dim}")'''
assert old_tr in s, "transform 初始化未找到"
s = s.replace(old_tr, new_tr, 1)

# 5c) MMD 调用传 labels
old_mmd = '''        src_mmd_idx = _class_balanced_subsample(src_tr_y, 128,
                                                seed=cfg["output"]["seed"] + 500 + cycle)
        src_for_mmd = torch.from_numpy(src_tr_p[src_mmd_idx]).to(device)
        tgt_desc = spectral_descriptor(
            torch.from_numpy(tgt_transport_p[:128]).to(device))
        g_D, mmd0 = domain_gradient_mmd(transform, src_for_mmd, tgt_desc)'''
new_mmd = '''        src_mmd_idx = _class_balanced_subsample(src_tr_y, 128,
                                                seed=cfg["output"]["seed"] + 500 + cycle)
        src_for_mmd = torch.from_numpy(src_tr_p[src_mmd_idx]).to(device)
        src_mmd_labels = torch.from_numpy(src_tr_y[src_mmd_idx]).long().to(device)
        tgt_desc = spectral_descriptor(
            torch.from_numpy(tgt_transport_p[:128]).to(device))
        g_D, mmd0 = domain_gradient_mmd(transform, src_for_mmd, tgt_desc,
                                         source_labels=src_mmd_labels)'''
assert old_mmd in s, "MMD 调用未找到"
s = s.replace(old_mmd, new_mmd, 1)

# 5d) _train_step 里的 transform
old_ts = '''                loss_safe = F.cross_entropy(model(transform(xb, phi_star)).logits, yb,
                                            label_smoothing=label_smoothing_value)'''
new_ts = '''                loss_safe = F.cross_entropy(model(transform(xb, phi_star, labels=yb)).logits, yb,
                                            label_smoothing=label_smoothing_value)'''
assert old_ts in s, "_train_step transform 调用未找到"
s = s.replace(old_ts, new_ts, 1)

p.write_text(s)
ast.parse(s)
print("✅ trainer_v8.py: 类条件支持已植入")
PYEOF

# ============ 6) Smoke test: Houston 1 epoch ============
echo ""
echo "=== Smoke test (Houston cc, 1 epoch) ==="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['tcst']['use_class_conditional'] = True
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_cc/smoke'
yaml.dump(c, open('configs/smoke_cc.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_cc/smoke
python -u scripts/train.py --config configs/smoke_cc.yaml 2>&1 | tail -20
echo "✅ Smoke test 通过"

# ============ 7) 生成 9 个配置 ============
mkdir -p results_cc/{houston,hyrank,shanghai}
python - << 'PYEOF'
import yaml, copy

BASES = {
    'houston':  'configs/hou_H6_ls05.yaml',
    'hyrank':   'configs/hy_Y2_warm40.yaml',
    'shanghai': 'configs/sh_S5_ls005.yaml',
}
GROUPS = {
    'A_baseline': {'use_class_conditional': False},
    'B_cc010':    {'use_class_conditional': True},   # rho 保持配置原值
    'C_cc015':    {'use_class_conditional': True},   # rho 覆盖为 0.15
}
for ds, base_path in BASES.items():
    base = yaml.safe_load(open(base_path))
    for grp, flags in GROUPS.items():
        c = copy.deepcopy(base)
        for k, v in flags.items():
            c['tcst'][k] = v
        if grp == 'C_cc015':
            c['tcst']['rho_a'] = 0.15
            c['tcst']['rho_b'] = 0.15
        if grp == 'B_cc010':
            c['tcst']['rho_a'] = 0.10
            c['tcst']['rho_b'] = 0.10
        tag = f'{ds[:2].upper()}_{grp}'
        c['output']['save_dir'] = f'/root/autodl-tmp/TCSTdemo/results_cc/{ds}/{grp}'
        yaml.dump(c, open(f'configs/{tag}.yaml','w'), sort_keys=False)
print("✅ 9 个配置已生成")
PYEOF

# ============ 8) 跑 9 变体 ============
run_one() {
    local cfg="$1" tag="$2" outdir="$3"
    echo ""
    echo "=================================================="
    echo "  RUN: $tag   开始: $(date +%H:%M:%S)"
    echo "=================================================="
    python -u scripts/train.py --config "$cfg" 2>&1 | \
        stdbuf -oL -eL tee "$outdir/log_$tag.log" | \
        stdbuf -oL grep --line-buffered -E \
          "warm [0-9]+/|cyc[0-9]+ ep|\[transform\]|\[risk\].*(ACCEPT|REJECT)|\[Phase|restored|OA |AA |Kappa|class[0-9]+ "
    echo "  DONE: $tag   结束: $(date +%H:%M:%S)"
}

for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/HO_$grp.yaml" "$grp" "results_cc/houston"
done
for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/HY_$grp.yaml" "$grp" "results_cc/hyrank"
done
for grp in A_baseline B_cc010 C_cc015; do
    run_one "configs/SH_$grp.yaml" "$grp" "results_cc/shanghai"
done

# ============ 9) 汇总 ============
echo ""
echo "=================================================="
echo "  类条件 T-CST 消融汇总"
echo "=================================================="
python - << 'PYEOF' | tee results_cc/SUMMARY.txt
import json, os
BASE = {
    'houston':  (80.78, 77.27),
    'hyrank':   (62.25, 54.14),
    'shanghai': (77.42, 72.52),
}
GROUPS = ['A_baseline', 'B_cc010', 'C_cc015']
for ds, (oa0, aa0) in BASE.items():
    print(f"\n[{ds}]  参考基线（v8.0.6）: OA={oa0:.2f}  AA={aa0:.2f}")
    print(f"{'group':<14} {'OA':>8} {'dOA':>8} {'AA':>8} {'dAA':>8}  per-class")
    print('-'*115)
    for grp in GROUPS:
        p = f'results_cc/{ds}/{grp}/final_metrics.json'
        if not os.path.exists(p):
            print(f"{grp:<14} (missing)"); continue
        m = json.load(open(p))['target_metrics']
        pc = [round(x,1) for x in m['class_accuracy']]
        print(f"{grp:<14} {m['OA']:>8.2f} {m['OA']-oa0:>+8.2f} {m['AA']:>8.2f} {m['AA']-aa0:>+8.2f}  {pc}")
PYEOF

# ============ 10) 结束（不关机）============
echo ""
echo "=================================================="
echo "  ✅ 类条件 T-CST 全部完成（不关机）"
echo "  结束时间: $(date)"
echo "  结果: ~/autodl-tmp/TCSTdemo/results_cc/"
echo "  汇总: ~/autodl-tmp/TCSTdemo/results_cc/SUMMARY.txt"
echo "  备份: ~/autodl-tmp/TCSTdemo/$BACKUP"
echo "  回滚: rm -rf core && mv $BACKUP core"
echo "=================================================="
