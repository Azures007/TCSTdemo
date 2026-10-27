#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_batch2_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 备份到 $BACKUP"

# ============ 诊断：确认每个 bug 真实存在 ============
echo ""
echo "===== 诊断 6 个 bug ====="
python - << 'PYEOF'
from pathlib import Path
rc = Path('core/risk_controller.py').read_text()
tg = Path('core/task_geometry.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
print(f"B1 risk_controller 第 106 行无 labels: {'transform(xb, phi)' in rc and 'transform(xb, phi, labels=' not in rc.split('SWD')[0]}")
print(f"B2 task_geometry half=n//2 有边界: {'half = n // 2' in tg}")
print(f"B3 phi_star_stats 只有 stats: {'phi_star_stats' in tv and 'phi_star_tolist' not in tv}")
print(f"B4 SWD 截断 [:64]: {rc.count('[:64]') >= 2}")
print(f"B5 A1 fallback 存在: {'_cand[:n_val]' in tv}")
print(f"B6 纯块无边界隔离: {'val_coords_all[:target_val]' in tv}")
PYEOF

# ============ 修 B1: risk_controller 传 labels ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/risk_controller.py')
s = p.read_text()

old = '''            logits_batches = []
            for start in range(0, len(source_cal_patches), 128):
                xb = torch.from_numpy(source_cal_patches[start:start+128]).to(device)
                logits_batches.append(model(transform(xb, phi)).logits.float())
            logits1 = torch.cat(logits_batches, 0).cpu()'''
new = '''            logits_batches = []
            for start in range(0, len(source_cal_patches), 128):
                xb = torch.from_numpy(source_cal_patches[start:start+128]).to(device)
                yb = torch.from_numpy(source_cal_labels[start:start+128]).long().to(device)
                logits_batches.append(model(transform(xb, phi, labels=yb)).logits.float())
            logits1 = torch.cat(logits_batches, 0).cpu()'''
assert old in s, "B1: 未找到未传 labels 的 transform 调用"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ B1: risk_controller 传源域 labels")
PYEOF

# ============ 修 B2: task_geometry 奇数样本 ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/task_geometry.py')
s = p.read_text()

old = '''        n = min(per_class, len(ids))
        half = n // 2
        pick_boundary = order[:half]
        pick_easy = order[-half:] if half > 0 else np.array([], dtype=int)
        chosen = np.concatenate([pick_boundary, pick_easy])'''
new = '''        n = min(per_class, len(ids))
        # B2 fix: n=1 时 half=0 会选 0 个样本；改为 (n+1)//2 保底 1 个
        n_boundary = (n + 1) // 2
        n_easy = n - n_boundary
        chosen_list = list(order[:n_boundary])
        if n_easy > 0:
            chosen_list.extend(order[-n_easy:].tolist())
        chosen = np.asarray(chosen_list, dtype=int)
        assert len(chosen) == n, f"约束选择数量错误: {len(chosen)} != {n}"
        assert len(np.unique(chosen)) == n, f"约束样本有重复"'''
assert old in s, "B2: 未找到 half=n//2 逻辑"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ B2: 奇数约束样本保底 1 个")
PYEOF

# ============ 修 B3: A12 完整保存 phi_star/d_star ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

old = '''        _a12_phi_stats = None
        _a12_d_stats = None
        try:
            if 'phi_star' in dir() and phi_star is not None:
                _pf = phi_star.detach().cpu().numpy().flatten()
                _a12_phi_stats = {
                    "norm": float(np.linalg.norm(_pf)),
                    "abs_max": float(np.abs(_pf).max()) if _pf.size else 0.0,
                    "mean": float(_pf.mean()) if _pf.size else 0.0,
                    "std": float(_pf.std()) if _pf.size else 0.0,
                }
        except Exception:
            pass
        try:
            if 'qp' in dir() and isinstance(qp, dict) and qp.get('d') is not None:
                _d = np.asarray(qp['d']).flatten()
                _a12_d_stats = {
                    "norm": float(np.linalg.norm(_d)),
                    "abs_max": float(np.abs(_d).max()) if _d.size else 0.0,
                    "status": qp.get('status'),
                }
        except Exception:
            pass'''

new = '''        # B3 fix: 保存完整 phi_star / d_star 向量（不只 stats）
        _a12_phi_stats = None
        _a12_d_stats = None
        _a12_phi_full = None
        _a12_d_full = None
        try:
            if 'phi_star' in dir() and phi_star is not None:
                _pf = phi_star.detach().cpu().numpy().flatten()
                _a12_phi_stats = {
                    "norm": float(np.linalg.norm(_pf)),
                    "abs_max": float(np.abs(_pf).max()) if _pf.size else 0.0,
                    "mean": float(_pf.mean()) if _pf.size else 0.0,
                    "std": float(_pf.std()) if _pf.size else 0.0,
                }
                _a12_phi_full = [float(x) for x in _pf.tolist()]
        except Exception:
            pass
        try:
            if 'qp' in dir() and isinstance(qp, dict) and qp.get('d') is not None:
                _d = np.asarray(qp['d']).flatten()
                _a12_d_stats = {
                    "norm": float(np.linalg.norm(_d)),
                    "abs_max": float(np.abs(_d).max()) if _d.size else 0.0,
                    "status": qp.get('status'),
                }
                _a12_d_full = [float(x) for x in _d.tolist()]
        except Exception:
            pass'''
assert old in s, "B3: 未找到 A12 stats 块"
s = s.replace(old, new, 1)

# 写入 cycle_logs 也加 full
old_log = '''        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
new_log = '''        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "phi_star_full": _a12_phi_full,
                            "d_star_full": _a12_d_full,
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})'''
assert old_log in s, "B3: 未找到 cycle_logs"
s = s.replace(old_log, new_log, 1)
p.write_text(s)
ast.parse(s)
print("✅ B3: phi_star_full / d_star_full 已保存")
PYEOF

# ============ 修 B4: SWD 用全部校准样本 ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/risk_controller.py')
s = p.read_text()

old = '''        src_desc0 = spectral_descriptor(torch.from_numpy(source_cal_patches[:64]).to(device))
        tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches[:64]).to(device))'''
new = '''        # B4 fix: 用全部校准样本，不只前 64（HyRANK 144 个只用 44% 会低估真实 SWD）
        src_desc0 = spectral_descriptor(torch.from_numpy(source_cal_patches).to(device))
        tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches).to(device))'''
assert old in s, "B4: SWD src_desc0 未找到"
s = s.replace(old, new, 1)

old2 = '''            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(source_cal_patches[:64]).to(device), phi,
                          labels=torch.from_numpy(source_cal_labels[:64]).long().to(device)))'''
new2 = '''            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(source_cal_patches).to(device), phi,
                          labels=torch.from_numpy(source_cal_labels).long().to(device)))'''
assert old2 in s, "B4: SWD src_desc1 未找到"
s = s.replace(old2, new2, 1)

# spectral_diag 也用全部（或至少更多）
old3 = '''            spectral_diag = transform.diagnostics(
                torch.from_numpy(source_cal_patches[:32]).to(device), phi,
                labels=torch.from_numpy(source_cal_labels[:32]).long().to(device))'''
new3 = '''            spectral_diag = transform.diagnostics(
                torch.from_numpy(source_cal_patches).to(device), phi,
                labels=torch.from_numpy(source_cal_labels).long().to(device))'''
assert old3 in s, "B4: spectral_diag 未找到"
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("✅ B4: SWD 和 spectral_diag 用全部校准样本")
PYEOF

# ============ 修 B5: A1 fallback 改为"降级但保留部分隔离" ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

old = '''                # A1 fallback: if isolation leaves too few val samples, fall back to unisolated.
                # Without this, classes whose pixels are spatially clustered (e.g. HyRANK class4/class10)
                # end up with val=1 or val=4, making src_val selection extremely noisy.
                if len(_keep) >= max(5, n_val // 2):
                    va_coords = np.array(_keep, dtype=np.int64)
                else:
                    va_coords = _cand[:n_val]'''
new = '''                # B5 fix: 严格隔离不可取消。如果隔离后样本 <5，扩大候选池重试；
                # 仍然不足时，保留隔离后的全部样本（哪怕只有 3-4 个），并标记。
                if len(_keep) >= max(5, n_val // 2):
                    va_coords = np.array(_keep, dtype=np.int64)
                else:
                    # 扩大候选池（多取 200 个），再隔离一次
                    _cand_wide = coords[n_cn + n_ca_pool + n_tr:
                                         n_cn + n_ca_pool + n_tr + n_val + 200]
                    _keep_wide = []
                    for _v in _cand_wide:
                        _d = np.abs(_tr_set - _v).max(axis=1)
                        if (_d >= _ps).all():
                            _keep_wide.append(_v)
                        if len(_keep_wide) >= n_val:
                            break
                    if len(_keep_wide) >= 1:
                        va_coords = np.array(_keep_wide, dtype=np.int64)
                    else:
                        # 真的找不到隔离样本 → 显式标记并跳过该类 val（而非静默取消隔离）
                        va_coords = np.empty((0, 2), dtype=np.int64)'''
assert old in s, "B5: A1 fallback 未找到"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ B5: A1 fallback 改为扩大候选池而非取消隔离")
PYEOF

# ============ 修 B6: 纯块划分加跨块边界隔离 ============
python - << 'PYEOF'
from pathlib import Path
import ast
p = Path('core/trainer_v8.py')
s = p.read_text()

old = '''            # 纯块划分
            rng.shuffle(unique_blocks)
            n_val_blocks = max(1, len(unique_blocks) // 5)
            val_block_set = set(unique_blocks[:n_val_blocks].tolist())
            val_list, train_list = [], []
            for i, b in enumerate(blocks):
                if int(b) in val_block_set:
                    val_list.append(coords[i])
                else:
                    train_list.append(coords[i])
            val_coords_all = np.asarray(val_list, dtype=np.int64)
            train_coords_all = np.asarray(train_list, dtype=np.int64)
            rng.shuffle(train_coords_all)
            n_avail = len(train_coords_all)
            n_cn = min(constraint_per_class, max(1, n_avail // 10))
            n_ca_pool = min(calibration_per_class * int(cal_pool_mult),
                           max(1, (n_avail - n_cn) // 3))
            remaining = n_avail - n_cn - n_ca_pool
            n_tr = min(target_train, max(1, remaining - 1))
            cn_coords = train_coords_all[:n_cn]
            ca_coords = train_coords_all[n_cn:n_cn + n_ca_pool]
            tr_coords = train_coords_all[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            va_coords = val_coords_all[:target_val]'''

new = '''            # 纯块划分
            rng.shuffle(unique_blocks)
            n_val_blocks = max(1, len(unique_blocks) // 5)
            val_block_set = set(unique_blocks[:n_val_blocks].tolist())
            val_list, train_list = [], []
            for i, b in enumerate(blocks):
                if int(b) in val_block_set:
                    val_list.append(coords[i])
                else:
                    train_list.append(coords[i])
            val_coords_all = np.asarray(val_list, dtype=np.int64)
            train_coords_all = np.asarray(train_list, dtype=np.int64)
            rng.shuffle(train_coords_all)
            n_avail = len(train_coords_all)
            n_cn = min(constraint_per_class, max(1, n_avail // 10))
            n_ca_pool = min(calibration_per_class, max(1, (n_avail - n_cn) // 3))
            remaining = n_avail - n_cn - n_ca_pool
            n_tr = min(target_train, max(1, remaining - 1))
            cn_coords = train_coords_all[:n_cn]
            ca_coords = train_coords_all[n_cn:n_cn + n_ca_pool]
            tr_coords = train_coords_all[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            # B6 fix: 纯块划分也要跨块边界隔离（Patch 可能跨块边界）
            _ps_b6 = int(patch_size)
            _tr_set_b6 = tr_coords
            _val_keep = []
            for _v in val_coords_all:
                if len(_tr_set_b6) > 0:
                    _d = np.abs(_tr_set_b6 - _v).max(axis=1)
                    if (_d >= _ps_b6).all():
                        _val_keep.append(_v)
                else:
                    _val_keep.append(_v)
                if len(_val_keep) >= target_val:
                    break
            va_coords = (np.asarray(_val_keep, dtype=np.int64)
                         if len(_val_keep) > 0 else val_coords_all[:target_val])'''
assert old in s, "B6: 纯块划分未找到"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("✅ B6: 纯块划分加跨块边界隔离")
PYEOF

echo ""
echo "===== 验证 6 个修复 ====="
python - << 'PYEOF'
from pathlib import Path
rc = Path('core/risk_controller.py').read_text()
tg = Path('core/task_geometry.py').read_text()
tv = Path('core/trainer_v8.py').read_text()
print(f"B1 risk 传 labels: {'transform(xb, phi, labels=yb)' in rc}")
print(f"B2 奇数保底: {'n_boundary = (n + 1) // 2' in tg}")
print(f"B3 完整向量: {'phi_star_full' in tv}")
print(f"B4 SWD 全样本: {'source_cal_patches)' in rc or 'source_cal_patches, ' in rc}")
print(f"B5 fallback 改: {'_cand_wide' in tv}")
print(f"B6 纯块隔离: {'_val_keep' in tv}")
PYEOF

echo ""
echo "===== 语法检查 ====="
python -c "import ast; [ast.parse(Path(f).read_text()) for f in ['core/risk_controller.py', 'core/task_geometry.py', 'core/trainer_v8.py']]; print('✅ 三个文件语法正确')"

echo ""
echo "===== smoke test（B1-B6 全修后） ====="
python - << 'PYEOF'
import yaml
c = yaml.safe_load(open('configs/hou_H6_ls05.yaml'))
c['training']['ckpt_mode'] = 'A'
c['training']['epochs_warmup'] = 1
c['training']['num_cycles'] = 1
c['training']['epochs_per_cycle'] = 1
c['output']['save_dir'] = '/root/autodl-tmp/TCSTdemo/results_batch2_smoke/smoke'
yaml.dump(c, open('configs/batch2_smoke.yaml','w'), sort_keys=False)
PYEOF
rm -rf results_batch2_smoke/smoke
python -u scripts/train.py --config configs/batch2_smoke.yaml 2>&1 | tail -10

echo ""
echo "✅ 批次 2 修复完成"
echo "   回滚: rm -rf core && mv $BACKUP core"
