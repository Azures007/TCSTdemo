#!/bin/bash
set -e
set -o pipefail
cd ~/autodl-tmp/TCSTdemo
BACKUP="core_backup_A2A5_$(date +%Y%m%d_%H%M%S)"
cp -r core "$BACKUP"
echo "✅ core 已备份到 $BACKUP"

python - << 'PYEOF'
from pathlib import Path
import ast

p = Path('core/trainer_v8.py')
s = p.read_text()

# ============ A5: _make_model 读配置 ============
old_make = '''def _make_model(bands, num_classes, device):
    """Doc principle 3: DualReliabilityEncoder only (no DCRN)."""
    _log(f"[backbone] DualReliabilityEncoder (bands={bands}, classes={num_classes})")
    return DualReliabilityEncoder(
        bands=bands, num_classes=num_classes, projection_dim=128,
        projection_norm="batch", aspp_mode="adaptive", pooling_mode="guided",
        spectral_fusion_mode="reliable", amplitude_channel=False).to(device)'''
new_make = '''def _make_model(bands, num_classes, device, cfg=None):
    """Doc principle 3: DualReliabilityEncoder only. Reads architecture from cfg['training']."""
    tr = (cfg or {}).get("training", {})
    proj_dim = int(tr.get("projection_dim", 128))
    proj_norm = tr.get("projection_norm", "batch")
    aspp_mode = tr.get("aspp_mode", "adaptive")
    pool_mode = tr.get("pooling_mode", "guided")
    fusion_mode = tr.get("spectral_fusion_mode", "reliable")
    amp_ch = bool(tr.get("amplitude_channel", False))
    _log(f"[backbone] DualReliabilityEncoder (bands={bands}, classes={num_classes}, "
         f"proj_dim={proj_dim}, proj_norm={proj_norm}, aspp={aspp_mode}, "
         f"pool={pool_mode}, fusion={fusion_mode}, amp={amp_ch})")
    return DualReliabilityEncoder(
        bands=bands, num_classes=num_classes, projection_dim=proj_dim,
        projection_norm=proj_norm, aspp_mode=aspp_mode, pooling_mode=pool_mode,
        spectral_fusion_mode=fusion_mode, amplitude_channel=amp_ch).to(device)'''
assert old_make in s, "A5: _make_model 未找到"
s = s.replace(old_make, new_make, 1)

# 调用点加 cfg
old_call = '    model = _make_model(bands, num_classes, device)'
new_call = '    model = _make_model(bands, num_classes, device, cfg=cfg)'
assert old_call in s, "A5: _make_model 调用点未找到"
s = s.replace(old_call, new_call, 1)

# ============ A2: 重置 optimizer 后重建 scheduler ============
old_reset = '''        optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                          weight_decay=cfg["training"]["weight_decay"])
        _log("[Optimizer] re-initialized after restoring best warmup model")'''
new_reset = '''        optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                          weight_decay=cfg["training"]["weight_decay"])
        # === A2 fix: rebuild scheduler on the new optimizer ===
        if scheduler is not None:
            scheduler = _build_scheduler(optimizer, scheduler_name, total_epochs)
            _log(f"[scheduler] re-bound to new optimizer: {scheduler_name}")
        _log("[Optimizer] re-initialized after restoring best warmup model")'''
assert old_reset in s, "A2: optimizer 重置块未找到"
s = s.replace(old_reset, new_reset, 1)

p.write_text(s)
ast.parse(s)
print("✅ A2 + A5 已修复，语法正确")
PYEOF

echo ""
echo "=== 验证 ==="
python - << 'PYEOF'
from pathlib import Path
tv = Path('core/trainer_v8.py').read_text()
print(f"  A5 _make_model 读配置: {'cfg=cfg' in tv and 'tr.get(\"projection_dim\"' in tv}")
print(f"  A5 调用点传 cfg     : {'_make_model(bands, num_classes, device, cfg=cfg)' in tv}")
print(f"  A2 scheduler 重建    : {'re-bound to new optimizer' in tv}")
PYEOF
echo ""
echo "✅ 完成。回滚命令: rm -rf core && mv $BACKUP core"
