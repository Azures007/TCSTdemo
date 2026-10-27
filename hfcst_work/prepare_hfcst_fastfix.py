#!/usr/bin/env python3
"""Run from hfcst_work root. Applies guarded split/sampling patch and writes fresh configs; does not train."""
from pathlib import Path
from datetime import datetime
import re, shutil, yaml

root = Path.cwd()
data = root / "core/data_hfcst.py"
trainer = root / "core/trainer_hfcst.py"
cfgdir = root / "configs_hfcst"
if not data.exists() or not trainer.exists() or not cfgdir.exists():
    raise SystemExit("Run this script from ~/autodl-tmp/TCSTdemo/hfcst_work")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dtext, ttext = data.read_text(encoding="utf-8"), trainer.read_text(encoding="utf-8")

repls = [
    (dtext,
     """def global_spatial_split(gt, valid_mask, num_classes, patch_size=11, grid_size=8,
                         seed=1237, train_cap=150, constraint_cap=12,
                         calibration_cap=80, validation_cap=150):""",
     """def global_spatial_split(gt, valid_mask, num_classes, patch_size=11, grid_size=8,
                         seed=1237, train_cap=150, constraint_cap=12,
                         calibration_cap=80, validation_cap=150,
                         block_overrides=None):""",
     "global_spatial_split signature"),
    (dtext,
     """    assignment, block_audit = _assign_source_blocks(blocks, gt, num_classes, ratios, rng)
    radius = patch_size // 2""",
     """    assignment, block_audit = _assign_source_blocks(blocks, gt, num_classes, ratios, rng)
    overrides_applied = []
    for item in (block_overrides or []):
        bid = int(item["block"])
        expected, destination = item.get("from"), str(item["to"])
        if bid not in assignment:
            raise ValueError(f"Block override references unknown block {bid}")
        if destination not in ratios:
            raise ValueError(f"Invalid split {destination!r} for block {bid}")
        actual = assignment[bid]
        if expected is not None and actual != str(expected):
            raise ValueError(f"Block {bid} expected split {expected!r}, found {actual!r}; aborting")
        assignment[bid] = destination
        overrides_applied.append({"block": bid, "from": actual, "to": destination})
    if overrides_applied:
        block_audit["explicit_overrides"] = overrides_applied
        block_audit["block_counts_after_overrides"] = {
            s: int(sum(v == s for v in assignment.values())) for s in ratios
        }
        class_pixels = {s: np.zeros(num_classes, dtype=np.int64) for s in ratios}
        for bid, r0, r1, c0, c1 in blocks:
            labels = gt[r0:r1, c0:c1]
            labels = labels[(labels > 0) & (labels <= num_classes)]
            if labels.size:
                class_pixels[assignment[bid]] += np.bincount(
                    labels, minlength=num_classes + 1
                )[1:num_classes + 1]
        block_audit["class_pixel_counts_by_block_split"] = {
            s: class_pixels[s].tolist() for s in ratios
        }
    radius = patch_size // 2""",
     "block override insertion"),
]
for text, old, new, label in repls:
    if text.count(old) != 1:
        raise SystemExit(f"Preflight failed: {label}; expected one anchor, found {text.count(old)}")
if ttext.count('        validation_cap=int(tr_cfg.get("per_class_validation_cap", 150)))') != 1:
    raise SystemExit("Preflight failed: trainer split-call anchor not unique")
old_loader = """    ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_labels).long())
    loader = DataLoader(ds, batch_size=int(training_cfg.get("batch_size", 16)), shuffle=True,
                        drop_last=False, num_workers=0,
                        generator=torch.Generator().manual_seed(int(seed)))"""
if ttext.count(old_loader) != 1:
    raise SystemExit("Preflight failed: DataLoader anchor not unique")

# Find the section assigned to tc_cfg.
m = re.search(r"""tc_cfg\s*=\s*cfg\.get\(\s*['"]([^'"]+)['"]""", ttext)
tc_key = m.group(1) if m else None
tasks = ("houston", "hyrank", "shanghai_hangzhou")
loaded = {}
for task in tasks:
    p = cfgdir / f"{task}_trust001.yaml"
    if not p.exists():
        raise SystemExit(f"Missing config: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("training"), dict):
        raise SystemExit(f"No training section in {p}")
    key = tc_key
    if not key or not isinstance(cfg.get(key), dict):
        candidates = [k for k, v in cfg.items() if isinstance(v, dict) and
                      any(x in v for x in ("shared_trust_radius", "residual_trust_radius", "min_effective_target_n", "K"))]
        if len(candidates) != 1:
            raise SystemExit(f"Cannot identify tc_cfg in {p}; candidates={candidates}")
        key = candidates[0]
    loaded[task] = (cfg, key)

# Only now create backups and edit source.
dbak = data.with_name(data.name + f".bak_fastfix_{stamp}")
tbak = trainer.with_name(trainer.name + f".bak_fastfix_{stamp}")
shutil.copy2(data, dbak)
shutil.copy2(trainer, tbak)

dtext = dtext.replace(repls[0][1], repls[0][2], 1).replace(repls[1][1], repls[1][2], 1)
new_call = """        validation_cap=int(tr_cfg.get("per_class_validation_cap", 150)),
        block_overrides=tr_cfg.get("block_split_overrides"))"""
ttext = ttext.replace('        validation_cap=int(tr_cfg.get("per_class_validation_cap", 150)))', new_call, 1)
new_loader = """    ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_labels).long())
    batch_size = int(training_cfg.get("batch_size", 16))
    generator = torch.Generator().manual_seed(int(seed))
    if bool(training_cfg.get("class_balanced_sampling", False)):
        labels_np = np.asarray(train_labels, dtype=np.int64)
        counts = np.bincount(labels_np, minlength=int(labels_np.max()) + 1).astype(np.float64)
        class_weights = np.zeros_like(counts)
        present = counts > 0
        class_weights[present] = 1.0 / np.sqrt(counts[present])
        weights = torch.as_tensor(class_weights[labels_np], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(labels_np), replacement=True, generator=generator
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                            drop_last=False, num_workers=0, generator=generator)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=0, generator=generator)"""
ttext = ttext.replace(old_loader, new_loader, 1)
data.write_text(dtext, encoding="utf-8")
trainer.write_text(ttext, encoding="utf-8")

overrides = {
    "houston": [
        {"block": 12, "from": "constraint", "to": "train"},
        {"block": 53, "from": "calibration", "to": "train"},
        {"block": 28, "from": "train", "to": "calibration"},
    ],
    "hyrank": [
        {"block": 15, "from": "constraint", "to": "train"},
        {"block": 38, "from": "train", "to": "constraint"},
    ],
    "shanghai_hangzhou": [],
}
for task, (cfg, key) in loaded.items():
    tr, tc = cfg["training"], cfg[key]
    tr["epochs_warmup"] = 80
    tr["epochs_per_cycle"] = 1
    tr["class_balanced_sampling"] = task in ("houston", "hyrank")
    tr["lambda_transport"] = 0.10
    tr["block_split_overrides"] = overrides[task]
    tc["num_cycles"] = 2
    tc["shared_trust_radius"] = 0.03
    tc["residual_trust_radius"] = 0.01
    if "epochs_per_cycle" in tc:
        tc["epochs_per_cycle"] = 1
    cfg.setdefault("output", {})["save_dir"] = f"results_hfcst/{task}_fastfix_{stamp}_s1237"
    for k in ("restore_checkpoint", "checkpoint", "resume", "pretrained", "init_checkpoint"):
        cfg.pop(k, None)
    out = cfgdir / f"{task}_fastfix_{stamp}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Config: {out}")

print(f"Backups: {dbak} ; {tbak}")
print("Patch applied; no training was started.")
