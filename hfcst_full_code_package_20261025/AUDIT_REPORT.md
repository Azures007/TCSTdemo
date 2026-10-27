# HFCST Code Audit — 2026-10-25

## Scope and limitations

Reviewed the uploaded source/config archive. The Python source compiles. The archive does not contain the source/target MAT data, event logs, or model checkpoints, so the three experiments cannot be rerun here and no accuracy improvement can be claimed from static review alone.

## Findings ranked by priority

### P0 — Source split does not enforce class coverage

`core/data_hfcst.py::global_spatial_split` reports per-split class counts but does not ensure every class is represented in the training and validation sets. `core/trainer_hfcst.py::_source_val_metrics` calculates AA only over classes present in validation, so a split missing classes can produce a misleading checkpoint-selection score. Class-balanced sampling cannot recover a class absent from the training set.

**Patch:** record missing classes per split and fail before training when source train/validation coverage is incomplete (`training.strict_source_class_coverage=true` by default). If a run fails, use the reported missing class IDs to revise block assignments while preserving spatial isolation.

### P0 — Residual-QP gradient rows can lose label alignment

`core/task_geometry_hfcst.py::compute_margin_gradients` filters non-finite rows, but previously returned only filtered gradients/margins and a count. In `core/trainer_hfcst.py::_qp_direction`, residual constraints were then grouped using the original, unfiltered label vector. If any gradient row is non-finite, residual constraints can be assigned to the wrong class or indexing can fail.

**Patch:** return the retained input-row indices and apply the same indices to the labels before class-specific QP grouping.

### P1 — Calibration risk checks silently ignore classes absent from calibration

`core/risk_controller_hfcst.py::source_risk_from_logits` assigns `None` to a class with no calibration samples, while the worst-class budget is calculated only over classes that are present. A candidate can therefore pass the class-risk check without evidence for every class.

**Patch:** explicitly report missing calibration classes and reject candidate/post-training acceptance when coverage is incomplete. This is fail-closed behavior; it may expose split problems rather than improve metrics immediately.

### P1 — Adaptation is currently a no-op in the supplied experiment results

The prior logs supplied in the conversation show all three tasks ending with `committed_cycles=0` after rollback. The final model therefore remains the source-only baseline. The posterior check reports transport degradation, source-class risk, and/or target pseudo-label flips. This is not proof that rollback code is broken; the update is failing its own acceptance criteria.

**Plan:** do not relax acceptance thresholds blindly. First strengthen/validate the source-only baseline, then instrument and adjust the adaptation training step so it preserves source margins and target prediction stability. Validate the same shared-code change on all three tasks.

### P1 — Fast-fix learning rate is much lower than the original configurations

The current fast-fix configs use `lr=3e-5`, versus `3e-4` in the original Houston/HyRANK configs and `1e-4` in the original Shanghai-Hangzhou config. The prior source-validation OA values were about 0.56 for Houston and 0.68 for HyRANK. These results suggest the source classifier itself needs a controlled baseline study before attributing all error to domain adaptation. This is a hypothesis to test, not a proven cause.

**Experiment configs included:** source-only runs at `1e-4` and `3e-4` for all three datasets, using the same task-specific split config and seed. `hfcst.num_cycles=0` isolates source training from the adaptation rollback issue. The sweep script runs all three tasks in parallel per learning-rate round.

Use source validation OA/AA and per-class accuracy for model/config selection. Target GT metrics remain final-only reports and must not be used to tune hyperparameters.

### P2 — Band alignment assumes equal band count means matching bands

`core/data_hfcst.py::align_cubes` returns identity alignment whenever source and target band counts match and no mapping is provided. Equal counts alone do not prove that band indices represent the same physical wavelengths. Dataset wavelength metadata must be checked before changing this logic; those MAT files were not included in the archive.

### P2 — Solver fallback is narrower than its documentation suggests

`core/transport_solver_hfcst.py` falls back to SciPy only when importing CVXPY raises `ImportError`. If CVXPY imports successfully but no supported solver is installed or solving fails, it returns a solver error instead of trying SciPy. Prior logs show QP activity, so this is a robustness issue rather than the leading explanation for current accuracy.

## Changes in the reviewed bundle

- Added missing-class diagnostics to spatial-split audit.
- Added fail-fast class coverage checks for source train and validation.
- Fixed residual-QP gradient/label row alignment when non-finite rows are filtered.
- Made candidate and posterior risk checks reject incomplete source calibration class coverage.
- Added source-only learning-rate sweep configs and a launcher script.
- Did not change the model architecture or loosen adaptation acceptance thresholds without evidence.

## Verification performed

- `python -m compileall` passes for the reviewed source tree.
- Synthetic tests pass for filtered-gradient row alignment, incomplete calibration coverage rejection, and missing-class split diagnostics.
- No full training or target accuracy test was possible because the uploaded archive contains no datasets/checkpoints and this environment does not have the user's training data/GPU setup.
