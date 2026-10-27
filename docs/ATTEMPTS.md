# Attempted Mechanisms and Failure Modes

## 1. Optimizer Re-initialization (Fixed in v8.0.5)
After restoring best warm-up checkpoint, reset AdamW states.
Without this, momentum mismatch inflated S->H by ~7 points.
Status: FIXED.

## 2. MMD Adaptive Bandwidth (Median Heuristic)
Changed `mk_mmd2` to use `bandwidths=None` -> median heuristic.
Also wired `domain_gradient_mmd` to pass `bandwidths=None`.
Result: swd_rel dropped from 0.003 to 0.0002 on S->H (bandwidth too small = kernel saturation).
Reverted to fixed (0.5, 1, 2, 4)*median.
Status: WORKS for HyRANK (median), FAILS for S->H (fixed better).

## 3. Spectral TV Plausibility Check
Added gain_tv, offset_tv, relative_change to `spectral_transform.diagnostics`.
Risk Search now rejects transforms with TV > threshold.
Status: WORKS (Houston accepts 2/4 instead of 3/4, slightly lower OA but more honest).

## 4. EMA Teacher (Tried in stage3)
Added `ema_teacher = copy.deepcopy(model)`, update per step with decay=0.999.
Result: NO EFFECT on all 3 datasets (identical to baseline).
Root cause: `force_final_epoch=False` restores `best_src_val` checkpoint. Warmup
already reaches src_val=100.00 (saturated), so cycles' EMA updates are discarded.
Status: FAILED (mechanism-level).

## 5. Pseudo-label Self-training (Tried in stage3)
Per cycle, select target samples with max_prob > 0.95, use as pseudo-labels in CE.
Result:
- Houston: 0 effect (ff=False discards cycle checkpoints)
- HyRANK: OA 62.25 -> 55.32 (-6.93). class2/class5 are 0% in teacher predictions,
  so pseudo-labels for these classes are WRONG and pollute training.
- S->H: 0 effect (no target sample reaches 0.95 confidence)
Status: FAILED (HyRANK negative transfer).

## 6. Class-Conditional T-CST (Tried in stage5)
Per-class Gain+Offset: `phi` dim = C*2K (Houston 112, HyRANK 192, S->H 48).
Result: 0 effect on all 3 datasets (identical to baseline).
Root cause: same as EMA - `best_src_val` selection discards cycle checkpoints.
Status: FAILED (mechanism-level).

## 7. MMD Bandwidth Tuning (Tried in multiple rounds)
Range from 0.02*median to 4*median.
Result: swd_rel monotonically decreases as bandwidth decreases (Gaussian kernel
bidirectional saturation). Gradient is maximized at bandwidth ~= median distance.
Status: NO IMPROVEMENT over (0.5, 1, 2, 4)*median.

## 8. QP Regularization lambda_d (Tried in stage4)
Reduced lambda_d from 1e-2 to 1e-4 (100x).
Result:
- HyRANK: 6/6 ACCEPT but OA drops 59.56 -> 49.16 (-10.40). Negative transfer:
  class1 -22.57, class7 -26.97, class8 -15.52 while class3 +31.92, class9 +14.36.
  This is direct evidence of category-conditional domain shift.
- S->H: 2/2 ACCEPT but OA identical to 0/2 (warmup already saturated at src_val=100).
Status: FAILED (negative transfer).

## 9. Focal Loss (Tried in stage2)
gamma=1.5 replacing CE.
Result:
- Houston: 79.37 -> 77.82 (-1.55), AA 78.55 -> 72.71 (-5.84)
- HyRANK: class3 +58.67 but class1 -20.84, class8 -24.05 (net 0)
- S->H: OA 70.89 -> 73.62 (+2.73) but AA 72.49 -> 70.06 (-2.43), Plant 43.65 -> 22.67
Status: FAILED (sample-level reweighting ill-suited for systematic domain shift).

## 10. 100/class (label budget reduction)
Reduced per_class_train_cap from 150 to 100.
Result:
- Houston: mean 77.85 -> 78.08 (+0.23), std 2.90 -> 2.09 (-0.81). Slightly better.
- HyRANK: mean 53.99 -> 39.54 (-14.45). Collapsed (12 classes need 150/class).
- S->H: mean 58.28 -> 49.38 (-8.90). Mode collapse worse.
Status: Houston OK, HyRANK/S->H need 150/class.

## Key Finding: S->H Mode Collapse
S->H (3 classes: Water, Land, Plant) has 3-seed std = 25.59 with 150/class.
- seed=1237: 78.27 (Water 84, Land 98, Plant 38)
- seed=42: 29.45 (Water 99, Land 26, Plant 6)
- seed=2024: 67.13 (Water 96, Land 94, Plant 4)

The model collapses to a single majority class depending on initialization.
Tried: noamp (std 12.74), lr=1e-4 (std 25.59), batch=128 (std 23.62),
100/class (std 27.20). None solve the mode collapse.

## Key Finding: HyRANK class2/class5 = 0%
12 classes, 150/class. class2 and class5 are 0% across all seeds and configs.
Root cause: Dioni source domain has < 30 labeled pixels for these classes, so
150/class sampling can't create a meaningful training set. Any mechanism that
tries to learn these classes (pseudo-labels, EMA) amplifies noise.
