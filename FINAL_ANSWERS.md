# T-CST v8.1.0 最终回答（对应提示词第十节 A/B/C/D）

## A. 原始 v8.0.9 的低准确率中，哪些问题确实来自实现与实验流程？

### A1. 已确认的实现 bug（P0/P1 修复）

| # | 问题 | 修复前 | 修复后 | 影响 |
|---|---|---|---|---|
| P0.3 | Patch 空间重叠 | Houston 7/7 类 100% 重叠、HyRANK 4/12 类 100% | `strict_mixed_isolation` 开关（默认 false = 旧行为，true 时 Houston 7/7 val=0） | Houston src_val=100% 含重叠；HyRANK src_val 部分虚高 |
| P1.1 | checkpoint 只保存 best_src_val | 无法审计 | 保存 warmup + cycle 全 manifest（src_val, nll, safe_executed_count） | 首次看到 Houston safe=60/60、HyRANK/S→H safe=0 |
| P1.2 | checkpoint 选择只看 src_val | warmup 饱和时 cycles 全丢 | 新增 ckpt_mode=D（合格集 + NLL 排序） | Houston +2.21（D vs A），但 S→H -13.30，最终统一 A |
| P1 增强 | `use_source_augmentation=true` 假生效 | warmup 和 cycles 都没增强 | 新增 `use_warmup_augmentation` 开关 | Houston -0.97、HyRANK -1.91、S→H +0.52 但 std 15.85 → 关闭 |
| P1.3 | 无法判断 safe 分支是否执行 | 只能看 ACCEPT 日志 | manifest 记录 `safe_executed_count` | Houston 60/60、HyRANK 0/51、S→H 0/15 |
| P2 | 无 per-class margin 诊断 | 只能看 target OA | 新增 `per_class_margin` + `tgt_pred_dist` | Houston class0/class3 margin 恶化 |

### A2. 实验流程问题

| 问题 | 证据 | 影响 |
|---|---|---|
| HyRANK class2/class5 不是 <30 像素 | P0.1 数据审计：class1=204、class5=361 | 之前"像素太少"的解释错误，真实原因是类别条件偏移 |
| 三数据集波段数相同，`align_bands` 不触发 | P0.2 | 归一化顺序问题在当前数据集不存在 |
| Houston water 只有 22 个测试像素 | P0.1 | water 准确率 95.5%–100% 的波动是 1 像素的 4.5% |

### A3. 不能归咎于实现的部分

- **Houston 的 76.12**：B0 Source-only = 76.12，T-CST cycles 净贡献 = 0（src_val 饱和）。**Houston 的低准确率不是 bug，是 warmup 饱和**
- **HyRANK 的 54.28**：B0 = 55.42，T-CST cycles 净贡献 = **-1.14**。**T-CST cycles 对 HyRANK 有害**
- **S→H 的 63.93**：B0 = 64.54，T-CST cycles 净贡献 = **-0.61**。**T-CST cycles 对 S→H 也有害**

**结论 A**：实现 bug 确实存在（P0/P1/P2 修了 6 项），但修完后 T-CST 的 cycles 对三个数据集**都没有正贡献**（Houston +0.00、HyRANK -1.14、S→H -0.61）。低准确率的主要来源不是实现 bug，是 **T-CST 的 MMD 方向本身无效**。

---

## B. 修复以后，原始 T-CST 是否在三个数据集上真正产生独立收益？

### B1. B0 Source-only vs v8.1.0（含 cycles）

| 数据集 | B0（纯 warmup） | v8.1.0（含 cycles） | T-CST 净贡献 |
|---|---|---|---|
| Houston | 76.12±1.12 | 76.12±1.12 | **+0.00** |
| HyRANK | **55.42±1.48** | 54.28±3.09 | **-1.14** |
| S→H | **64.54±9.30** | 63.93±10.11 | **-0.61** |

**逐 seed**：

| 数据集 | seed=1237 | seed=42 | seed=2024 |
|---|---|---|---|
| Houston B0 vs v8.1.0 | 75.99 vs 75.99 | 75.07 vs 75.07 | 77.30 vs 77.30 |
| HyRANK B0 vs v8.1.0 | 54.52 vs 51.00 | 54.61 vs 54.72 | 57.13 vs 57.13 |
| S→H B0 vs v8.1.0 | 74.75 vs 74.75 | 56.56 vs 54.72 | 62.32 vs 62.32 |

### B2. 为什么 T-CST 没有正贡献

| 数据集 | safe 执行 | 原因 |
|---|---|---|
| Houston | 60/60（T-CST 4/4 ACCEPT） | 但 warmup src_val=100%，cycles 的 checkpoint 被 `best_src_val` 丢弃。D 模式能选中 cycles（+2.21），但 D 对 S→H 有害，不能统一 |
| HyRANK | **0/51（6/6 REJECT）** | swd_rel ~3e-3 < 0.01，T-CST 从未执行。cycles 变成纯 CE 训练，B0 的 55.42 → cycles 后的 54.28，**纯 CE 过拟合** |
| S→H | **0/15（2/2 REJECT）** | 同 HyRANK，T-CST 从未执行。cycles 变成纯 CE，B0 的 64.54 → 63.93 |

### B3. ckpt_mode=D 的消融

| 数据集 | A 模式 | D 模式 | Δ |
|---|---|---|---|
| Houston | 76.12±1.12 | **78.33±1.19** | **+2.21** |
| HyRANK | 54.28±3.09 | 56.90±5.30 | +2.62（std 3.09→5.30） |
| S→H | 63.93±10.11 | **50.63±12.63（6-seed）** | **-13.30**，min 28.90 |

**结论 B**：**修复后，原始 T-CST 在三个数据集上都没有产生独立正收益**。Houston 的 T-CST 生效了但被 checkpoint 选择丢弃；HyRANK/S→H 的 T-CST 被 Risk Search 全拒绝。D 模式能让 Houston 涨 2.21，但 S→H 崩 13.30，不能统一。

---

## C. 光谱分支是否存在监督不足或过度抑制的问题？实际证据是什么？

### C1. 光谱分支诊断（source val）

| 数据集 | context OA | **spectral OA** | fused OA | spectral 落后 | 预测不一致 | reliability 中位数 |
|---|---|---|---|---|---|---|
| Houston | 99.90 | **30.58** | 100.00 | **-69.32** | 69.52% | 0.0258 |
| HyRANK | 84.89 | **30.23** | 85.73 | **-54.66** | 71.60% | 0.0464 |
| S→H | 99.56 | **59.33** | 99.56 | **-40.22** | 41.11% | 0.4931 |

### C2. reliability 分布

| 数据集 | mean | std | 5% 分位 | 25% 分位 | 50% 分位 | 75% 分位 | 95% 分位 |
|---|---|---|---|---|---|---|---|
| Houston | 0.3173 | 0.4383 | 0.0181 | 0.0227 | **0.0258** | 0.9732 | 0.9883 |
| HyRANK | 0.3279 | 0.4191 | 0.0209 | 0.0285 | **0.0464** | 0.9727 | 0.9931 |
| S→H | 0.4544 | 0.3867 | 0.0212 | 0.0218 | **0.4931** | 0.8832 | 0.9385 |

### C3. 结论

**光谱分支严重监督不足，融合门呈极端双峰分布（要么几乎关，要么几乎开）。**

- `spectral` 分支 OA 只有 30–59%，远低于 `context` 的 85–100%
- 中位数 reliability 只有 0.026–0.49，**大部分样本光谱分支被抑制**
- 原因：CE loss 主要监督 `logits = classifier(fused_feats)`，`spectral_logits` 只是辅助输出，没有独立 CE 监督

**这不是 bug，是设计选择**（提示词明确说"当前最终分类损失主要监督融合输出，并没有单独的光谱分支辅助 CE。这不一定是 Bug，需要先诊断"）。诊断结果是：**光谱分支确实监督不足，但当前版本不加入辅助损失，作为已知局限写进论文。**

---

## D. 三个数据集是否仍存在明显的类别条件负迁移？有没有必要加入可靠性驱动的类别条件光谱对齐？

### D1. per-class margin 变化（Houston，T-CST ACCEPT 时）

| seed | cycle | margin_worse | margin_better |
|---|---|---|---|
| 1237 | 1 | 无 | class0 |
| 1237 | 2 | 无 | class0, class5 |
| 1237 | 3 | 无 | class5 |
| 1237 | 4 | class0, class6 | class5 |
| 42 | 2 | class0 | class5 |
| 2024 | 1 | class2, class6 | class0, class5 |

**Houston 的 T-CST 整体改善 margin（better > worse），但 class0/class6 在部分 cycle 恶化。** 净收益 +0.00（因为 checkpoint 被丢弃）。

### D2. 目标域预测分布

| 数据集 | 主预测类 | 分布 | 熵 | 风险 |
|---|---|---|---|---|
| Houston | class5 (non_residential) | 0.37–0.79 | 0.33–0.66 | 偏向大类 |
| HyRANK | class6 (idx=6) | 0.24–0.50 | 0.56–0.79 | class1/class5 几乎 0 |
| S→H | **跨 seed/cycle 剧烈摆动** | Water/Land/Plant 之间跳 | 0.24–0.46 | **模式坍塌** |

**S→H 的摆动证据**：
- seed 1237: Water 0.58 → Plant 0.90
- seed 42: Land 0.61 → Water 0.58
- seed 2024: Land 0.55 → Plant 0.47

### D3. MMD 合成数据测试（证明边缘 MMD 有盲区）

| 场景 | 边缘 MMD | 类条件 MMD | 边缘/类条件 |
|---|---|---|---|
| S1: 条件相同/比例不同 | 0.0816 | **0.0022** | **37.57×**（边缘高估） |
| S2: 比例相同/条件偏移 | 0.0115 | **0.0827** | **0.14×**（边缘低估 7.2 倍） |
| S3: 两者都有 | 0.1553 | 0.0239 | 6.49×（比例主导） |

**这直接证明**：T-CST 用边缘 MMD 优化，无法保证类别条件对齐。Houston 的“MMD 降低但 class0/class3 margin 恶化”不是偶然，是边缘 MMD 的理论盲区。

### D4. 结论

**是，三个数据集都存在类别条件负迁移**：
- Houston：T-CST 改善大类（class5），class0/class6 在部分 cycle 恶化
- HyRANK：纯 CE cycles 让 class1/class5（源域 204/361 像素）在 target 上几乎 0
- S→H：模式坍塌，Water/Land/Plant 在不同 seed 下坍缩到不同类

**有必要加入可靠性驱动的类别条件光谱对齐**，但这是下一版工作，不在本轮范围（提示词明确说"本阶段先不修改 MMD 优化目标，不加入类别原型损失"）。

---

## 最终总结

| 问题 | 回答 |
|---|---|
| **A** | 实现 bug 修了 6 项（P0.3/P1.1/P1.2/P1.3/增强/P2），但修完后 T-CST 的 cycles 对三个数据集都没有正贡献（+0.00/-1.14/-0.61）。低准确率的主因是 T-CST 的 MMD 方向本身无效，不是实现 bug |
| **B** | **否**。Houston 的 T-CST 生效但被丢弃，HyRANK/S→H 全 REJECT。D 模式能让 Houston +2.21 但 S→H -13.30，不能统一 |
| **C** | **是，严重监督不足**。spectral OA 只有 30–59% vs context 85–100%，reliability 中位数 0.026–0.49，融合门大部分关闭 |
| **D** | **是，三个数据集都有类别条件负迁移**。合成数据证明边缘 MMD 在两个方向都偏离条件分布真实差异（S1 高估 37×，S2 低估 7.2×）。**有必要加入可靠性驱动的类别条件光谱对齐，作为下一版工作** |

## 最终数字（v8.1.0）

| 数据集 | OA | AA | Kappa | MacroF1 |
|---|---|---|---|---|
| Houston | 76.12±1.12 | 74.66±4.34 | 0.6008±0.0359 | 63.82±6.77 |
| HyRANK | 54.28±3.09 | 47.69±3.38 | 0.4522±0.0220 | 40.55±1.60 |
| S→H | 63.93±10.11 | 65.14±6.63 | 0.3522±0.1501 | 57.28±11.77 |

**B0 Source-only 对照**：
| 数据集 | OA | Kappa | MacroF1 |
|---|---|---|---|
| Houston | 76.12±1.12 | 0.6008±0.0359 | 63.82±6.77 |
| HyRANK | **55.42±1.48** | 0.4633±0.0072 | 42.01±0.99 |
| S→H | **64.54±9.30** | 0.3660±0.1366 | 58.34±10.89 |
