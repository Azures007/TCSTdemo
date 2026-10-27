# HFCST 三数据集只读诊断报告

- 结果根目录：`/root/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst`
- 报告目录：`/root/autodl-tmp/TCSTdemo/hfcst_work/reports_hfcst`
- 目标域指标来自运行完成记录或结果文件；目标域 GT 仅用于最终事后评估时，不能据此声称它参与了模型选择。
- 类别索引同时列出 0 起始索引和 1 起始类别序号，避免编号混淆。

## 1. 三任务指标总览

| 任务 | 运行目录 | 状态 | 目标样本数 | OA | AA | Macro-F1 | Kappa | 提交周期 | 回滚次数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Houston13 -> Houston18 | `houston_splitfix_20261025_113646_s1237` | completed | 53200 | 54.86% | 45.20% | 28.52% | 0.2603 | 2 | 0 |

### Houston13 -> Houston18

- 运行目录：`/root/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst/houston_splitfix_20261025_113646_s1237`
- 指标来源：`events.jsonl:run_complete`
- 状态：`completed`
- 源域验证：n=386, OA=60.88%, AA=69.41%
- 适应：committed_cycles=2, rollback_count=0, stopped_reason=NA
- 训练集缺类（0 起始索引）：无
- 验证集缺类（0 起始索引）：无
- target_gt_used_for_selection：False

**紧凑适应事件：**
- cycle=0: accepted=True, score=1.1482->1.1415, reasons=无
- cycle=1: accepted=True, score=1.1375->1.1248, reasons=无

| HyRANK Dioni -> Loukia | `hyrank_restoredfastfix_v2` | completed | 12208 | 49.78% | 39.19% | 34.32% | 0.3942 | 0 | 1 |

### HyRANK Dioni -> Loukia

- 运行目录：`/root/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst/hyrank_restoredfastfix_v2`
- 指标来源：`events.jsonl:run_complete`
- 状态：`completed`
- 源域验证：n=1166, OA=61.41%, AA=59.28%
- 适应：committed_cycles=0, rollback_count=1, stopped_reason=post_training_acceptance_failed
- 训练集缺类（0 起始索引）：无
- 验证集缺类（0 起始索引）：1（第2类）
- target_gt_used_for_selection：False

**紧凑适应事件：**
- cycle=0: accepted=False, score=0.8681->0.8733, reasons=post_training_source_error_budget_exceeded, post_training_class_error_budget_exceeded
- rollback: cycle=0, rollback_count=1

| Shanghai -> Hangzhou | `shanghai_hangzhou_restoredfastfix_v2` | completed | 135700 | 71.18% | 69.88% | 65.76% | 0.4565 | 0 | 1 |

### Shanghai -> Hangzhou

- 运行目录：`/root/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst/shanghai_hangzhou_restoredfastfix_v2`
- 指标来源：`events.jsonl:run_complete`
- 状态：`completed`
- 源域验证：n=450, OA=96.00%, AA=96.00%
- 适应：committed_cycles=0, rollback_count=1, stopped_reason=post_training_acceptance_failed
- 训练集缺类（0 起始索引）：无
- 验证集缺类（0 起始索引）：无
- target_gt_used_for_selection：False

**紧凑适应事件：**
- cycle=0: accepted=False, score=0.4539->0.4164, reasons=post_training_block_target_pseudo_flip_budget_exceeded
- rollback: cycle=0, rollback_count=1

## 2. 每类目标域准确率

| 任务 | 类别索引（0 起） | 类别序号（1 起） | Support | 正确数 | 类别准确率 |
|---|---:|---:|---:|---:|---:|
| Houston13 -> Houston18 | 0 | 1 | 1353 | 10 | 0.74% |
| Houston13 -> Houston18 | 1 | 2 | 4888 | 2 | 0.04% |
| Houston13 -> Houston18 | 2 | 3 | 2766 | 454 | 16.41% |
| Houston13 -> Houston18 | 3 | 4 | 22 | 22 | 100.00% |
| Houston13 -> Houston18 | 4 | 5 | 5347 | 2676 | 50.05% |
| Houston13 -> Houston18 | 5 | 6 | 32459 | 20560 | 63.34% |
| Houston13 -> Houston18 | 6 | 7 | 6365 | 5463 | 85.83% |
| HyRANK Dioni -> Loukia | 0 | 1 | 288 | 44 | 15.28% |
| HyRANK Dioni -> Loukia | 1 | 2 | 67 | NA | 0.00% |
| HyRANK Dioni -> Loukia | 2 | 3 | 542 | 178 | 32.84% |
| HyRANK Dioni -> Loukia | 3 | 4 | 79 | 37 | 46.84% |
| HyRANK Dioni -> Loukia | 4 | 5 | 1401 | NA | 0.00% |
| HyRANK Dioni -> Loukia | 5 | 6 | 500 | NA | 0.00% |
| HyRANK Dioni -> Loukia | 6 | 7 | 3793 | 3229 | 85.13% |
| HyRANK Dioni -> Loukia | 7 | 8 | 2803 | 453 | 16.16% |
| HyRANK Dioni -> Loukia | 8 | 9 | 404 | 359 | 88.86% |
| HyRANK Dioni -> Loukia | 9 | 10 | 487 | NA | 0.00% |
| HyRANK Dioni -> Loukia | 10 | 11 | 1393 | 1393 | 100.00% |
| HyRANK Dioni -> Loukia | 11 | 12 | 451 | 384 | 85.14% |
| Shanghai -> Hangzhou | 0 | 1 | 18043 | 17193 | 95.29% |
| Shanghai -> Hangzhou | 1 | 2 | 77450 | 69493 | 89.73% |
| Shanghai -> Hangzhou | 2 | 3 | 40207 | 9900 | 24.62% |

## 3. 源域类别覆盖

| 任务 | 划分 | 缺失类别索引（0 起） | 缺失类别序号（1 起） |
|---|---|---|---|
| Houston13 -> Houston18 | train | 无 | 无 |
| Houston13 -> Houston18 | constraint | 0;3 | 1;4 |
| Houston13 -> Houston18 | calibration | 无 | 无 |
| Houston13 -> Houston18 | validation | 无 | 无 |
| HyRANK Dioni -> Loukia | train | 无 | 无 |
| HyRANK Dioni -> Loukia | constraint | 5 | 6 |
| HyRANK Dioni -> Loukia | calibration | 1;3;9 | 2;4;10 |
| HyRANK Dioni -> Loukia | validation | 1 | 2 |
| Shanghai -> Hangzhou | train | 无 | 无 |
| Shanghai -> Hangzhou | constraint | 无 | 无 |
| Shanghai -> Hangzhou | calibration | 无 | 无 |
| Shanghai -> Hangzhou | validation | 无 | 无 |

## 4. 指标解释与注意事项

- OA、AA、Macro-F1、Kappa 均按运行产物中记录的值输出，不重新推断或改写。
- 每类准确率优先读取 `per_class_accuracy`；若缺失则根据混淆矩阵对角线 / 对应真实类别行和计算。
- `support` 是混淆矩阵中对应真实类别行的样本数。
- 源域类别索引按审计字段 `source_split_class_counts_zero_based` 的约定处理；表中同时列出 0 起始索引和 1 起始类别序号。
- 缺失类别覆盖是数据划分诊断，不等于单独证明模型准确率低的唯一原因。
- 本脚本不启动训练，不读取目标域标签来选择模型，也不修改任何训练产物；只汇总现有运行记录。
