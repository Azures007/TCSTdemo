# HFCST 三数据集诊断脚本（修正版）

此包只包含诊断脚本，不是完整 HFCST 训练源码包。

## 运行方式

在服务器上执行：

```bash
cd ~/autodl-tmp/TCSTdemo
python audit_three_datasets_corrected.py
```

如果结果目录不在默认位置，可显式指定：

```bash
python audit_three_datasets_corrected.py --root ~/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst
```

也可以指定报告输出目录：

```bash
python audit_three_datasets_corrected.py \
  --root ~/autodl-tmp/TCSTdemo/hfcst_work/results_hfcst \
  --out ~/autodl-tmp/TCSTdemo/hfcst_work/reports_hfcst
```

## 输出文件

- `three_dataset_summary.csv`：三任务 OA、AA、Macro-F1、Kappa、适应周期和回滚信息。
- `three_dataset_per_class.csv`：每个类别的准确率、support 和正确数。
- `three_dataset_source_split_coverage.csv`：train、constraint、calibration、validation 的类别覆盖。
- `three_dataset_diagnostic_report.md`：可阅读的完整报告。

## 修正点

- 优先使用 `events.jsonl` 中的 `run_complete` 事件读取目标域指标。
- 若找不到完整事件，再递归检查 `results.json`，不假定指标必须位于 JSON 顶层。
- 每个任务优先选择最新的已完成运行，避免新建但未完成的目录覆盖已完成结果。
- 只打印适应事件的 cycle、accepted、score、reasons，不输出庞大的 bootstrap 数组。
- 兼容审计字段 `source_split_class_counts_zero_based` 以及旧结构 `source_split.class_counts`。
- 只读现有结果，不启动训练、不修改配置和模型。
