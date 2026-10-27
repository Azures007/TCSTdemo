#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读汇总 HFCST 三个跨域任务的最新已完成运行：
- 目标域 OA / AA / Macro-F1 / Kappa
- 每类准确率、每类 support
- 源域 train / constraint / calibration / validation 类别覆盖
- 适应提交次数、回滚次数、停止原因
- 紧凑打印适应接受/拒绝事件，不打印 bootstrap 数组

默认从以下目录读取：
  hfcst_work/results_hfcst/

只读取已有产物；不会启动训练、修改配置或修改模型文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


TASKS = [
    {"key": "houston", "label": "Houston13 -> Houston18", "num_classes": 7},
    {"key": "hyrank", "label": "HyRANK Dioni -> Loukia", "num_classes": 12},
    {"key": "shanghai", "label": "Shanghai -> Hangzhou", "num_classes": 3},
]

SPLITS = ("train", "constraint", "calibration", "validation")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def event_run_complete(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [e for e in events if e.get("event") == "run_complete"]
    return matches[-1] if matches else None


def classify_task(run_dir: Path, events: list[dict[str, Any]]) -> str | None:
    parts = [run_dir.name.lower()]
    for event in events:
        for key in ("task", "config", "final_checkpoint", "data_audit"):
            value = event.get(key)
            if isinstance(value, str):
                parts.append(value.lower())
    text = " ".join(parts)

    # Check Shanghai first so no path fragments can accidentally match another task.
    if "shanghai" in text and "hangzhou" in text:
        return "shanghai"
    if "hyrank" in text or "dioni" in text or "loukia" in text:
        return "hyrank"
    if "houston" in text:
        return "houston"
    return None


def resolve_root(raw_root: str | None) -> Path:
    if raw_root:
        root = Path(raw_root).expanduser()
        if root.is_dir():
            return root.resolve()
        raise FileNotFoundError(f"结果目录不存在：{root}")

    cwd = Path.cwd()
    candidates = [
        cwd / "hfcst_work" / "results_hfcst",
        cwd / "results_hfcst",
        cwd.parent / "hfcst_work" / "results_hfcst",
        Path.home() / "autodl-tmp" / "TCSTdemo" / "hfcst_work" / "results_hfcst",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "找不到 results_hfcst 目录。请在 TCSTdemo 目录运行，或使用 --root 指定路径：\n"
        + tried
    )


def artifact_mtime(run_dir: Path) -> float:
    candidates = [
        run_dir / "events.jsonl",
        run_dir / "results.json",
        run_dir / "data_audit.json",
        run_dir / "final.pt",
    ]
    times = []
    for path in candidates:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(times, default=0.0)


def find_latest_run(root: Path, task_key: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        events = read_events(run_dir / "events.jsonl")
        detected = classify_task(run_dir, events)
        if detected != task_key:
            continue
        candidates.append(
            {
                "run_dir": run_dir,
                "events": events,
                "run_complete": event_run_complete(events),
                "mtime": artifact_mtime(run_dir),
            }
        )

    if not candidates:
        return None

    # Prefer the most recent completed run; do not let a newer failed/incomplete
    # directory silently replace a valid completed run. If none completed, show
    # the most recently modified candidate and mark it as incomplete.
    completed = [c for c in candidates if c["run_complete"] is not None]
    pool = completed if completed else candidates
    return max(pool, key=lambda c: c["mtime"])


def get_ci(mapping: Any, *names: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def recursively_find_target_metrics(obj: Any) -> dict[str, Any] | None:
    """Find a target-metric dictionary in results.json without assuming its nesting."""
    if isinstance(obj, dict):
        oa = get_ci(obj, "OA")
        aa = get_ci(obj, "AA")
        kappa = get_ci(obj, "Kappa")
        has_class_data = (
            get_ci(obj, "per_class_accuracy") is not None
            or get_ci(obj, "confusion_matrix") is not None
        )
        if oa is not None and aa is not None and kappa is not None and has_class_data:
            return obj
        # Prefer the specifically named final target metric block.
        for key in (
            "target_metrics_final_only",
            "final_target_metrics",
            "target_metrics",
        ):
            value = obj.get(key)
            found = recursively_find_target_metrics(value)
            if found is not None:
                return found
        for value in obj.values():
            found = recursively_find_target_metrics(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursively_find_target_metrics(value)
            if found is not None:
                return found
    return None


def get_target_metrics(run: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    complete = run.get("run_complete")
    if isinstance(complete, dict):
        for key in (
            "target_metrics_final_only",
            "final_target_metrics",
            "target_metrics",
        ):
            value = complete.get(key)
            if isinstance(value, dict):
                return value, "events.jsonl:run_complete"

    result_path = run["run_dir"] / "results.json"
    result_obj = read_json(result_path)
    if result_obj is not None:
        found = recursively_find_target_metrics(result_obj)
        if found is not None:
            return found, "results.json（递归查找）"

    # Last fallback: some older runs may have final metrics in another event.
    for event in reversed(run.get("events", [])):
        for key in (
            "target_metrics_final_only",
            "final_target_metrics",
            "target_metrics",
        ):
            value = event.get(key)
            if isinstance(value, dict):
                return value, f"events.jsonl:{event.get('event', 'unknown')}"

    return None, "未找到目标域指标"


def get_baseline_metrics(run: dict[str, Any]) -> dict[str, Any]:
    complete = run.get("run_complete")
    if isinstance(complete, dict):
        value = complete.get("baseline_source_validation")
        if isinstance(value, dict):
            return value
    for event in reversed(run.get("events", [])):
        value = event.get("baseline_source_validation")
        if isinstance(value, dict):
            return value
    return {}


def normalize_split_counts(audit: dict[str, Any] | None) -> dict[str, dict[int, int]]:
    if not audit:
        return {}

    raw = audit.get("source_split_class_counts_zero_based")
    if not isinstance(raw, dict) or not raw:
        source_split = audit.get("source_split", {})
        if isinstance(source_split, dict):
            raw = source_split.get("class_counts_zero_based")
            if not isinstance(raw, dict) or not raw:
                raw = source_split.get("class_counts")

    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[int, int]] = {}
    for split, counts in raw.items():
        if isinstance(counts, list):
            normalized[str(split)] = {
                i: int(value) for i, value in enumerate(counts)
            }
        elif isinstance(counts, dict):
            parsed: dict[int, int] = {}
            for class_key, value in counts.items():
                try:
                    parsed[int(class_key)] = int(value)
                except (TypeError, ValueError):
                    continue
            normalized[str(split)] = parsed
    return normalized


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    number = to_float(value)
    return "NA" if number is None else f"{number * 100:.2f}%"


def fmt_cell(value: Any) -> str:
    # Do not use truthiness here: numeric 0 is a valid value (e.g. zero rollbacks).
    return "NA" if value is None or value == "" else str(value)


def fmt_classes(indices: list[int]) -> str:
    if not indices:
        return "无"
    return ", ".join(f"{i}（第{i + 1}类）" for i in indices)


def get_class_rows(
    task_label: str,
    run_dir: str,
    num_classes: int,
    metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not metrics:
        return []

    per_class = get_ci(metrics, "per_class_accuracy")
    confusion = get_ci(metrics, "confusion_matrix")
    rows: list[dict[str, Any]] = []

    for i in range(num_classes):
        support = None
        correct = None
        accuracy = None

        if isinstance(confusion, list) and i < len(confusion):
            row = confusion[i]
            if isinstance(row, list):
                try:
                    support = int(sum(float(v) for v in row))
                    if i < len(row):
                        correct = int(float(row[i]))
                    if support > 0 and correct is not None:
                        accuracy = correct / support
                except (TypeError, ValueError):
                    support = correct = None
                    accuracy = None

        if isinstance(per_class, list) and i < len(per_class):
            value = to_float(per_class[i])
            if value is not None:
                accuracy = value

        rows.append(
            {
                "task": task_label,
                "run_dir": run_dir,
                "class_index_zero_based": i,
                "class_number_one_based": i + 1,
                "support": "" if support is None else support,
                "correct": "" if correct is None else correct,
                "class_accuracy": "" if accuracy is None else accuracy,
                "class_accuracy_percent": "" if accuracy is None else accuracy * 100,
            }
        )
    return rows


def compact_event_lines(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        kind = event.get("event")
        if kind == "posterior_acceptance":
            cycle = event.get("cycle", "?")
            accepted = event.get("accepted", "?")
            reasons = event.get("reasons") or []
            reason_text = ", ".join(str(x) for x in reasons) if reasons else "无"
            pre = fmt_num(event.get("pre_transport_score"))
            post = fmt_num(event.get("post_transport_score"))
            lines.append(
                f"cycle={cycle}: accepted={accepted}, "
                f"score={pre}->{post}, reasons={reason_text}"
            )
        elif kind == "transaction_rollback":
            lines.append(
                f"rollback: cycle={event.get('cycle', '?')}, "
                f"rollback_count={event.get('rollback_count', '?')}"
            )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读汇总 HFCST 三个跨域任务的最新已完成运行。"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="结果目录，默认自动寻找 hfcst_work/results_hfcst",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="报告输出目录，默认写到 results_hfcst 的上一级 reports_hfcst",
    )
    args = parser.parse_args()

    try:
        root = resolve_root(args.root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve() if args.out else root.parent / "reports_hfcst"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    report_lines = [
        "# HFCST 三数据集只读诊断报告",
        "",
        f"- 结果根目录：`{root}`",
        f"- 报告目录：`{out_dir}`",
        "- 目标域指标来自运行完成记录或结果文件；目标域 GT 仅用于最终事后评估时，不能据此声称它参与了模型选择。",
        "- 类别索引同时列出 0 起始索引和 1 起始类别序号，避免编号混淆。",
        "",
        "## 1. 三任务指标总览",
        "",
        "| 任务 | 运行目录 | 状态 | 目标样本数 | OA | AA | Macro-F1 | Kappa | 提交周期 | 回滚次数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    missing_task_count = 0

    for task in TASKS:
        run = find_latest_run(root, task["key"])
        if run is None:
            missing_task_count += 1
            row = {
                "task": task["label"],
                "run_dir": "",
                "run_status": "未找到运行产物",
                "target_n": "",
                "OA": "",
                "AA": "",
                "MacroF1": "",
                "Kappa": "",
                "committed_cycles": "",
                "rollback_count": "",
                "adaptation_stopped_reason": "",
                "metrics_source": "",
                "source_validation_n": "",
                "source_validation_OA": "",
                "source_validation_AA": "",
                "missing_train_classes_zero_based": "",
                "missing_validation_classes_zero_based": "",
                "target_gt_used_for_selection": "",
            }
            summary_rows.append(row)
            report_lines.append(
                f"| {task['label']} | 未找到 | 未找到 | NA | NA | NA | NA | NA | NA | NA |"
            )
            continue

        run_dir: Path = run["run_dir"]
        events: list[dict[str, Any]] = run["events"]
        complete = run.get("run_complete")
        metrics, metrics_source = get_target_metrics(run)
        baseline = get_baseline_metrics(run)
        audit = read_json(run_dir / "data_audit.json")
        split_counts = normalize_split_counts(audit)

        target_n = get_ci(metrics, "n") if metrics else None
        oa = get_ci(metrics, "OA") if metrics else None
        aa = get_ci(metrics, "AA") if metrics else None
        macro_f1 = get_ci(metrics, "MacroF1", "Macro-F1") if metrics else None
        kappa = get_ci(metrics, "Kappa") if metrics else None

        committed = complete.get("committed_cycles") if complete else None
        rollback = complete.get("rollback_count") if complete else None
        stop_reason = complete.get("adaptation_stopped_reason") if complete else None
        target_gt_used = (
            complete.get("target_gt_used_for_selection")
            if complete else None
        )
        if target_gt_used is None and metrics:
            target_gt_used = get_ci(metrics, "selection_used_target_gt")

        missing_by_split: dict[str, list[int]] = {}
        for split in SPLITS:
            counts = split_counts.get(split, {})
            missing = [i for i in range(task["num_classes"]) if counts.get(i, 0) == 0]
            missing_by_split[split] = missing
            coverage_rows.append(
                {
                    "task": task["label"],
                    "run_dir": str(run_dir),
                    "split": split,
                    "class_counts_zero_based_json": json.dumps(
                        {str(k): counts.get(k, 0) for k in range(task["num_classes"])},
                        ensure_ascii=False,
                    ) if split in split_counts else "",
                    "missing_classes_zero_based": (
                        (";".join(map(str, missing)) if missing else "无")
                        if split in split_counts else "NA"
                    ),
                    "missing_classes_one_based": (
                        (";".join(str(i + 1) for i in missing) if missing else "无")
                        if split in split_counts else "NA"
                    ),
                }
            )

        status = "completed" if complete else "no_run_complete_event"
        summary_row = {
            "task": task["label"],
            "run_dir": str(run_dir),
            "run_status": status,
            "target_n": "" if target_n is None else target_n,
            "OA": "" if oa is None else oa,
            "AA": "" if aa is None else aa,
            "MacroF1": "" if macro_f1 is None else macro_f1,
            "Kappa": "" if kappa is None else kappa,
            "committed_cycles": "" if committed is None else committed,
            "rollback_count": "" if rollback is None else rollback,
            "adaptation_stopped_reason": "" if stop_reason is None else stop_reason,
            "metrics_source": metrics_source,
            "source_validation_n": baseline.get("n", ""),
            "source_validation_OA": baseline.get("oa", ""),
            "source_validation_AA": baseline.get("aa", ""),
            "missing_train_classes_zero_based": ";".join(map(str, missing_by_split.get("train", []))),
            "missing_validation_classes_zero_based": ";".join(map(str, missing_by_split.get("validation", []))),
            "target_gt_used_for_selection": "" if target_gt_used is None else target_gt_used,
        }
        summary_rows.append(summary_row)

        report_lines.append(
            f"| {task['label']} | `{run_dir.name}` | {status} | "
            f"{target_n if target_n is not None else 'NA'} | {fmt_pct(oa)} | "
            f"{fmt_pct(aa)} | {fmt_pct(macro_f1)} | {fmt_num(kappa)} | "
            f"{committed if committed is not None else 'NA'} | "
            f"{rollback if rollback is not None else 'NA'} |"
        )

        report_lines.extend(
            [
                "",
                f"### {task['label']}",
                "",
                f"- 运行目录：`{run_dir}`",
                f"- 指标来源：`{metrics_source}`",
                f"- 状态：`{status}`",
                f"- 源域验证：n={baseline.get('n', 'NA')}, OA={fmt_pct(baseline.get('oa'))}, AA={fmt_pct(baseline.get('aa'))}",
                f"- 适应：committed_cycles={committed if committed is not None else 'NA'}, rollback_count={rollback if rollback is not None else 'NA'}, stopped_reason={stop_reason if stop_reason is not None else 'NA'}",
                f"- 训练集缺类（0 起始索引）：{fmt_classes(missing_by_split.get('train', [])) if split_counts else '没有可用审计数据'}",
                f"- 验证集缺类（0 起始索引）：{fmt_classes(missing_by_split.get('validation', [])) if split_counts else '没有可用审计数据'}",
                f"- target_gt_used_for_selection：{target_gt_used if target_gt_used is not None else 'NA'}",
                "",
                "**紧凑适应事件：**",
            ]
        )
        compact = compact_event_lines(events)
        if compact:
            report_lines.extend(f"- {line}" for line in compact)
        else:
            report_lines.append("- 没有 posterior_acceptance / transaction_rollback 事件。")
        report_lines.append("")

        class_rows.extend(
            get_class_rows(task["label"], str(run_dir), task["num_classes"], metrics)
        )

    summary_fields = [
        "task", "run_dir", "run_status", "target_n", "OA", "AA", "MacroF1", "Kappa",
        "committed_cycles", "rollback_count", "adaptation_stopped_reason", "metrics_source",
        "source_validation_n", "source_validation_OA", "source_validation_AA",
        "missing_train_classes_zero_based", "missing_validation_classes_zero_based",
        "target_gt_used_for_selection",
    ]
    class_fields = [
        "task", "run_dir", "class_index_zero_based", "class_number_one_based",
        "support", "correct", "class_accuracy", "class_accuracy_percent",
    ]
    coverage_fields = [
        "task", "run_dir", "split", "class_counts_zero_based_json",
        "missing_classes_zero_based", "missing_classes_one_based",
    ]

    write_csv(out_dir / "three_dataset_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "three_dataset_per_class.csv", class_rows, class_fields)
    write_csv(out_dir / "three_dataset_source_split_coverage.csv", coverage_rows, coverage_fields)

    report_lines.extend(
        [
            "## 2. 每类目标域准确率",
            "",
            "| 任务 | 类别索引（0 起） | 类别序号（1 起） | Support | 正确数 | 类别准确率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        acc = row["class_accuracy"]
        report_lines.append(
            f"| {row['task']} | {row['class_index_zero_based']} | "
            f"{row['class_number_one_based']} | {row['support'] or 'NA'} | "
            f"{row['correct'] or 'NA'} | {fmt_pct(acc if acc != '' else None)} |"
        )

    report_lines.extend(
        [
            "",
            "## 3. 源域类别覆盖",
            "",
            "| 任务 | 划分 | 缺失类别索引（0 起） | 缺失类别序号（1 起） |",
            "|---|---|---|---|",
        ]
    )
    for row in coverage_rows:
        report_lines.append(
            f"| {row['task']} | {row['split']} | "
            f"{row['missing_classes_zero_based'] or 'NA'} | "
            f"{row['missing_classes_one_based'] or 'NA'} |"
        )

    report_lines.extend(
        [
            "",
            "## 4. 指标解释与注意事项",
            "",
            "- OA、AA、Macro-F1、Kappa 均按运行产物中记录的值输出，不重新推断或改写。",
            "- 每类准确率优先读取 `per_class_accuracy`；若缺失则根据混淆矩阵对角线 / 对应真实类别行和计算。",
            "- `support` 是混淆矩阵中对应真实类别行的样本数。",
            "- 源域类别索引按审计字段 `source_split_class_counts_zero_based` 的约定处理；表中同时列出 0 起始索引和 1 起始类别序号。",
            "- 缺失类别覆盖是数据划分诊断，不等于单独证明模型准确率低的唯一原因。",
            "- 本脚本不启动训练，不读取目标域标签来选择模型，也不修改任何训练产物；只汇总现有运行记录。",
            "",
        ]
    )

    report_path = out_dir / "three_dataset_diagnostic_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 90)
    print("HFCST 三数据集只读诊断")
    print("结果目录:", root)
    print("报告目录:", out_dir)
    print()
    print("任务 | 状态 | N | OA | AA | Macro-F1 | Kappa | committed | rollback")
    print("-" * 90)
    for row in summary_rows:
        print(
            f"{row['task']} | {row['run_status']} | {fmt_cell(row['target_n'])} | "
            f"{fmt_pct(row['OA'])} | {fmt_pct(row['AA'])} | "
            f"{fmt_pct(row['MacroF1'])} | {fmt_num(row['Kappa'])} | "
            f"{fmt_cell(row['committed_cycles'])} | {fmt_cell(row['rollback_count'])}"
        )

    print("\n每类目标域准确率（百分比）")
    print("任务 | 类别索引(0起) | 类别序号(1起) | support | accuracy")
    for row in class_rows:
        print(
            f"{row['task']} | {row['class_index_zero_based']} | "
            f"{row['class_number_one_based']} | {row['support'] or 'NA'} | "
            f"{fmt_pct(row['class_accuracy'] if row['class_accuracy'] != '' else None)}"
        )

    print("\n源域划分缺失类别（0 起始索引；括号内为 1 起始类别序号）")
    for row in coverage_rows:
        zero = row["missing_classes_zero_based"]
        one = row["missing_classes_one_based"]
        print(f"{row['task']} | {row['split']} | zero_based={zero} | one_based={one}")

    print("\n已生成报告：")
    print(" -", out_dir / "three_dataset_summary.csv")
    print(" -", out_dir / "three_dataset_per_class.csv")
    print(" -", out_dir / "three_dataset_source_split_coverage.csv")
    print(" -", report_path)
    if missing_task_count:
        print(f"\n[WARNING] 有 {missing_task_count} 个任务没有找到运行产物；请核对 --root 路径。")
    print("\n完成：只读检查，没有启动训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
