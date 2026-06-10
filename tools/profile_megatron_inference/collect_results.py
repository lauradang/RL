# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.profile_megatron_inference.render_report import render_report


METRIC_PREFIXES = (
    "timing/train/get_logprobs/",
    "timing/train/get_reference_policy_logprobs/",
)
METRIC_NAMES = (
    "timing/train/logprob_inference_prep",
    "timing/train/policy_and_reference_logprobs",
    "timing/train/policy_training",
    "timing/train/weight_sync",
    "timing/train/total_step_time",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
)
NSYS_REPORTS = ("nvtxsum", "gpukernsum", "cudaapisum")


def parse_profile_range(profile_range: str) -> list[int]:
    """Parse a 1-indexed Nsight profile range like ``4:6`` into step ids."""
    start_text, stop_text = profile_range.split(":", 1)
    start = int(start_text.strip())
    stop = int(stop_text.strip())
    if start < 1 or start >= stop:
        raise ValueError(f"profile range must be non-empty and 1-indexed: {profile_range}")
    return list(range(start, stop))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("%", "").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_metrics(metrics_path: Path) -> dict[str, dict[int, float]]:
    """Load the JSON emitted by ``tests/json_dump_tb_logs.py``."""
    if not metrics_path.exists():
        return {}
    raw = json.loads(metrics_path.read_text())
    metrics: dict[str, dict[int, float]] = {}
    for metric_name, values_by_step in raw.items():
        if not isinstance(values_by_step, dict):
            continue
        converted: dict[int, float] = {}
        for step, value in values_by_step.items():
            numeric = _to_float(value)
            if numeric is not None:
                converted[int(step)] = numeric
        metrics[metric_name] = converted
    return metrics


def selected_metric_names(metrics: dict[str, dict[int, float]]) -> list[str]:
    """Return report-relevant metric names in stable order."""
    names = set(METRIC_NAMES)
    for metric_name in metrics:
        if metric_name.startswith(METRIC_PREFIXES):
            names.add(metric_name)
    return sorted(name for name in names if name in metrics)


def _metric_at(metrics: dict[str, dict[int, float]], metric_name: str, step: int) -> Optional[float]:
    return metrics.get(metric_name, {}).get(step)


def build_step_summaries(
    metrics: dict[str, dict[int, float]], profile_steps: list[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build per-step metric and derived summaries."""
    warnings: list[str] = []
    metric_names = selected_metric_names(metrics)
    missing_core_metrics = [name for name in METRIC_NAMES if name not in metrics]
    if missing_core_metrics:
        warnings.append(
            "Missing expected metrics: " + ", ".join(sorted(missing_core_metrics))
        )

    steps: list[dict[str, Any]] = []
    for step in profile_steps:
        step_metrics = {
            name: metrics[name][step]
            for name in metric_names
            if step in metrics.get(name, {})
        }
        prep = _metric_at(metrics, "timing/train/logprob_inference_prep", step)
        policy_ref = _metric_at(
            metrics, "timing/train/policy_and_reference_logprobs", step
        )
        total_step = _metric_at(metrics, "timing/train/total_step_time", step)
        envelope = None
        if prep is not None or policy_ref is not None:
            envelope = (prep or 0.0) + (policy_ref or 0.0)
        driver_submit = sum(
            value
            for value in (
                _metric_at(metrics, "timing/train/get_logprobs/shard_data", step),
                _metric_at(
                    metrics, "timing/train/get_logprobs/submit_logprob_futures", step
                ),
                _metric_at(
                    metrics, "timing/train/get_reference_policy_logprobs/shard_data", step
                ),
                _metric_at(
                    metrics,
                    "timing/train/get_reference_policy_logprobs/submit_reference_policy_logprob_futures",
                    step,
                ),
            )
            if value is not None
        )
        derived = {
            "megatron_logprob_envelope": envelope,
            "megatron_logprob_share": (
                envelope / total_step
                if envelope is not None and total_step not in (None, 0)
                else None
            ),
            "remote_logprob_wait_and_compute": (
                policy_ref - driver_submit if policy_ref is not None else None
            ),
        }
        steps.append({"step": step, "metrics": step_metrics, "derived": derived})
    return steps, warnings


def aggregate_step_values(steps: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate metric and derived values across profiled steps."""
    grouped: dict[str, list[float]] = {}
    for step in steps:
        for namespace in ("metrics", "derived"):
            for name, value in step[namespace].items():
                numeric = _to_float(value)
                if numeric is not None:
                    grouped.setdefault(name, []).append(numeric)

    aggregates: dict[str, dict[str, float]] = {}
    for name, values in grouped.items():
        aggregates[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return aggregates


def discover_nsys_reports(slurm_log_dir: Path) -> tuple[list[Path], list[str]]:
    """Find synced Nsight reports under the Ray log tree."""
    if not slurm_log_dir.exists():
        return [], [f"Slurm log directory not found: {slurm_log_dir}"]
    reports = sorted(slurm_log_dir.glob("ray/**/nsight/*.nsys-rep"))
    warnings = [] if reports else [f"No .nsys-rep files found under {slurm_log_dir}/ray"]
    return reports, warnings


def parse_nsys_csv(csv_text: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Parse Nsight CSV output, tolerating progress lines before the header."""
    lines = [line for line in csv_text.splitlines() if line.strip()]
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if "," in line and ("Name" in line or "Range" in line or "Time" in line)
        ),
        None,
    )
    if header_idx is None:
        return []

    reader = csv.DictReader(lines[header_idx:])
    rows: list[dict[str, Any]] = []
    for row in reader:
        clean_row = {
            (key or "").strip(): (value or "").strip() for key, value in row.items()
        }
        sort_value = _to_float(clean_row.get("Total Time (ns)"))
        if sort_value is None:
            for value in clean_row.values():
                sort_value = _to_float(value)
                if sort_value is not None:
                    break
        clean_row["_sort_value"] = sort_value or 0.0
        rows.append(clean_row)
    rows.sort(key=lambda row: row["_sort_value"], reverse=True)
    return rows[:limit]


def collect_nsys_stats(
    reports: list[Path],
    output_dir: Path,
    *,
    run_nsys_stats: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Run and parse ``nsys stats`` reports when available."""
    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = [
        {
            "path": str(report),
            "name": report.name,
            "size_bytes": report.stat().st_size,
        }
        for report in reports
    ]
    stats_by_report: dict[str, Any] = {}

    nsys_bin = shutil.which("nsys")
    if not run_nsys_stats:
        warnings.append("Skipping nsys stats by request.")
    elif nsys_bin is None:
        warnings.append("nsys was not found on PATH; report includes .nsys-rep inventory only.")
    else:
        for report in reports:
            parsed_reports: dict[str, Any] = {}
            for nsys_report in NSYS_REPORTS:
                stats_path = output_dir / f"{report.stem}.{nsys_report}.csv"
                proc = subprocess.run(
                    [
                        nsys_bin,
                        "stats",
                        "--report",
                        nsys_report,
                        "--format",
                        "csv",
                        str(report),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                combined_output = proc.stdout or proc.stderr
                stats_path.write_text(combined_output)
                if proc.returncode != 0:
                    warnings.append(
                        f"nsys stats failed for {report.name} ({nsys_report}); see {stats_path}"
                    )
                    continue
                parsed_reports[nsys_report] = {
                    "csv_path": str(stats_path),
                    "top_rows": parse_nsys_csv(combined_output),
                }
            stats_by_report[str(report)] = parsed_reports

    return {"reports": inventory, "stats": stats_by_report}, warnings


def _git_value(args: list[str], *, cwd: Path) -> Optional[str]:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def write_summary_csv(summary: dict[str, Any], output_path: Path) -> None:
    """Write a compact per-step CSV for spreadsheets."""
    metric_names = sorted(
        {
            name
            for step in summary["steps"]
            for name in list(step["metrics"]) + list(step["derived"])
        }
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *metric_names])
        writer.writeheader()
        for step in summary["steps"]:
            row = {"step": step["step"]}
            row.update(step["metrics"])
            row.update(
                {k: v for k, v in step["derived"].items() if v is not None}
            )
            writer.writerow(row)


def collect_profile_results(
    *,
    result_dir: Path,
    slurm_log_dir: Optional[Path],
    profile: str,
    profile_range: str,
    metadata_path: Optional[Path] = None,
    run_nsys_stats: bool = True,
) -> dict[str, Any]:
    """Collect metrics, Nsight inventory, stats, and render a static report."""
    result_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        log_dir = result_dir / "logs"
        if log_dir.exists():
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "tests/json_dump_tb_logs.py",
                    str(log_dir),
                    "--output_path",
                    str(metrics_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                warnings.append(
                    f"Could not create metrics.json from TensorBoard logs: {proc.stderr.strip()}"
                )
        else:
            warnings.append(f"No metrics.json or logs directory found under {result_dir}")

    profile_steps = parse_profile_range(profile_range)
    metrics = load_metrics(metrics_path)
    steps, metric_warnings = build_step_summaries(metrics, profile_steps)
    warnings.extend(metric_warnings)
    aggregates = aggregate_step_values(steps)

    metadata: dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    repo_root = Path(__file__).resolve().parents[2]
    metadata.setdefault("git_sha", _git_value(["rev-parse", "HEAD"], cwd=repo_root))
    metadata.setdefault(
        "git_dirty",
        bool(_git_value(["status", "--short"], cwd=repo_root)),
    )
    metadata.setdefault("profile", profile)
    metadata.setdefault("profile_range", profile_range)

    reports: list[Path] = []
    if slurm_log_dir is None:
        warnings.append("No Slurm log directory was provided; Nsight report discovery skipped.")
    else:
        reports, report_warnings = discover_nsys_reports(slurm_log_dir)
        warnings.extend(report_warnings)

    nsys, nsys_warnings = collect_nsys_stats(
        reports,
        result_dir / "nsys_stats",
        run_nsys_stats=run_nsys_stats,
    )
    warnings.extend(nsys_warnings)

    summary = {
        "schema_version": 1,
        "run": {
            **metadata,
            "result_dir": str(result_dir),
            "slurm_log_dir": str(slurm_log_dir) if slurm_log_dir else None,
        },
        "steps": steps,
        "aggregates": aggregates,
        "nsys": nsys,
        "warnings": warnings,
    }

    summary_path = result_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_summary_csv(summary, result_dir / "profile_summary.csv")
    render_report(summary, result_dir / "report.html")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Megatron inference profiling results")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--slurm-log-dir", type=Path, default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-range", required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--no-nsys-stats", action="store_true")
    parser.add_argument("--error-on-missing-nsys", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_profile_results(
        result_dir=args.result_dir,
        slurm_log_dir=args.slurm_log_dir,
        profile=args.profile,
        profile_range=args.profile_range,
        metadata_path=args.metadata_path,
        run_nsys_stats=not args.no_nsys_stats,
    )
    if args.error_on_missing_nsys and not summary["nsys"]["reports"]:
        raise SystemExit("No Nsight reports were found.")


if __name__ == "__main__":
    main()
