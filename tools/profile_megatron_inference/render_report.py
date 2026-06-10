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
import html
import json
from pathlib import Path
from typing import Any


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_format_value(value)}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _aggregate_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for name, values in sorted(summary.get("aggregates", {}).items()):
        rows.append(
            [
                name,
                values.get("mean"),
                values.get("median"),
                values.get("min"),
                values.get("max"),
            ]
        )
    return rows


def _step_rows(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    metric_names = sorted(
        {
            name
            for step in summary.get("steps", [])
            for name in list(step.get("metrics", {}))
            + list(step.get("derived", {}))
        }
    )
    headers = ["step", *metric_names]
    rows = []
    for step in summary.get("steps", []):
        metrics = step.get("metrics", {})
        derived = step.get("derived", {})
        rows.append(
            [
                step.get("step"),
                *[metrics.get(name, derived.get(name)) for name in metric_names],
            ]
        )
    return headers, rows


def _nsys_inventory_rows(summary: dict[str, Any]) -> list[list[Any]]:
    return [
        [report.get("name"), report.get("size_bytes"), report.get("path")]
        for report in summary.get("nsys", {}).get("reports", [])
    ]


def _stats_sections(summary: dict[str, Any]) -> str:
    stats = summary.get("nsys", {}).get("stats", {})
    sections = []
    for report_path, reports in sorted(stats.items()):
        sections.append(f"<h3>{html.escape(report_path)}</h3>")
        for report_name, payload in sorted(reports.items()):
            rows = payload.get("top_rows", [])
            if not rows:
                continue
            keys = [key for key in rows[0].keys() if key != "_sort_value"]
            sections.append(f"<h4>{html.escape(report_name)}</h4>")
            sections.append(_table(keys, [[row.get(key, "") for key in keys] for row in rows]))
    if not sections:
        return "<p>No parsed Nsight stats are available.</p>"
    return "".join(sections)


def render_report(summary: dict[str, Any], output_path: Path) -> None:
    """Render a self-contained HTML profiling report."""
    run = summary.get("run", {})
    step_headers, step_rows = _step_rows(summary)
    warnings = summary.get("warnings", [])
    warning_html = (
        "<ul>" + "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings) + "</ul>"
        if warnings
        else "<p>No warnings.</p>"
    )
    summary_json = html.escape(json.dumps(summary, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Megatron Inference Profile</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .panel {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Megatron Inference Profile</h1>
  <div class="grid">
    <div class="panel"><strong>Profile</strong><br>{html.escape(str(run.get("profile", "")))}</div>
    <div class="panel"><strong>Range</strong><br>{html.escape(str(run.get("profile_range", "")))}</div>
    <div class="panel"><strong>Git SHA</strong><br><code>{html.escape(str(run.get("git_sha", "")))}</code></div>
    <div class="panel"><strong>Result Dir</strong><br><code>{html.escape(str(run.get("result_dir", "")))}</code></div>
  </div>

  <h2>Warnings</h2>
  {warning_html}

  <h2>Aggregate Metrics</h2>
  <div class="scroll">
  {_table(["metric", "mean", "median", "min", "max"], _aggregate_rows(summary))}
  </div>

  <h2>Profile Steps</h2>
  <div class="scroll">
  {_table(step_headers, step_rows)}
  </div>

  <h2>Nsight Reports</h2>
  <div class="scroll">
  {_table(["name", "size_bytes", "path"], _nsys_inventory_rows(summary))}
  </div>

  <h2>Parsed Nsight Stats</h2>
  <div class="scroll">
  {_stats_sections(summary)}
  </div>

  <script type="application/json" id="profile-data">{summary_json}</script>
</body>
</html>
"""
    output_path.write_text(document)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a profiling summary HTML report")
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_report(json.loads(args.summary_json.read_text()), args.output)


if __name__ == "__main__":
    main()
