import json
from pathlib import Path

import pytest

from tools.profile_megatron_inference.collect_results import (
    aggregate_step_values,
    build_step_summaries,
    collect_profile_results,
    discover_nsys_reports,
    parse_nsys_csv,
    parse_profile_range,
)
from tools.profile_megatron_inference.render_report import render_report
from tools.profile_megatron_inference.run_profiled_command import (
    MEGATRON_LD_LIBRARY_PATH,
    build_profile_environment,
)


def test_profile_preset_env_mapping():
    env = build_profile_environment(
        profile="megatron-logprobs",
        profile_range="4:6",
        base_env={"LD_LIBRARY_PATH": "/existing"},
    )

    assert env["NRL_NSYS_WORKER_PATTERNS"] == "megatron_policy_worker"
    assert env["NRL_NSYS_PROFILE_STEP_RANGE"] == "4:6"
    assert env["LD_LIBRARY_PATH"].startswith(MEGATRON_LD_LIBRARY_PATH)
    assert env["LD_LIBRARY_PATH"].endswith(":/existing")


def test_e2e_async_preset_profiles_megatron_and_vllm_workers():
    env = build_profile_environment(
        profile="e2e-async",
        profile_range="3:5",
        base_env={},
    )

    assert env["NRL_NSYS_WORKER_PATTERNS"] == (
        "megatron_policy_worker,"
        "vllm_generation_worker,"
        "vllm_async_generation_worker"
    )


def test_invalid_profile_preset_raises():
    with pytest.raises(ValueError, match="Unsupported PROFILE"):
        build_profile_environment(
            profile="unknown",
            profile_range="4:6",
            base_env={},
        )


def test_metric_aggregation_and_derived_values():
    metrics = {
        "timing/train/logprob_inference_prep": {4: 1.0, 5: 2.0},
        "timing/train/policy_and_reference_logprobs": {4: 9.0, 5: 10.0},
        "timing/train/total_step_time": {4: 20.0, 5: 24.0},
        "timing/train/get_logprobs/shard_data": {4: 0.5, 5: 0.5},
        "timing/train/get_logprobs/submit_logprob_futures": {4: 0.25, 5: 0.25},
        "timing/train/get_reference_policy_logprobs/shard_data": {4: 0.5, 5: 0.5},
        "timing/train/get_reference_policy_logprobs/submit_reference_policy_logprob_futures": {
            4: 0.25,
            5: 0.25,
        },
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": {
            4: 100.0,
            5: 120.0,
        },
    }

    steps, warnings = build_step_summaries(metrics, parse_profile_range("4:6"))
    aggregates = aggregate_step_values(steps)

    assert not warnings or "policy_training" in warnings[0]
    assert steps[0]["derived"]["megatron_logprob_envelope"] == 10.0
    assert steps[0]["derived"]["megatron_logprob_share"] == 0.5
    assert steps[0]["derived"]["remote_logprob_wait_and_compute"] == 7.5
    assert aggregates["megatron_logprob_envelope"]["mean"] == 11.0
    assert (
        aggregates[
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
        ]["median"]
        == 110.0
    )


def test_discover_nsys_reports(tmp_path: Path):
    report_dir = tmp_path / "123-logs" / "ray" / "node" / "session" / "logs" / "nsight"
    report_dir.mkdir(parents=True)
    report = report_dir / "megatron_policy_worker_4:6_123.nsys-rep"
    report.write_bytes(b"rep")

    reports, warnings = discover_nsys_reports(tmp_path / "123-logs")

    assert reports == [report]
    assert warnings == []


def test_missing_nsys_reports_warns(tmp_path: Path):
    log_dir = tmp_path / "123-logs"
    log_dir.mkdir()

    reports, warnings = discover_nsys_reports(log_dir)

    assert reports == []
    assert "No .nsys-rep files found" in warnings[0]


def test_parse_nsys_csv_tolerates_preamble_and_sorts():
    csv_text = """Processing report
Time (%),Total Time (ns),Instances,Range
10.0,100,1,small
90.0,900,2,big
"""

    rows = parse_nsys_csv(csv_text)

    assert rows[0]["Range"] == "big"
    assert rows[0]["_sort_value"] == 900.0


def test_render_report_writes_static_html(tmp_path: Path):
    output = tmp_path / "report.html"
    summary = {
        "run": {"profile": "megatron-logprobs", "profile_range": "4:6"},
        "steps": [
            {
                "step": 4,
                "metrics": {"timing/train/policy_and_reference_logprobs": 9.0},
                "derived": {"megatron_logprob_share": 0.5},
            }
        ],
        "aggregates": {"megatron_logprob_share": {"mean": 0.5}},
        "nsys": {"reports": [], "stats": {}},
        "warnings": [],
    }

    render_report(summary, output)

    html = output.read_text()
    assert "Megatron Inference Profile" in html
    assert "profile-data" in html
    assert "megatron_logprob_share" in html


def test_collect_profile_results_without_nsys_reports(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "metrics.json").write_text(
        json.dumps(
            {
                "timing/train/logprob_inference_prep": {"4": 1.0},
                "timing/train/policy_and_reference_logprobs": {"4": 9.0},
                "timing/train/total_step_time": {"4": 20.0},
            }
        )
    )
    slurm_log_dir = tmp_path / "123-logs"
    slurm_log_dir.mkdir()

    summary = collect_profile_results(
        result_dir=result_dir,
        slurm_log_dir=slurm_log_dir,
        profile="megatron-logprobs",
        profile_range="4:5",
        run_nsys_stats=False,
    )

    assert (result_dir / "profile_summary.json").exists()
    assert (result_dir / "profile_summary.csv").exists()
    assert (result_dir / "report.html").exists()
    assert summary["steps"][0]["derived"]["megatron_logprob_envelope"] == 10.0
    assert any("No .nsys-rep files found" in warning for warning in summary["warnings"])
