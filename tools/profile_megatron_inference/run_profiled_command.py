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
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from tools.profile_megatron_inference.collect_results import collect_profile_results


MEGATRON_LD_LIBRARY_PATH = (
    "/usr/local/cuda/targets/x86_64-linux/lib:"
    "/usr/local/cuda/lib64:"
    "/usr/local/cuda/lib:"
    "/usr/local/nvidia/lib64:"
    "/usr/local/nvidia/lib:"
    "/usr/lib/x86_64-linux-gnu"
)

PROFILE_WORKER_PATTERNS = {
    "none": "",
    "megatron-logprobs": "megatron_policy_worker",
    "megatron-all": "megatron_policy_worker",
    "e2e-async": (
        "megatron_policy_worker,"
        "vllm_generation_worker,"
        "vllm_async_generation_worker"
    ),
}


def build_profile_environment(
    *,
    profile: str,
    profile_range: str,
    base_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Return environment variables required by a profiling preset."""
    if profile not in PROFILE_WORKER_PATTERNS:
        raise ValueError(
            f"Unsupported PROFILE={profile}. "
            f"Expected one of: {', '.join(sorted(PROFILE_WORKER_PATTERNS))}"
        )
    env = dict(base_env or os.environ)
    worker_patterns = PROFILE_WORKER_PATTERNS[profile]
    if not worker_patterns:
        return env
    env["NRL_NSYS_WORKER_PATTERNS"] = worker_patterns
    env["NRL_NSYS_PROFILE_STEP_RANGE"] = profile_range
    if profile in {"megatron-logprobs", "megatron-all", "e2e-async"}:
        existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            MEGATRON_LD_LIBRARY_PATH
            if not existing_ld_library_path
            else f"{MEGATRON_LD_LIBRARY_PATH}:{existing_ld_library_path}"
        )
    return env


def _git_value(args: list[str]) -> Optional[str]:
    proc = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def write_metadata(
    *,
    metadata_path: Path,
    profile: str,
    profile_range: str,
    command: list[str],
    env: dict[str, str],
    result_dir: Path,
    slurm_log_dir: Optional[Path],
) -> None:
    """Write profiling run provenance before the child command starts."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "profile": profile,
        "profile_range": profile_range,
        "command": command,
        "cwd": os.getcwd(),
        "result_dir": str(result_dir),
        "slurm_log_dir": str(slurm_log_dir) if slurm_log_dir else None,
        "git_sha": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--short"])),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_submit_dir": env.get("SLURM_SUBMIT_DIR"),
        "worker_patterns": env.get("NRL_NSYS_WORKER_PATTERNS"),
        "profile_env_range": env.get("NRL_NSYS_PROFILE_STEP_RANGE"),
        "container": env.get("CONTAINER"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with a NeMo-RL profile preset")
    parser.add_argument("--profile", default="none")
    parser.add_argument("--profile-range", default="4:6")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--slurm-log-dir", type=Path, default=None)
    parser.add_argument("--post-run-log-sync-sleep", type=float, default=90.0)
    parser.add_argument("--no-nsys-stats", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command must be provided after --")
    return args


def main() -> None:
    args = parse_args()
    result_dir: Path = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = result_dir / "profile_metadata.json"
    env = build_profile_environment(
        profile=args.profile,
        profile_range=args.profile_range,
        base_env=dict(os.environ),
    )
    write_metadata(
        metadata_path=metadata_path,
        profile=args.profile,
        profile_range=args.profile_range,
        command=args.command,
        env=env,
        result_dir=result_dir,
        slurm_log_dir=args.slurm_log_dir,
    )

    proc = subprocess.run(args.command, env=env, check=False)
    if args.post_run_log_sync_sleep > 0:
        time.sleep(args.post_run_log_sync_sleep)

    try:
        collect_profile_results(
            result_dir=result_dir,
            slurm_log_dir=args.slurm_log_dir,
            profile=args.profile,
            profile_range=args.profile_range,
            metadata_path=metadata_path,
            run_nsys_stats=not args.no_nsys_stats,
        )
    except Exception as exc:
        print(f"[profile] Result collection failed: {exc}", flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
