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

"""Build or list SWE-Gym Apptainer SIF images for NeMo-Gym SWE-agent runs."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_NAMESPACE = "xingyaoww/sweb.eval.x86_64"
DEFAULT_IMAGE_TAG = "latest"
DEFAULT_FILENAME_PREFIX = "xingyaoww_sweb.eval.x86_64"
DEFAULT_ID_REPLACEMENT = "_s_"


class SifCacheError(ValueError):
    """Raised when a SWE-Gym SIF cache manifest cannot be prepared."""


@dataclass(frozen=True)
class SifBuildSpec:
    instance_id: str
    docker_id: str
    sif_path: Path
    image_uri: str


def _loads_instance_dict(value: Any, line_number: int) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SifCacheError(
                f"line {line_number}: metadata.instance_dict is not valid JSON"
            ) from exc
        if not isinstance(loaded, dict):
            raise SifCacheError(
                f"line {line_number}: metadata.instance_dict must decode to an object"
            )
        return loaded
    raise SifCacheError(
        f"line {line_number}: metadata.instance_dict must be a JSON string or object"
    )


def extract_instance_id(row: dict[str, Any], line_number: int) -> str:
    if not isinstance(row, dict):
        raise SifCacheError(f"line {line_number}: row must be a JSON object")

    responses_create_params = row.get("responses_create_params")
    if isinstance(responses_create_params, dict):
        metadata = responses_create_params.get("metadata")
        if isinstance(metadata, dict):
            if "instance_dict" in metadata:
                instance_dict = _loads_instance_dict(
                    metadata["instance_dict"], line_number
                )
                instance_id = instance_dict.get("instance_id")
                if isinstance(instance_id, str) and instance_id.strip():
                    return instance_id
            instance_id = metadata.get("instance_id")
            if isinstance(instance_id, str) and instance_id.strip():
                return instance_id

    instance_id = row.get("instance_id")
    if isinstance(instance_id, str) and instance_id.strip():
        return instance_id

    raise SifCacheError(f"line {line_number}: missing instance_id")


def make_docker_id(
    instance_id: str, *, id_replacement: str, preserve_case: bool
) -> str:
    docker_id = instance_id.replace("__", id_replacement)
    if not preserve_case:
        docker_id = docker_id.lower()
    return docker_id


def make_sif_build_spec(
    instance_id: str,
    *,
    output_dir: Path,
    image_namespace: str,
    image_tag: str,
    filename_prefix: str,
    id_replacement: str,
    preserve_case: bool,
) -> SifBuildSpec:
    docker_id = make_docker_id(
        instance_id, id_replacement=id_replacement, preserve_case=preserve_case
    )
    image_repo = f"{image_namespace.rstrip('.')}.{docker_id}"
    return SifBuildSpec(
        instance_id=instance_id,
        docker_id=docker_id,
        sif_path=output_dir / f"{filename_prefix.rstrip('.')}.{docker_id}.sif",
        image_uri=f"docker://{image_repo}:{image_tag}",
    )


def iter_sif_build_specs(
    input_path: Path,
    *,
    output_dir: Path,
    image_namespace: str = DEFAULT_IMAGE_NAMESPACE,
    image_tag: str = DEFAULT_IMAGE_TAG,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    id_replacement: str = DEFAULT_ID_REPLACEMENT,
    preserve_case: bool = False,
    limit: int | None = None,
) -> list[SifBuildSpec]:
    specs: list[SifBuildSpec] = []
    seen_paths: set[Path] = set()
    row_count = 0

    with input_path.open() as src:
        for line_number, line in enumerate(src, start=1):
            if limit is not None and row_count >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SifCacheError(f"line {line_number}: invalid JSON") from exc

            instance_id = extract_instance_id(row, line_number)
            spec = make_sif_build_spec(
                instance_id,
                output_dir=output_dir,
                image_namespace=image_namespace,
                image_tag=image_tag,
                filename_prefix=filename_prefix,
                id_replacement=id_replacement,
                preserve_case=preserve_case,
            )
            if spec.sif_path not in seen_paths:
                specs.append(spec)
                seen_paths.add(spec.sif_path)
            row_count += 1

    return specs


def shard_specs(
    specs: list[SifBuildSpec], *, shard_index: int | None, num_shards: int | None
) -> list[SifBuildSpec]:
    if shard_index is None and num_shards is None:
        return specs
    if shard_index is None or num_shards is None:
        raise SifCacheError("--shard-index and --num-shards must be set together")
    if num_shards <= 0:
        raise SifCacheError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise SifCacheError("--shard-index must satisfy 0 <= index < num_shards")
    return [spec for index, spec in enumerate(specs) if index % num_shards == shard_index]


def write_manifest(specs: list[SifBuildSpec], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as manifest:
        for spec in specs:
            manifest.write(
                f"{spec.sif_path}\t{spec.image_uri}\t"
                f"{spec.instance_id}\t{spec.docker_id}\n"
            )


def build_sif(
    spec: SifBuildSpec, *, apptainer: str, force: bool
) -> tuple[str, SifBuildSpec, str]:
    if spec.sif_path.exists() and spec.sif_path.stat().st_size > 0 and not force:
        return "skipped", spec, "already exists"

    spec.sif_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [apptainer, "build", str(spec.sif_path), spec.image_uri]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return "built", spec, ""

    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"exit code {result.returncode}"
    return "failed", spec, message


def build_specs(
    specs: list[SifBuildSpec], *, apptainer: str, force: bool, max_workers: int
) -> tuple[int, int, int]:
    built = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(build_sif, spec, apptainer=apptainer, force=force)
            for spec in specs
        ]
        for future in as_completed(futures):
            status, spec, message = future.result()
            if status == "built":
                built += 1
                print(f"built\t{spec.sif_path}", file=sys.stderr)
            elif status == "skipped":
                skipped += 1
                print(f"skipped\t{spec.sif_path}\t{message}", file=sys.stderr)
            else:
                failed += 1
                print(f"failed\t{spec.sif_path}\t{message}", file=sys.stderr)

    return built, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the SWE-Gym train SIF cache used by NeMo-Gym SWE-agent. "
            "By default this only writes a manifest; pass --build to run "
            "apptainer build for each source Docker image."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--image-namespace", default=DEFAULT_IMAGE_NAMESPACE)
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG)
    parser.add_argument("--filename-prefix", default=DEFAULT_FILENAME_PREFIX)
    parser.add_argument("--id-replacement", default=DEFAULT_ID_REPLACEMENT)
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Do not lowercase Docker image and SIF instance IDs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_workers <= 0:
        print("error: --max-workers must be positive", file=sys.stderr)
        return 1

    try:
        specs = iter_sif_build_specs(
            args.input,
            output_dir=args.output_dir,
            image_namespace=args.image_namespace,
            image_tag=args.image_tag,
            filename_prefix=args.filename_prefix,
            id_replacement=args.id_replacement,
            preserve_case=args.preserve_case,
            limit=args.limit,
        )
        specs = shard_specs(
            specs, shard_index=args.shard_index, num_shards=args.num_shards
        )
    except SifCacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.manifest is not None:
        write_manifest(specs, args.manifest)
        print(f"Wrote {len(specs)} entries to {args.manifest}", file=sys.stderr)
    else:
        print(f"Prepared {len(specs)} SIF build specs", file=sys.stderr)

    if not args.build:
        return 0

    built, skipped, failed = build_specs(
        specs,
        apptainer=args.apptainer,
        force=args.force,
        max_workers=args.max_workers,
    )
    print(
        f"SIF build complete: built={built} skipped={skipped} failed={failed}",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
