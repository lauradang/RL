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

"""Normalize SWE-agent NeMo-Gym registry JSONL for the NeMo-RL rollout path."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AGENT_REF_TYPE = "responses_api_agents"
INSTANCE_DICT_REQUIRED_KEYS = (
    "instance_id",
    "dataset_name",
    "split",
    "patch",
    "test_patch",
)
INSTANCE_DICT_CANDIDATE_KEYS = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "base_commit",
    "created_at",
    "dataset_name",
    "difficulty",
    "environment_setup_commit",
    "golden_patch",
    "hints_text",
    "instance_id",
    "patch",
    "problem_statement",
    "repo",
    "split",
    "test_patch",
    "version",
)


class NormalizeError(ValueError):
    """Raised when a JSONL row cannot be normalized."""


def _loads_instance_dict(value: Any, line_number: int) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise NormalizeError(
                f"line {line_number}: metadata.instance_dict is not valid JSON"
            ) from exc
        if not isinstance(loaded, dict):
            raise NormalizeError(
                f"line {line_number}: metadata.instance_dict must decode to an object"
            )
        return loaded
    raise NormalizeError(
        f"line {line_number}: metadata.instance_dict must be a JSON string or object"
    )


def _get_from_row_or_metadata(
    row: dict[str, Any], metadata: dict[str, Any], key: str
) -> Any:
    if key in row:
        return row[key]
    if key in metadata:
        return metadata[key]
    return None


def _build_instance_dict(
    row: dict[str, Any], metadata: dict[str, Any], line_number: int
) -> dict[str, Any]:
    if "instance_dict" in metadata:
        instance_dict = _loads_instance_dict(metadata["instance_dict"], line_number)
    else:
        instance_dict = {}
        for key in INSTANCE_DICT_CANDIDATE_KEYS:
            value = _get_from_row_or_metadata(row, metadata, key)
            if value is not None:
                instance_dict[key] = value

        if "patch" not in instance_dict and "golden_patch" in instance_dict:
            instance_dict["patch"] = instance_dict["golden_patch"]
        if "golden_patch" not in instance_dict and "patch" in instance_dict:
            instance_dict["golden_patch"] = instance_dict["patch"]

    # Current SWE-agent Gym reads these keys from instance_dict for setup and
    # validation. Keep missing values explicit so preflight checks can flag them.
    for key in INSTANCE_DICT_REQUIRED_KEYS:
        instance_dict.setdefault(key, "")

    return instance_dict


def normalize_row(
    row: dict[str, Any],
    *,
    agent_ref_name: str,
    model_name: str,
    line_number: int,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(row, dict):
        raise NormalizeError(f"line {line_number}: row must be a JSON object")

    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise NormalizeError(
            f"line {line_number}: row.responses_create_params must be an object"
        )

    metadata = responses_create_params.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise NormalizeError(
            f"line {line_number}: responses_create_params.metadata must be an object"
        )

    normalized = dict(row)
    normalized_params = dict(responses_create_params)
    normalized_metadata = dict(metadata)

    normalized_params["model"] = model_name
    instance_dict = _build_instance_dict(row, normalized_metadata, line_number)
    normalized_metadata["instance_dict"] = json.dumps(instance_dict, sort_keys=True)
    normalized_params["metadata"] = normalized_metadata

    normalized["responses_create_params"] = normalized_params
    normalized["agent_ref"] = {
        "type": AGENT_REF_TYPE,
        "name": agent_ref_name,
    }

    missing = [
        key
        for key in INSTANCE_DICT_REQUIRED_KEYS
        if not str(instance_dict.get(key, "")).strip()
    ]
    warnings = []
    if missing:
        warnings.append(
            f"line {line_number}: instance_dict has empty required SWE fields: "
            f"{', '.join(missing)}"
        )

    return normalized, warnings


def normalize_file(
    input_path: Path,
    output_path: Path,
    *,
    agent_ref_name: str,
    model_name: str,
    strict_swe_fields: bool = False,
    limit: int | None = None,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    row_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as src, output_path.open("w") as dst:
        for line_number, line in enumerate(src, start=1):
            if limit is not None and row_count >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise NormalizeError(f"line {line_number}: invalid JSON") from exc

            normalized, row_warnings = normalize_row(
                row,
                agent_ref_name=agent_ref_name,
                model_name=model_name,
                line_number=line_number,
            )
            warnings.extend(row_warnings)
            if strict_swe_fields and row_warnings:
                raise NormalizeError(row_warnings[0])

            dst.write(json.dumps(normalized, sort_keys=True) + "\n")
            row_count += 1

    return row_count, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize SWE-agent NeMo-Gym registry JSONL for NeMo-RL. "
            "The output keeps responses_create_params, pins its model name, "
            "adds top-level agent_ref, and ensures metadata.instance_dict exists."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-ref-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--strict-swe-fields",
        action="store_true",
        help="Fail if normalized instance_dict has empty required SWE fields.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally write only the first N non-empty rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows, warnings = normalize_file(
            args.input,
            args.output,
            agent_ref_name=args.agent_ref_name,
            model_name=args.model_name,
            strict_swe_fields=args.strict_swe_fields,
            limit=args.limit,
        )
    except NormalizeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {rows} rows to {args.output}", file=sys.stderr)
    if warnings:
        print(
            f"Warnings: {len(warnings)} rows/fields need review. "
            f"First warning: {warnings[0]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
