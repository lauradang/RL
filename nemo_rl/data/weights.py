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

"""Deterministic per-task quotas for weighted multi-dataset training."""

from collections import Counter
from dataclasses import dataclass
from typing import Any, TypeAlias

TaskName: TypeAlias = str
TaskWeights: TypeAlias = dict[TaskName, float]
TaskQuota: TypeAlias = dict[TaskName, int]
TaskDeficits: TypeAlias = dict[TaskName, int]
TaskDataloaderState: TypeAlias = dict[TaskName, dict[str, Any]]
UNWEIGHTED_TASK_NAME: TaskName = "_unweighted"


@dataclass(frozen=True)
class TaskWeightSpec:
    """One task's user-declared weighting configuration."""

    task_name: TaskName
    weight: float | None
    evaluation_only: bool


def normalize_weights(specs: list[TaskWeightSpec]) -> TaskWeights:
    """Normalize weights across trainable tasks, validating the declaration."""
    task_names = [spec.task_name for spec in specs]
    duplicate_names = sorted(
        task_name for task_name, count in Counter(task_names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(
            f"Weighted dataset task names must be unique; duplicates: {duplicate_names}"
        )
    if all(spec.weight is None for spec in specs):
        return {}

    missing = [
        spec.task_name
        for spec in specs
        if not spec.evaluation_only and spec.weight is None
    ]
    if missing:
        raise ValueError(
            "Dataset weights are all-or-nothing; these non-evaluation datasets "
            f"do not declare `weight`: {missing}"
        )

    negative = [
        spec.task_name for spec in specs if spec.weight is not None and spec.weight < 0
    ]
    if negative:
        raise ValueError(
            f"Dataset weights must be non-negative; got negative weights for {negative}"
        )

    total = sum(
        spec.weight
        for spec in specs
        if not spec.evaluation_only and spec.weight is not None
    )
    if total <= 0:
        raise ValueError(
            "Total weight of non-evaluation-only training datasets must be positive"
        )

    return {
        spec.task_name: (
            0.0
            if spec.evaluation_only
            else float(spec.weight) / total  # missing train weights checked above
        )
        for spec in specs
    }


def distribute_counts(
    total_count: int,
    weights: list[float],
    distribute_remainder: bool = True,
) -> list[int]:
    """Split a count by largest-remainder apportionment with stable tie breaks."""
    if total_count < 0:
        raise ValueError(f"total_count must be non-negative, got {total_count}")
    exact_counts = [total_count * weight for weight in weights]
    counts = [int(count) for count in exact_counts]
    remainder = total_count - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    if distribute_remainder:
        for index in order[:remainder]:
            counts[index] += 1
    return counts


def compute_quota(total_count: int, weights: TaskWeights) -> TaskQuota:
    """Turn normalized task weights into exact prompt counts for one step."""
    task_names = [task_name for task_name, weight in weights.items() if weight > 0]
    counts = distribute_counts(total_count, [weights[name] for name in task_names])
    return dict(zip(task_names, counts, strict=True))
