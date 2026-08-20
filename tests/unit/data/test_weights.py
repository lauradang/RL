# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

from nemo_rl.data.weights import (
    TaskWeightSpec,
    compute_quota,
    distribute_counts,
    normalize_weights,
)


def _spec(name, weight=None, evaluation_only=False):
    return TaskWeightSpec(name, weight, evaluation_only)


def test_normalize_weights_and_exclude_evaluation_only():
    assert normalize_weights(
        [_spec("a", 3), _spec("b", 1), _spec("eval", None, True)]
    ) == {"a": 0.75, "b": 0.25, "eval": 0.0}


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([_spec("a", 1), _spec("b")], "all-or-nothing"),
        ([_spec("a", -1), _spec("b", 1)], "non-negative"),
        ([_spec("a", 0), _spec("b", 0)], "must be positive"),
        ([_spec("a", 1), _spec("a", 1)], "must be unique"),
    ],
)
def test_invalid_weight_declarations_raise(specs, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(specs)


def test_largest_remainder_is_exact_and_deterministic():
    assert distribute_counts(10, [1 / 3, 1 / 3, 1 / 3]) == [4, 3, 3]
    assert distribute_counts(10, [1 / 3, 1 / 3, 1 / 3], distribute_remainder=False) == [
        3,
        3,
        3,
    ]


def test_compute_quota_omits_zero_weight_tasks():
    weights = normalize_weights(
        [_spec("a", 3), _spec("b", 1), _spec("eval", None, True)]
    )
    assert compute_quota(8, weights) == {"a": 6, "b": 2}
