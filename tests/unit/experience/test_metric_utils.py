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

import torch

from nemo_rl.experience.metric_utils import (
    collect_nemo_gym_metric_samples,
    extract_nemo_gym_metric_values,
    summarize_nemo_gym_metric_samples,
)


def test_extract_nemo_gym_health_metrics() -> None:
    metrics = extract_nemo_gym_metric_values(
        {
            "resolved": True,
            "instance_config": {"mask_sample": True},
            "response": {
                "output": [
                    {"generation_token_ids": [1, 2, 3]},
                    {"generation_str": "not trainable"},
                ]
            },
            "timing": {
                "ray_queue_time": 2.5,
                "agent_error_kind": "context_window",
            },
            "eval_timed_out": float("nan"),
        }
    )

    assert metrics["resolved"] == 1.0
    assert metrics["mask_sample"] == 1.0
    assert metrics["trainable"] == 1.0
    assert metrics["trainable_items"] == 1.0
    assert metrics["generated_tokens"] == 3.0
    assert metrics["output_items"] == 2.0
    assert metrics["ray_queue_time"] == 2.5
    assert "eval_timed_out" not in metrics
    assert metrics["agent_error_kind/context_window"] == 1.0


def test_collect_uses_postprocessed_message_logs() -> None:
    samples = collect_nemo_gym_metric_samples(
        [{"response": {"output": [{"generation_str": "answer"}]}}],
        message_logs=[
            [
                {"role": "user", "token_ids": torch.tensor([1, 2])},
                {"role": "assistant", "token_ids": torch.tensor([3, 4, 5])},
            ]
        ],
    )

    assert samples["trainable"] == [1.0]
    assert samples["trainable_items"] == [1.0]
    assert samples["generated_tokens"] == [3.0]


def test_summarize_nemo_gym_health_metrics() -> None:
    metrics = summarize_nemo_gym_metric_samples(
        {"resolved": [1.0, 0.0], "generated_tokens": [2.0, 6.0]},
        prefix="task/rollout_metrics/nemo_gym",
    )

    assert metrics["task/rollout_metrics/nemo_gym/resolved/mean"] == 0.5
    assert metrics["task/rollout_metrics/nemo_gym/generated_tokens/min"] == 2.0
    assert metrics["task/rollout_metrics/nemo_gym/generated_tokens/max"] == 6.0
    assert "task/rollout_metrics/nemo_gym/generated_tokens/histogram" in metrics
