# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest
import ray
import torch
from yaml import safe_load

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    _PriorityRolloutLimiter,
    setup_nemo_gym_config,
)
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)
from tests.unit.models.generation.test_vllm_generation import (
    tokenizer as nemo_gym_tokenizer,  # noqa: F401
)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


def test_priority_rollout_limiter_enforces_max_concurrency():
    async def scenario():
        limiter = _PriorityRolloutLimiter(max_concurrent=1)
        second_acquired = asyncio.Event()

        async def acquire_second():
            async with limiter.context("train"):
                second_acquired.set()

        async with limiter.context("train"):
            second_task = asyncio.create_task(acquire_second())
            await _wait_until(
                lambda: limiter.waiting_by_kind["train"] == 1
                and limiter.active_by_kind["train"] == 1
            )
            assert not second_acquired.is_set()

        await asyncio.wait_for(second_task, timeout=1)
        assert second_acquired.is_set()
        stats = await limiter.stats()
        assert stats["active_total_requests"] == 0
        assert stats["max_concurrent_requests"] == 1

    asyncio.run(scenario())


def test_priority_rollout_limiter_prioritizes_validation_waiters():
    async def scenario():
        limiter = _PriorityRolloutLimiter(max_concurrent=1)
        release_validation = asyncio.Event()
        release_train = asyncio.Event()
        train_acquired = asyncio.Event()
        validation_acquired = asyncio.Event()

        async def acquire_training():
            async with limiter.context("train"):
                train_acquired.set()
                await release_train.wait()

        async def acquire_validation():
            async with limiter.context("validation"):
                validation_acquired.set()
                await release_validation.wait()

        async with limiter.context("train"):
            train_task = asyncio.create_task(acquire_training())
            await _wait_until(lambda: limiter.waiting_by_kind["train"] == 1)
            validation_task = asyncio.create_task(acquire_validation())
            await _wait_until(lambda: limiter.waiting_by_kind["validation"] == 1)

        await asyncio.wait_for(validation_acquired.wait(), timeout=1)
        assert not train_acquired.is_set()
        stats = await limiter.stats()
        assert stats["active_validation_requests"] == 1
        assert stats["waiting_train_requests"] == 1

        release_validation.set()
        await asyncio.wait_for(train_acquired.wait(), timeout=1)
        release_train.set()
        await asyncio.wait_for(validation_task, timeout=1)
        await asyncio.wait_for(train_task, timeout=1)

    asyncio.run(scenario())


def test_priority_rollout_limiter_training_resumes_after_validation_mode():
    async def scenario():
        limiter = _PriorityRolloutLimiter(max_concurrent=1)
        train_acquired = asyncio.Event()

        async def acquire_training():
            async with limiter.context("train"):
                train_acquired.set()

        await limiter.enter_validation_mode()
        train_task = asyncio.create_task(acquire_training())
        await _wait_until(lambda: limiter.waiting_by_kind["train"] == 1)
        assert not train_acquired.is_set()

        await limiter.exit_validation_mode()
        await asyncio.wait_for(train_task, timeout=1)
        assert train_acquired.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_cap", [0, -1, True, 1.5, "8"])
def test_priority_rollout_limiter_rejects_invalid_cap(invalid_cap):
    with pytest.raises(ValueError, match="positive int"):
        _PriorityRolloutLimiter(invalid_cap)


def test_nemo_gym_run_rollouts_omits_semaphore_without_limiter():
    async def scenario():
        class FakeRunCollectionHelper:
            def __init__(self):
                self.kwargs = None

            def run_examples(self, **kwargs):
                self.kwargs = kwargs

                async def done():
                    return {"_rowidx": 0}, {"message_log": []}

                return [done()]

        env = NemoGym.__new__(NemoGym)
        env._rollout_limiter = None
        env.rollout_max_attempts_to_avoid_lp_nan = 1
        env.head_server_config = object()
        env.rch = FakeRunCollectionHelper()
        env._postprocess_nemo_gym_to_nemo_rl_result = lambda result, tokenizer: result

        results, _ = await NemoGym.run_rollouts(env, [{}], None, "timing/test")

        assert results == [{"message_log": []}]
        assert "semaphore" not in env.rch.kwargs

    asyncio.run(scenario())


def test_nemo_gym_run_rollouts_passes_semaphore_with_limiter():
    async def scenario():
        class FakeRunCollectionHelper:
            def __init__(self):
                self.kwargs = None

            def run_examples(self, **kwargs):
                self.kwargs = kwargs

                async def done():
                    return {"_rowidx": 0}, {"message_log": []}

                return [done()]

        env = NemoGym.__new__(NemoGym)
        env._rollout_limiter = _PriorityRolloutLimiter(max_concurrent=1)
        env.rollout_max_attempts_to_avoid_lp_nan = 1
        env.head_server_config = object()
        env.rch = FakeRunCollectionHelper()
        env._postprocess_nemo_gym_to_nemo_rl_result = lambda result, tokenizer: result

        await NemoGym.run_rollouts(
            env, [{}], None, "timing/test", rollout_kind="validation"
        )

        assert "semaphore" in env.rch.kwargs

    asyncio.run(scenario())


def test_nemo_gym_limiter_actor_methods_report_stats():
    async def scenario():
        env = NemoGym.__new__(NemoGym)
        env._rollout_limiter = None

        assert await NemoGym.has_rollout_limiter(env) is False
        stats = await NemoGym.get_rollout_limiter_stats(env)
        assert stats["max_concurrent_requests"] == 0

        env._rollout_limiter = _PriorityRolloutLimiter(max_concurrent=1)
        assert await NemoGym.has_rollout_limiter(env) is True
        release_validation = asyncio.Event()

        async def acquire_validation():
            async with env._rollout_limiter.context("validation"):
                await release_validation.wait()

        async with env._rollout_limiter.context("train"):
            validation_task = asyncio.create_task(acquire_validation())
            await _wait_until(
                lambda: env._rollout_limiter.waiting_by_kind["validation"] == 1
            )
            stats = await NemoGym.get_rollout_limiter_stats(env)
            assert stats["active_train_requests"] == 1
            assert stats["waiting_validation_requests"] == 1
            assert stats["active_total_requests"] == 1

        release_validation.set()
        await asyncio.wait_for(validation_task, timeout=1)

    asyncio.run(scenario())


@pytest.mark.nemo_gym
def test_nemo_gym_stub_module():
    from nemo_gym import config_types

    print(
        f"NeMo-Gym test successfully run! NeMo-Gym config_types module: {config_types}"
    )


@pytest.fixture(scope="function")
def nemo_gym_vllm_generation(cluster, nemo_gym_tokenizer):  # noqa: F811
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={"generation": generation_config}
    )
    setup_nemo_gym_config(master_config, nemo_gym_tokenizer)

    generation_config["vllm_cfg"]["max_model_len"] = 16_384
    # This is the tool parser for Qwen/Qwen3-0.6B. This needs to be changed for other models.
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    vllm_generation = VllmGeneration(cluster, generation_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="function")
def nemo_gym(nemo_gym_vllm_generation):
    """Create a NeMo-Gym actor for testing."""

    yaml_str = r"""example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
      domain: instruction_following
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
openai_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true
      uses_reasoning_parser: true
rollout_max_attempts_to_avoid_lp_nan: 1
"""

    config = NemoGymConfig(
        model_name=nemo_gym_vllm_generation.cfg["model_name"],
        base_urls=nemo_gym_vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.nemo_gym.NemoGym"
            ),
        }
    ).remote(config)

    # Blocking wait for NeMo-Gym to spin up
    ray.get(env.health_check.remote())

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture(scope="function")
def nemo_gym_sanity_test_data():
    fpath = Path(__file__).parent / "nemo_gym_test_data/test_nemo_gym_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def _write_actual_test_data(original_input: list, actual_result: list):
    """Write actual rollout results to actual_test_nemo_gym_sanity.json.

    This makes it easy to update the expected output after a Gym commit bump:
        cp nemo_gym_test_data/actual_test_nemo_gym_sanity.json nemo_gym_test_data/test_nemo_gym_sanity.json
    """

    def _convert(obj):
        """Recursively convert torch tensors to Python lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    cleaned = deepcopy(actual_result)
    for r in cleaned:
        r.pop("full_result", None)
        for msg in r.get("message_log", [])[1:]:
            if "token_ids" in msg:
                msg["token_ids"] = []
            if "generation_logprobs" in msg:
                msg["generation_logprobs"] = []

    output_path = (
        Path(__file__).parent / "nemo_gym_test_data/actual_test_nemo_gym_sanity.json"
    )
    data = _convert({"input": original_input, "expected_output": cleaned})
    with open(output_path, "w") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"Wrote updated test data to {output_path}")


@pytest.mark.nemo_gym
def test_nemo_gym_sanity(
    nemo_gym,
    nemo_gym_sanity_test_data,
    nemo_gym_vllm_generation,
    nemo_gym_tokenizer,  # noqa: F811
):
    """Test basic functionality of MathEnvironment step with simple messages."""

    # Save original input before mutation for writing the actual test data file
    original_input = deepcopy(nemo_gym_sanity_test_data["input"])

    # We need to match NeMo RL generation config params before sending to NeMo-Gym
    generation_config = nemo_gym_vllm_generation.cfg
    examples = nemo_gym_sanity_test_data["input"]
    for idx, example in enumerate(examples):
        example["responses_create_params"]["temperature"] = generation_config[
            "temperature"
        ]
        example["responses_create_params"]["top_p"] = generation_config["top_p"]
        example["_rowidx"] = idx

    actual_result, _ = ray.get(
        nemo_gym.run_rollouts.remote(
            nemo_gym_sanity_test_data["input"], nemo_gym_tokenizer, ""
        )
    )
    expected_result = nemo_gym_sanity_test_data["expected_output"]

    # These are tensors originally and we swap them back to a list for comparison below
    for d in actual_result:
        for message in d["input_message_log"]:
            message["token_ids"] = message["token_ids"].tolist()
        # Right now, we don't need to swap the token ids in the message log since they pointto the same underlying dictionary as above.
        # for message in d["message_log"][:1]:
        #     message["token_ids"] = message["token_ids"].tolist()

    # Write the actual result to a file so it can be used to update the expected output.
    # To update: cp actual_test_nemo_gym_sanity.json test_nemo_gym_sanity.json
    _write_actual_test_data(original_input, actual_result)

    def _standardize_single_result(d: dict):
        d = deepcopy(d)
        d.pop("full_result", None)

        # We remove these fields and message from comparison since we cannot guarantee exact generation reproducibility
        d["message_log"] = d["message_log"][:2]
        for message in d["message_log"][1:]:
            if "token_ids" in message:
                message["token_ids"] = []
            if "generation_logprobs" in message:
                message["generation_logprobs"] = []
            if "prompt_str" in message:
                message["prompt_str"] = "dummy prompt_str"
            if "generation_str" in message:
                message["generation_str"] = "dummy generation_str"

        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)
