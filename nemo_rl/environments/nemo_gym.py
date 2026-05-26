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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


RolloutKind = Literal["train", "validation"]
G_ROLLOUT_KINDS: tuple[RolloutKind, RolloutKind] = ("train", "validation")


class _RolloutLimiterContext:
    def __init__(self, limiter: "_PriorityRolloutLimiter", rollout_kind: RolloutKind):
        self._limiter = limiter
        self._rollout_kind = rollout_kind

    async def __aenter__(self) -> "_RolloutLimiterContext":
        await self._limiter.acquire(self._rollout_kind)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._limiter.release(self._rollout_kind)


class _PriorityRolloutLimiter:
    """Priority-aware async limiter shared by training and validation rollouts."""

    def __init__(self, max_concurrent: int):
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent <= 0
        ):
            raise ValueError(
                f"max_concurrent_rollout_requests must be a positive int, got {max_concurrent!r}"
            )

        self.max_concurrent = max_concurrent
        self.active_by_kind: dict[RolloutKind, int] = {
            "train": 0,
            "validation": 0,
        }
        self.waiting_by_kind: dict[RolloutKind, int] = {
            "train": 0,
            "validation": 0,
        }
        self.peak_active_by_kind: dict[RolloutKind, int] = {
            "train": 0,
            "validation": 0,
        }
        self.peak_active_total = 0
        self.validation_mode_depth = 0
        self._condition = asyncio.Condition()

    def context(self, rollout_kind: RolloutKind) -> _RolloutLimiterContext:
        self._validate_rollout_kind(rollout_kind)
        return _RolloutLimiterContext(self, rollout_kind)

    async def acquire(self, rollout_kind: RolloutKind) -> None:
        self._validate_rollout_kind(rollout_kind)
        async with self._condition:
            self.waiting_by_kind[rollout_kind] += 1
            try:
                await self._condition.wait_for(
                    lambda: self._can_acquire(rollout_kind)
                )
                self.active_by_kind[rollout_kind] += 1
                self._update_peaks()
            finally:
                self.waiting_by_kind[rollout_kind] -= 1

    async def release(self, rollout_kind: RolloutKind) -> None:
        self._validate_rollout_kind(rollout_kind)
        async with self._condition:
            if self.active_by_kind[rollout_kind] <= 0:
                raise RuntimeError(
                    f"Cannot release {rollout_kind!r} rollout permit; none are active."
                )
            self.active_by_kind[rollout_kind] -= 1
            self._condition.notify_all()

    async def enter_validation_mode(self) -> None:
        async with self._condition:
            self.validation_mode_depth += 1
            if self.validation_mode_depth == 1:
                self._reset_peaks()
            self._condition.notify_all()

    async def exit_validation_mode(self) -> None:
        async with self._condition:
            if self.validation_mode_depth <= 0:
                raise RuntimeError("Cannot exit validation mode; it is not active.")
            self.validation_mode_depth -= 1
            self._condition.notify_all()

    async def stats(self) -> dict[str, int]:
        async with self._condition:
            active_train = self.active_by_kind["train"]
            active_validation = self.active_by_kind["validation"]
            waiting_train = self.waiting_by_kind["train"]
            waiting_validation = self.waiting_by_kind["validation"]
            return {
                "active_train_requests": active_train,
                "active_validation_requests": active_validation,
                "active_total_requests": active_train + active_validation,
                "waiting_train_requests": waiting_train,
                "waiting_validation_requests": waiting_validation,
                "waiting_total_requests": waiting_train + waiting_validation,
                "max_concurrent_requests": self.max_concurrent,
                "validation_mode_depth": self.validation_mode_depth,
                "peak_active_train_requests": self.peak_active_by_kind["train"],
                "peak_active_validation_requests": self.peak_active_by_kind[
                    "validation"
                ],
                "peak_active_total_requests": self.peak_active_total,
            }

    def _can_acquire(self, rollout_kind: RolloutKind) -> bool:
        active_total = sum(self.active_by_kind.values())
        if active_total >= self.max_concurrent:
            return False
        if rollout_kind == "validation":
            return True
        return (
            self.validation_mode_depth == 0
            and self.waiting_by_kind["validation"] == 0
        )

    def _validate_rollout_kind(self, rollout_kind: RolloutKind) -> None:
        if rollout_kind not in G_ROLLOUT_KINDS:
            raise ValueError(
                f"rollout_kind must be one of {G_ROLLOUT_KINDS}, got {rollout_kind!r}"
            )

    def _reset_peaks(self) -> None:
        self.peak_active_by_kind = {
            "train": self.active_by_kind["train"],
            "validation": self.active_by_kind["validation"],
        }
        self.peak_active_total = sum(self.active_by_kind.values())

    def _update_peaks(self) -> None:
        for rollout_kind in G_ROLLOUT_KINDS:
            self.peak_active_by_kind[rollout_kind] = max(
                self.peak_active_by_kind[rollout_kind],
                self.active_by_kind[rollout_kind],
            )
        self.peak_active_total = max(
            self.peak_active_total,
            sum(self.active_by_kind.values()),
        )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

        self.node_ip = _get_node_ip_local()
        self.head_server_port = _get_free_port_local()

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = dict(
            self.cfg.get("initial_global_config_dict") or dict()
        )
        max_concurrent_rollout_requests = initial_global_config_dict.pop(
            "max_concurrent_rollout_requests", None
        )
        self._rollout_limiter: Optional[_PriorityRolloutLimiter] = None
        if max_concurrent_rollout_requests is not None:
            self._rollout_limiter = _PriorityRolloutLimiter(
                max_concurrent_rollout_requests
            )
            print(
                "Configured NeMo-Gym rollout limiter with "
                f"max_concurrent_rollout_requests={max_concurrent_rollout_requests}."
            )

        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = (
            "dummy_key"  # No key necessary for training.
        )
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]
        # In multinode runs, Gym-managed service configs must advertise a real node IP
        # rather than falling back to localhost, or remote workers will connect to
        # their own loopback interface instead of the actor-hosted service.
        initial_global_config_dict.setdefault("default_host", self.node_ip)

        initial_global_config_dict.setdefault(
            "global_aiohttp_connector_limit_per_host", 16_384
        )
        initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)
        print(
            f"""Set global_aiohttp_connector_limit_per_host={initial_global_config_dict["global_aiohttp_connector_limit_per_host"]} and global_aiohttp_connector_limit={initial_global_config_dict["global_aiohttp_connector_limit"]}.
Depending on your data shape, you may want to change these values."""
        )

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
        print(f"Ray head node address: {ray_context.gcs_address}")

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rollout_max_attempts_to_avoid_lp_nan = initial_global_config_dict.pop(
            "rollout_max_attempts_to_avoid_lp_nan", 1
        )

        assert self.rollout_max_attempts_to_avoid_lp_nan >= 1, (
            "`rollout_max_attempts_to_avoid_lp_nan` must be at least 1"
        )

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            )
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    def health_check(self) -> bool:
        return True

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
        rollout_kind: RolloutKind = "train",
    ) -> tuple[list[dict], dict[str, Any]]:
        timer = Timer()
        if rollout_kind not in G_ROLLOUT_KINDS:
            raise ValueError(
                f"rollout_kind must be one of {G_ROLLOUT_KINDS}, got {rollout_kind!r}"
            )

        timer.start("_run_rollouts_total")
        max_attempts, trial = self.rollout_max_attempts_to_avoid_lp_nan, 0
        while trial < max_attempts:
            nemo_gym_num_rows = len(nemo_gym_examples)
            run_examples_kwargs = {
                "examples": nemo_gym_examples,
                "head_server_config": self.head_server_config,
            }
            if self._rollout_limiter is not None:
                run_examples_kwargs["semaphore"] = self._rollout_limiter.context(
                    rollout_kind
                )
            nemo_gym_result_iterator = self.rch.run_examples(**run_examples_kwargs)

            nemo_rl_rowidxs = []
            nemo_rl_results = []
            for task in nemo_gym_result_iterator:
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await task

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_result, tokenizer
                    )

                nemo_rl_rowidxs.append(nemo_gym_row["_rowidx"])
                nemo_rl_results.append(nemo_rl_result)

            # determine if generation_logprobs contain NaN; if not, break;
            logprob_contains_nan = False
            for nemo_rl_result in nemo_rl_results:
                for message in nemo_rl_result["message_log"]:
                    if (
                        "generation_logprobs" in message
                        and message["generation_logprobs"] is not None
                    ):
                        if torch.isnan(message["generation_logprobs"]).any():
                            logprob_contains_nan = True
                            break
            if logprob_contains_nan:
                trial += 1
                print(
                    f"Generation logprobs contain NaN; retrying... (trial {trial}/{max_attempts})"
                )
                continue
            else:
                break

        nemo_rl_sort_results = [None] * nemo_gym_num_rows
        for rowidx, result in zip(nemo_rl_rowidxs, nemo_rl_results):
            nemo_rl_sort_results[rowidx] = result
        nemo_rl_results = nemo_rl_sort_results

        timer.stop("_run_rollouts_total")
        timing_metrics = timer.get_timing_metrics("sum")
        total_time = timing_metrics.pop("_run_rollouts_total")
        timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
            100 * timing_metrics[f"{timer_prefix}/postprocess_results"] / total_time
        )

        return nemo_rl_results, timing_metrics

    async def enter_validation_mode(self) -> None:
        if self._rollout_limiter is not None:
            await self._rollout_limiter.enter_validation_mode()

    async def exit_validation_mode(self) -> None:
        if self._rollout_limiter is not None:
            await self._rollout_limiter.exit_validation_mode()

    async def has_rollout_limiter(self) -> bool:
        return self._rollout_limiter is not None

    async def get_rollout_limiter_stats(self) -> dict[str, int]:
        if self._rollout_limiter is None:
            return {
                "active_train_requests": 0,
                "active_validation_requests": 0,
                "active_total_requests": 0,
                "waiting_train_requests": 0,
                "waiting_validation_requests": 0,
                "waiting_total_requests": 0,
                "max_concurrent_requests": 0,
                "validation_mode_depth": 0,
                "peak_active_train_requests": 0,
                "peak_active_validation_requests": 0,
                "peak_active_total_requests": 0,
            }
        return await self._rollout_limiter.stats()

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            assert (
                seen_token_ids
                == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
            ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(
                        output_item_dict["prompt_token_ids"][len(seen_token_ids) :]
                    ),
                }
            )
            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(output_item_dict["generation_token_ids"]),
                    "generation_logprobs": torch.tensor(
                        output_item_dict["generation_log_probs"]
                    ),
                }
            )

            seen_token_ids.extend(nemo_rl_message_log[-2]["token_ids"])
            seen_token_ids.extend(nemo_rl_message_log[-1]["token_ids"])

            # We pop to remove larger tensors from logging.
            output_item_dict["prompt_str"] = tokenizer.decode(
                output_item_dict.pop("prompt_token_ids")
            )
            output_item_dict["generation_str"] = tokenizer.decode(
                output_item_dict.pop("generation_token_ids")
            )
            output_item_dict.pop("generation_log_probs")

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            prompt_token_ids = tokenizer.apply_chat_template(
                input_messages, tokenize=True
            )
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"This typically means the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated.\n"
                f"  Prompt length: {len(prompt_token_ids)} tokens.\n"
                f"  → Fix: increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"to a value larger than {len(prompt_token_ids)}."
            )

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


########################################
# Global config utils
########################################


def setup_nemo_gym_config(config, tokenizer) -> None:
    generation_config = config.policy["generation"]

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None
