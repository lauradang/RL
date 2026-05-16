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

import threading as _threading
import time
import traceback
from typing import Any, Literal, Optional

import ray
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to the replay buffer.

    Supports both forced- and unforced-lag modes via the ``lag_mode``
    constructor argument:

    - ``lag_mode="forced"`` (default): each prompt batch is reserved for a
      specific future training step at generation time, and the trainer
      consumes only trajectories whose ``target_weight_version`` matches the
      current step. The in-flight semaphore is sized at
      ``num_prompts_per_step * max_trajectory_age_steps``.
    - ``lag_mode="unforced"``: each rollout is stamped only with the
      ``generation_weight_version`` snapshotted when its worker thread starts;
      no per-batch target reservation is made. The in-flight semaphore is
      sized at ``num_parallel_generations``. Mirrors Megatron-LM's
      ``enforce_order=False`` mode.
    """

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        lag_mode: Literal["forced", "unforced"] = "forced",
        num_parallel_generations: Optional[int] = None,
        start_step: int = 0,
    ):
        if lag_mode not in ("forced", "unforced"):
            raise ValueError(
                f"lag_mode must be 'forced' or 'unforced', got {lag_mode!r}"
            )

        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.running = False

        self.lag_mode: Literal["forced", "unforced"] = lag_mode
        self._log_prefix = "" if lag_mode == "forced" else "[unforced] "

        self._pg_lock: _threading.Lock = _threading.Lock()

        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step

        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        if self.lag_mode == "forced":
            # Track when generation limits cause collection to pause
            self._last_limit_warning_version: Optional[int] = None
            # Event to signal when generation limits are cleared (more efficient than polling)
            self._generation_limit_cleared = _threading.Event()
            self._generation_limit_cleared.set()
            # Lock to prevent race conditions when checking/spawning workers
            self._generation_check_lock: _threading.Lock = _threading.Lock()
            # Track which target weights are currently being generated (globally)
            self._generating_targets: set[int] = set()

            # Limit in-flight generator requests to num_prompts_per_step * max_trajectory_age_steps
            max_inflight = (
                int(self.master_config.grpo["num_prompts_per_step"])
                * int(self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"])
            ) or 1
            self._inflight_sema = _threading.Semaphore(max_inflight)
        else:
            if num_parallel_generations is None or num_parallel_generations <= 0:
                raise ValueError(
                    "num_parallel_generations must be a positive int for "
                    f"lag_mode='unforced', got {num_parallel_generations!r}"
                )
            self.num_parallel_generations: int = num_parallel_generations
            self._inflight_sema = _threading.Semaphore(num_parallel_generations)

    # ------------------------------------------------------------------ #
    # Forced-lag-only helpers                                            #
    # ------------------------------------------------------------------ #

    def _calculate_target_weights(self, generation_weight_version: int) -> list[int]:
        """Calculate target weight versions for given generation weight version.

        The list of versions returned enumerate the possible version a generation
        server can target. These versions are looped over to see what training
        step they can target. If all target versions are exhausted, this generation
        server will remain idle until the next weight update.

        Example:
        generation_weight_version = 10
        max_trajectory_age_steps = 4

        Returns:
            [11, 12, 13, 14]  # Meaning this generation server can create trajectories for training step 11, 12, 13, 14
        """
        async_cfg = self.master_config.grpo.get("async_grpo", {})
        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
        if generation_weight_version == self.initial_weight_version:
            return [
                i
                for i in range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_trajectory_age + 1,
                )
            ]

        return [generation_weight_version + i for i in range(1, max_trajectory_age + 1)]

    def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        last_target_weight_already_generated = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in target_weights:
                if (
                    target_weight > last_target_weight_already_generated
                    and target_weight not in self._generating_targets
                ):
                    self._generating_targets.add(target_weight)
                    print(f"🎯 Reserved target weight {target_weight} for generation")
                    return target_weight

        return None

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            last_target_weight_already_generated = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            with self._generation_check_lock:
                for target_weight in target_weights:
                    if (
                        target_weight > last_target_weight_already_generated
                        and target_weight not in self._generating_targets
                    ):
                        return False

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Public lifecycle / control API                                     #
    # ------------------------------------------------------------------ #

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        if self.lag_mode == "forced":
            was_paused = not self._generation_limit_cleared.is_set()
            if was_paused:
                self._generation_limit_cleared.set()
                print(f"🔄 Updated weight version to {version}, resuming collection")
            else:
                print(f"🔄 Updated weight version to {version}")
        else:
            print(f"🔄 {self._log_prefix}Updated weight version to {version}")

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()
        print(f"{self._log_prefix}Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()
        print(f"{self._log_prefix}Trajectory collection resumed")

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        if self.lag_mode == "unforced":
            print(
                f"{self._log_prefix}Started continuous trajectory collection "
                f"(max in-flight rollouts={self.num_parallel_generations})"
            )
        else:
            print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print(f"{self._log_prefix}Collection thread started, start_collection returning")

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}

    # ------------------------------------------------------------------ #
    # Refit coordination                                                 #
    # ------------------------------------------------------------------ #

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For vLLM V1 async engine, leverages in-flight weight updates via collective_rpc,
        allowing ongoing generations to continue with their current KV caches while
        weights are updated. This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print(f"🔄 {self._log_prefix}Preparing for refit: pausing new generations...")

        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        vllm_cfg = self.master_config.policy.get("generation", {}).get("vllm_cfg", {})
        is_async_engine = vllm_cfg.get("async_engine", False)
        in_flight_weight_updates = self.master_config.grpo.get("async_grpo", {}).get(
            "in_flight_weight_updates", False
        )

        if is_async_engine and in_flight_weight_updates:
            # vLLM V1 async engine supports in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                "🚀 Using vLLM V1 in-flight weight update - skipping wait for pending generations"
            )
            print(
                f"   {len(self._inflight_threads)} ongoing generations will complete with current weights"
            )
        else:
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print(f"🔄 {self._log_prefix}Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style).
        # Otherwise, keep using the stale KV caches (Magistral-style).
        async_cfg = self.master_config.grpo.get("async_grpo", {})
        if async_cfg.get("in_flight_weight_updates", False) and async_cfg.get(
            "recompute_kv_cache_after_weight_updates", False
        ):
            try:
                print("🔄 Invalidating vLLM prefix/KV caches after weight update")
                invalidated = self.policy_generation.invalidate_kv_cache()
                if invalidated:
                    print("✅ Invalidated vLLM prefix/KV caches after weight update")
                else:
                    print(
                        "⚠️ vLLM cache invalidation reported partial/unsuccessful on some workers"
                    )
            except Exception as e:
                print(f"⚠️ Failed to invalidate vLLM caches: {e}")

        self._refit_pause_cleared.set()

    def wait_for_pending_generations(self) -> None:
        """Wait for all in-flight generation threads to complete."""
        start_time = time.time()

        while True:
            with self._threads_lock:
                finished = {t for t in self._inflight_threads if not t.is_alive()}
                for t in finished:
                    self._inflight_threads.remove(t)

                pending_count = len(self._inflight_threads)

            if pending_count == 0:
                print("✅ All generation threads completed")
                break

            elapsed = time.time() - start_time
            print(
                f"⏳ Waiting for {pending_count} pending generation threads... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Collection loop                                                    #
    # ------------------------------------------------------------------ #

    def _collection_loop(self) -> None:
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                if not self._refit_pause_cleared.is_set() and self.running:
                    print(f"⏸️ {self._log_prefix}Pausing collection for refit...")
                    self._refit_pause_cleared.wait()
                    print(f"▶️ {self._log_prefix}Refit completed, resuming collection")

                # Forced lag has an extra "all targets exhausted" pause path; unforced
                # does not (its in-flight cap + buffer-full backpressure handle it).
                if (
                    self.lag_mode == "forced"
                    and self._should_pause_for_generation_limits()
                    and self.running
                ):
                    if self._last_limit_warning_version != self.current_weight_version:
                        async_cfg = self.master_config.grpo.get("async_grpo", {})
                        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
                        target_weights = [
                            self.current_weight_version + i
                            for i in range(max_trajectory_age)
                        ]

                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            f"already exist in buffer. Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                        self._generation_limit_cleared.clear()

                    self._generation_limit_cleared.wait()

                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ {self._log_prefix}Error in trajectory collection: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            print(f"🛑 {self._log_prefix}Trajectory collection stopped")

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        if self.lag_mode == "forced":
            self._process_batch_forced(batch)
        else:
            self._process_batch_unforced(batch)

    def _process_batch_forced(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Forced lag: reserve a target weight for this batch, then spawn workers."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts = batch.size

            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            print(
                f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            for prompt_idx in range(num_prompts):
                if not self._refit_pause_cleared.is_set() and self.running:
                    with self._threads_lock:
                        active_threads = len(self._inflight_threads)
                    print(
                        f"⏸️ Waiting for refit to complete before starting new generation ({active_threads} threads still active)"
                    )
                    print(
                        "   Note: With vLLM V1 async engine, active threads can complete during weight update"
                    )
                    self._refit_pause_cleared.wait()

                    # After refit finishes if weight version has updated, reflect that in the new trajectories
                    generation_weight_version = self.current_weight_version

                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)

                self._inflight_sema.acquire()
                worker = _threading.Thread(
                    target=self._run_prompt_group_worker_forced,
                    args=(
                        repeated_batch,
                        generation_weight_version,
                        target_weight,
                        prompt_idx,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            traceback.print_exc()

    def _process_batch_unforced(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Unforced lag: spawn one worker per prompt; each stamps its own weight."""
        try:
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts = batch.size

            for prompt_idx in range(num_prompts):
                if not self._refit_pause_cleared.is_set() and self.running:
                    with self._threads_lock:
                        active_threads = len(self._inflight_threads)
                    print(
                        f"⏸️ {self._log_prefix}Waiting for refit before starting new generation "
                        f"({active_threads} threads still active)"
                    )
                    self._refit_pause_cleared.wait()

                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)

                self._inflight_sema.acquire()

                if not self.running:
                    self._inflight_sema.release()
                    return

                generation_weight_version = self.current_weight_version

                worker = _threading.Thread(
                    target=self._run_prompt_group_worker_unforced,
                    args=(repeated_batch, generation_weight_version, prompt_idx),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ {self._log_prefix}Error processing batch: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Per-prompt rollout workers                                         #
    # ------------------------------------------------------------------ #

    def _run_rollout(
        self, repeated_batch: BatchedDataDict[DatumSpec]
    ) -> tuple[Any, Any]:
        """Run a single rollout for a per-prompt repeated batch.

        Returns ``(final_batch, rollout_metrics)``.
        """
        # Imported lazily to avoid circular dependency with grpo.py
        from nemo_rl.algorithms.grpo import _should_use_nemo_gym
        from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

        if _should_use_nemo_gym(self.master_config):
            generation_config = self.master_config.policy["generation"]
            nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                policy_generation=self.policy_generation,
                input_batch=repeated_batch,
                tokenizer=self.tokenizer,
                task_to_env=self.task_to_env,
                max_seq_len=self.master_config.policy["max_total_sequence_length"],
                generation_config=generation_config,
                max_rollout_turns=None,
                greedy=False,
            )
            return (
                nemo_gym_rollout_result.final_batch,
                nemo_gym_rollout_result.rollout_metrics,
            )

        return run_async_multi_turn_rollout(
            policy_generation=self.policy_generation,
            input_batch=repeated_batch,
            tokenizer=self.tokenizer,
            task_to_env=self.task_to_env,
            max_seq_len=self.master_config.policy["max_total_sequence_length"],
            max_rollout_turns=self.master_config.grpo["max_rollout_turns"],
            greedy=False,
        )

    def _run_prompt_group_worker_forced(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
    ) -> None:
        try:
            final_batch, rollout_metrics = self._run_rollout(repeated_batch)

            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.add.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
                        )

                        # Release reservation when FIRST prompt group for this target is successfully buffered
                        if prompt_idx == 0:
                            with self._generation_check_lock:
                                if target_weight_version in self._generating_targets:
                                    self._generating_targets.discard(
                                        target_weight_version
                                    )
                                    print(
                                        f"🧹 Released reservation for target weight {target_weight_version} (first prompt buffered)"
                                    )
                        break
                    elif status == "full":
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            traceback.print_exc()
        finally:
            # Clean up reservation in case of error (if not already cleaned up)
            with self._generation_check_lock:
                if target_weight_version in self._generating_targets:
                    self._generating_targets.discard(target_weight_version)
                    print(
                        f"🧹 Emergency cleanup: Released reservation for target weight {target_weight_version}"
                    )

            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                traceback.print_exc()

    def _run_prompt_group_worker_unforced(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        prompt_idx: int,
    ) -> None:
        try:
            final_batch, rollout_metrics = self._run_rollout(repeated_batch)

            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.add.remote(
                            trajectory_group,
                            generation_weight_version,
                        )
                    )
                    if status == "success":
                        print(
                            f"📦 {self._log_prefix}Buffered prompt group "
                            f"(prompt_idx {prompt_idx}, weight_version {generation_weight_version})"
                        )
                        break
                    elif status == "full":
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ {self._log_prefix}Failed to enqueue prompt group: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"❌ {self._log_prefix}Error in prompt group worker: {e}")
            traceback.print_exc()
        finally:
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                traceback.print_exc()
