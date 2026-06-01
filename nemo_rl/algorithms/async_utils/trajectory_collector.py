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
LagMode = Literal["forced", "unforced"]


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
        lag_mode: LagMode = "forced",
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
        self.lag_mode = lag_mode

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()  # Start in cleared state

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # Limit in-flight generator requests to num_prompts_per_step * max_trajectory_age_steps
        # This value limits the parallelism of the generation requests.
        max_inflight = (
            int(self.master_config.grpo["num_prompts_per_step"])
            * int(self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self.max_inflight_generations = max_inflight
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()

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
        # Read async config strictly from grpo.async_grpo
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

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        if self.lag_mode == "unforced":
            print(f"🔄 [unforced] Updated weight version to {version}")
            return

        # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        if self.lag_mode == "unforced":
            return False

        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            last_target_weight_already_generated = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    if (
                        target_weight > last_target_weight_already_generated
                        and target_weight not in self._generating_targets
                    ):
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
            )
            return True
        except Exception:
            return False

    def _log_prefix(self) -> str:
        return "[unforced] " if self.lag_mode == "unforced" else ""

    def _wait_for_manual_pause(self) -> bool:
        if not self._manual_pause_cleared.is_set() and self.running:
            self._manual_pause_cleared.wait()
        return self.running

    def _wait_for_refit_pause(self, *, before_prompt: bool) -> bool:
        if self._refit_pause_cleared.is_set() or not self.running:
            return self.running

        if before_prompt:
            with self._threads_lock:
                active_threads = len(self._inflight_threads)
            print(
                f"⏸️ {self._log_prefix()}Waiting for refit before starting new generation "
                f"({active_threads} threads still active)"
            )
            if self.lag_mode == "forced":
                print(
                    "   Note: With vLLM V1 async engine, active threads can complete during weight update"
                )
            self._refit_pause_cleared.wait()
        else:
            print(f"⏸️ {self._log_prefix()}Pausing collection for refit...")
            self._refit_pause_cleared.wait()
            print(f"▶️ {self._log_prefix()}Refit completed, resuming collection")

        return self.running

    def _wait_for_generation_limit_clear(self) -> bool:
        if not self._should_pause_for_generation_limits() or not self.running:
            return self.running

        if self._last_limit_warning_version != self.current_weight_version:
            async_cfg = self.master_config.grpo.get("async_grpo", {})
            max_trajectory_age = async_cfg["max_trajectory_age_steps"]
            target_weights = [
                self.current_weight_version + i for i in range(max_trajectory_age)
            ]

            print(
                f"⏸️ Pausing collection: all target weights {target_weights} "
                f"for weight version {self.current_weight_version} "
                f"already exist in buffer. Waiting for weight update..."
            )
            self._last_limit_warning_version = self.current_weight_version
            self._generation_limit_cleared.clear()

        self._generation_limit_cleared.wait()
        return self.running

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        if self.lag_mode == "unforced":
            print(
                "[unforced] Started continuous trajectory collection "
                f"(max in-flight rollouts={self.max_inflight_generations})"
            )
        else:
            print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        if self.lag_mode == "unforced":
            print("[unforced] Collection thread started, start_collection returning")
        else:
            print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                if not self._wait_for_manual_pause():
                    break
                if not self._wait_for_refit_pause(before_prompt=False):
                    break
                if not self._wait_for_generation_limit_clear():
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print(f"🛑 {self._log_prefix()}Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts = batch.size

            target_weight: Optional[int] = None
            if self.lag_mode == "forced":
                target_weight = self._get_next_target_for_generation(
                    generation_weight_version
                )

                if target_weight is None:
                    print(
                        f"🔄 No targets need generation for weight {generation_weight_version}"
                    )
                    return

                print(
                    f"🎯 Generating for target weight {target_weight} "
                    f"from generation_weight_version {generation_weight_version}"
                )

            for prompt_idx in range(num_prompts):
                if not self.running:
                    break
                if not self._wait_for_manual_pause():
                    break
                refit_was_paused = not self._refit_pause_cleared.is_set()
                if not self._wait_for_refit_pause(before_prompt=True):
                    break

                if self.lag_mode == "unforced" or refit_was_paused:
                    generation_weight_version = self.current_weight_version

                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)
                if not self._start_prompt_group_worker(
                    repeated_batch=repeated_batch,
                    generation_weight_version=generation_weight_version,
                    target_weight_version=target_weight,
                    prompt_idx=prompt_idx,
                ):
                    break

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def _start_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: Optional[int],
        prompt_idx: int,
    ) -> bool:
        self._inflight_sema.acquire()

        if not self.running:
            self._inflight_sema.release()
            return False

        worker = _threading.Thread(
            target=self._run_prompt_group_worker,
            args=(
                repeated_batch,
                generation_weight_version,
                target_weight_version,
                prompt_idx,
            ),
            daemon=True,
        )
        with self._threads_lock:
            self._inflight_threads.add(worker)
        worker.start()
        return True

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> int:
        """Pause trajectory collection. Returns the number of in-flight threads at pause time."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        with self._threads_lock:
            inflight = len(self._inflight_threads)
        if self.lag_mode == "unforced":
            print(
                f"[unforced] Trajectory collection paused ({inflight} threads still in flight)"
            )
        else:
            print(f"Trajectory collection paused ({inflight} threads still in flight)")
        return inflight

    def get_inflight_count(self) -> int:
        """Return the current number of in-flight generation threads."""
        with self._threads_lock:
            return len(self._inflight_threads)

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        if self.lag_mode == "unforced":
            print("[unforced] Trajectory collection resumed")
        else:
            print("Trajectory collection resumed")

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For vLLM V1 async engine, leverages in-flight weight updates via collective_rpc,
        allowing ongoing generations to continue with their current KV caches while
        weights are updated. This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using vLLM async engine
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
            # For non-async engines, wait for all pending generations to complete
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
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

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: Optional[int] = None,
        prompt_idx: Optional[int] = None,
    ) -> None:
        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            # Check if we should use nemo_gym (similar to synchronous GRPO)
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
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
            else:
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=self.master_config.grpo["max_rollout_turns"],
                    greedy=False,
                )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
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
                        if self.lag_mode == "unforced":
                            print(
                                f"📦 [unforced] Buffered prompt group "
                                f"(weight_version {generation_weight_version})"
                            )
                        else:
                            print(
                                "📦 Buffered per-prompt group "
                                f"(prompt_idx {prompt_idx}, "
                                f"target_weight {target_weight_version})"
                            )

                        # Release reservation when FIRST prompt group for this target is successfully buffered
                        if target_weight_version is not None and prompt_idx == 0:
                            with self._generation_check_lock:
                                if target_weight_version in self._generating_targets:
                                    self._generating_targets.discard(
                                        target_weight_version
                                    )
                                    print(
                                        "🧹 Released reservation for target weight "
                                        f"{target_weight_version} "
                                        "(first prompt buffered)"
                                    )
                        break
                    elif status == "full":
                        # Exponential backoff up to 0.5 second
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up reservation in case of error (if not already cleaned up)
            if target_weight_version is not None:
                with self._generation_check_lock:
                    if target_weight_version in self._generating_targets:
                        self._generating_targets.discard(target_weight_version)
                        print(
                            f"🧹 Emergency cleanup: Released reservation for target weight {target_weight_version}"
                        )

            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                import traceback

                traceback.print_exc()
