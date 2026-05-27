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
        self._post_refit_cleanup_pending = False

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
        # Track prompt-group reservations by target weight. A target is complete
        # only when buffered groups plus in-flight reservations reaches a full
        # training step.
        self._target_reservations: dict[int, int] = {}

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
        self, generation_weight_version: int, num_prompt_groups_to_reserve: int
    ) -> tuple[Optional[int], int]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        buffered_counts = ray.get(
            self.replay_buffer.get_target_weight_counts.remote()
        )
        target_group_goal = int(self.master_config.grpo["num_prompts_per_step"])

        with self._generation_check_lock:
            for target_weight in target_weights:
                buffered_count = buffered_counts.get(target_weight, 0)
                if buffered_count >= target_group_goal:
                    continue

                reserved_count = self._target_reservations.get(target_weight, 0)
                missing_count = target_group_goal - buffered_count - reserved_count

                if missing_count <= 0:
                    print(
                        f"🎯 Target weight {target_weight} is fully reserved "
                        f"({buffered_count} buffered, {reserved_count} reserved, goal {target_group_goal}); "
                        "waiting before advancing to later targets"
                    )
                    return None, 0

                if missing_count > 0:
                    reservation_count = min(
                        num_prompt_groups_to_reserve, missing_count
                    )
                    self._target_reservations[target_weight] = (
                        reserved_count + reservation_count
                    )
                    print(
                        f"🎯 Reserved {reservation_count} prompt groups for target weight {target_weight} "
                        f"({buffered_count} buffered, {reserved_count} already reserved, goal {target_group_goal})"
                    )
                    return target_weight, reservation_count

        return None, 0

    def _release_target_reservation(self, target_weight_version: int) -> None:
        with self._generation_check_lock:
            reserved_count = self._target_reservations.get(target_weight_version, 0)
            if reserved_count <= 1:
                self._target_reservations.pop(target_weight_version, None)
            else:
                self._target_reservations[target_weight_version] = reserved_count - 1
        self._generation_limit_cleared.set()

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
            buffered_counts = ray.get(
                self.replay_buffer.get_target_weight_counts.remote()
            )
            target_group_goal = int(self.master_config.grpo["num_prompts_per_step"])

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    buffered_count = buffered_counts.get(target_weight, 0)
                    if buffered_count >= target_group_goal:
                        continue

                    reserved_count = self._target_reservations.get(target_weight, 0)
                    if buffered_count + reserved_count >= target_group_goal:
                        print(
                            f"⏸️ Target weight {target_weight} has {buffered_count} buffered "
                            f"and {reserved_count} in-flight reservations; waiting before "
                            "starting later targets"
                        )
                        return True

                    return False

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
            )
            return True
        except Exception:
            return False

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
        if self.lag_mode == "unforced":
            self._collection_loop_unforced()
            return

        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                # Check if manually paused and wait
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                # Check if refit is in progress and wait
                if not self._refit_pause_cleared.is_set() and self.running:
                    print("⏸️ Pausing collection for refit...")
                    self._refit_pause_cleared.wait()
                    print("▶️ Refit completed, resuming collection")

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        target_weights = self._calculate_target_weights(
                            self.current_weight_version
                        )

                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            "are already buffered or reserved. Waiting before starting more generation..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                    self._generation_limit_cleared.clear()

                    # Efficiently wait for generation limits to be cleared (no polling!)
                    self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _collection_loop_unforced(self) -> None:
        """Run the unforced-lag collection loop in the background thread."""
        try:
            num_generations = self.master_config.grpo["num_generations_per_prompt"]

            for batch in self.dataloader:
                if not self.running:
                    break

                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                if not self._refit_pause_cleared.is_set() and self.running:
                    print("⏸️ [unforced] Pausing collection for refit...")
                    self._refit_pause_cleared.wait()
                    print("▶️ [unforced] Refit completed, resuming collection")

                if not self.running:
                    break

                for prompt_idx in range(batch.size):
                    if not self.running:
                        break

                    if not self._manual_pause_cleared.is_set() and self.running:
                        self._manual_pause_cleared.wait()

                    if not self._refit_pause_cleared.is_set() and self.running:
                        with self._threads_lock:
                            active_threads = len(self._inflight_threads)
                        print(
                            f"⏸️ [unforced] Waiting for refit before starting new generation "
                            f"({active_threads} threads still active)"
                        )
                        self._refit_pause_cleared.wait()

                    if not self.running:
                        break

                    repeated_batch = batch.slice(
                        prompt_idx, prompt_idx + 1
                    ).repeat_interleave(num_generations)

                    self._inflight_sema.acquire()

                    if not self.running:
                        self._inflight_sema.release()
                        break

                    generation_weight_version = self.current_weight_version

                    worker = _threading.Thread(
                        target=self._run_prompt_group_worker,
                        args=(repeated_batch, generation_weight_version),
                        daemon=True,
                    )
                    with self._threads_lock:
                        self._inflight_threads.add(worker)
                    worker.start()

                self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error in unforced trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 [unforced] Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts = batch.size

            # Get the next target weight that needs generation
            target_weight, reserved_prompt_groups = self._get_next_target_for_generation(
                generation_weight_version, num_prompts
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            print(
                f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            # Generate for all prompts in this batch for the target weight
            for prompt_idx in range(reserved_prompt_groups):
                # Wait for refit to complete if in progress
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
                    target=self._run_prompt_group_worker,
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
            import traceback

            traceback.print_exc()

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
        self._post_refit_cleanup_pending = True
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

    def resume_after_refit(self, resume_collection: bool = True) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
        async_cfg = self.master_config.grpo.get("async_grpo", {})
        # This cleanup is separate from unblocking collection so validation can keep
        # new training starts paused without rerunning post-refit work later.
        if self._post_refit_cleanup_pending:
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
            self._post_refit_cleanup_pending = False

        if resume_collection:
            self._refit_pause_cleared.set()
        else:
            print("⏸️ Keeping new generation starts paused after refit")

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
                    if self.lag_mode == "unforced":
                        status = ray.get(
                            self.replay_buffer.add.remote(
                                trajectory_group,
                                generation_weight_version,
                            )
                        )
                    else:
                        if target_weight_version is None:
                            raise ValueError(
                                "target_weight_version is required in forced-lag mode"
                            )
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
                                f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
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
            # Clean up the per-prompt reservation after success or failure.
            if self.lag_mode == "forced" and target_weight_version is not None:
                self._release_target_reservation(target_weight_version)

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
