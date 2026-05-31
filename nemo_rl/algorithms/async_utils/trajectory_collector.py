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
    AsyncNemoGymRolloutResult,
    merge_async_nemo_gym_rollout_results,
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationInterface,
)

TokenizerType = PreTrainedTokenizerBase
LagMode = Literal["forced", "unforced"]


class _TwoLaneSemaphore:
    """Counting semaphore with a high- and low-priority waiting lane.

    A freed permit is handed to a waiting high-priority (validation, priority 0)
    acquirer before any waiting low-priority (training) acquirer. Only reorders
    waiting acquirers; it cannot reclaim permits already held by in-flight rollouts.

    Invariant: ``_free > 0`` implies no parked waiters, so the fast path never lets a
    new acquirer jump ahead of a parked validation acquirer.
    """

    def __init__(self, value: int):
        self._lock = _threading.Lock()
        self._free = max(1, value)
        self._high_waiting = 0
        self._low_waiting = 0
        self._high_lane = _threading.Semaphore(0)
        self._low_lane = _threading.Semaphore(0)

    def acquire(self, priority: int = 1) -> None:
        with self._lock:
            if self._free > 0:
                self._free -= 1
                return
            if priority <= 0:
                self._high_waiting += 1
                lane = self._high_lane
            else:
                self._low_waiting += 1
                lane = self._low_lane
        # Block until a releaser hands us a permit (already accounted for in _free).
        lane.acquire()

    def release(self) -> None:
        with self._lock:
            self._free += 1
            if self._high_waiting > 0:
                self._high_waiting -= 1
                self._free -= 1
                self._high_lane.release()
            elif self._low_waiting > 0:
                self._low_waiting -= 1
                self._free -= 1
                self._low_lane.release()


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
        # Uses a two-lane gate so that when a permit frees up it is handed to a waiting
        # validation rollout before any waiting training rollout.
        max_inflight = (
            int(self.master_config.grpo["num_prompts_per_step"])
            * int(self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self.max_inflight_generations = max_inflight
        self._inflight_sema = _TwoLaneSemaphore(max_inflight)

        # Track how many held semaphore permits belong to training vs validation
        # rollouts, plus the peak held during the current window (reset around
        # validation). Used for wandb metrics on rollout-thread usage.
        self._inflight_stats_lock: _threading.Lock = _threading.Lock()
        self._training_inflight = 0
        self._validation_inflight = 0
        self._peak_training_inflight = 0
        self._peak_validation_inflight = 0

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

                        self._generation_limit_cleared.clear()  # Clear the event to pause

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

                    self._acquire_inflight_permit("training")

                    if not self.running:
                        self._release_inflight_permit("training")
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

            # Generate for all prompts in this batch for the target weight
            for prompt_idx in range(num_prompts):
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

                self._acquire_inflight_permit("training")
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

    def _acquire_inflight_permit(self, kind: Literal["training", "validation"]) -> None:
        """Acquire a shared ``_inflight_sema`` permit and count it by rollout kind."""
        self._inflight_sema.acquire(0 if kind == "validation" else 1)
        with self._inflight_stats_lock:
            if kind == "validation":
                self._validation_inflight += 1
                self._peak_validation_inflight = max(
                    self._peak_validation_inflight, self._validation_inflight
                )
            else:
                self._training_inflight += 1
                self._peak_training_inflight = max(
                    self._peak_training_inflight, self._training_inflight
                )

    def _release_inflight_permit(self, kind: Literal["training", "validation"]) -> None:
        """Release a shared ``_inflight_sema`` permit and decrement its counter."""
        with self._inflight_stats_lock:
            if kind == "validation":
                self._validation_inflight = max(0, self._validation_inflight - 1)
            else:
                self._training_inflight = max(0, self._training_inflight - 1)
        self._inflight_sema.release()

    def get_inflight_split(self) -> dict[str, int]:
        """Return live held-permit counts split by training vs validation rollouts."""
        with self._inflight_stats_lock:
            return {
                "training": self._training_inflight,
                "validation": self._validation_inflight,
            }

    def reset_peak_inflight(self) -> None:
        """Reset peak held-permit counters to the current live counts."""
        with self._inflight_stats_lock:
            self._peak_training_inflight = self._training_inflight
            self._peak_validation_inflight = self._validation_inflight

    def get_peak_inflight_split(self) -> dict[str, int]:
        """Return peak held-permit counts split by training vs validation rollouts."""
        with self._inflight_stats_lock:
            return {
                "training": self._peak_training_inflight,
                "validation": self._peak_validation_inflight,
            }

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

    def run_validation_rollouts(
        self,
        val_batch: BatchedDataDict[DatumSpec],
        generation_config: GenerationConfig,
        max_seq_len: Optional[int] = None,
    ) -> AsyncNemoGymRolloutResult:
        """Run NeMo-Gym validation rollouts through the same ``_inflight_sema`` used by training.

        Mirrors the training collection loop pattern: iterate over prompts in ``val_batch``,
        slice one prompt at a time, acquire ``_inflight_sema``, spawn a daemon worker, then
        join all workers and merge results. Each val prompt is intrinsically one ``/run``
        (validation does not repeat by ``num_generations``), so the per-permit unit is
        ``1 permit = 1 prompt = 1 /run``. System-wide concurrent ``/run`` cap during
        validation is therefore ``max_inflight``, which is stricter than training's
        ``max_inflight \u00d7 num_generations`` and ensures validation can never flood the
        engine even when in-flight training rollouts have not all drained.

        Val prompts queue on the semaphore alongside any remaining in-flight training
        rollouts, so the semaphore itself acts as the queue and no separate queue data
        structure is needed.

        The collector's internal pause / refit gating is intentionally skipped here: the
        caller is expected to have already paused the training collection loop, and
        validation must run regardless of any concurrent refit signalling.
        """
        num_prompts = val_batch.size
        if num_prompts == 0:
            raise ValueError("Cannot run validation rollouts on an empty batch.")

        per_prompt_results: list[Optional[AsyncNemoGymRolloutResult]] = [None] * num_prompts
        worker_errors: list[Optional[BaseException]] = [None] * num_prompts

        def _worker(prompt_idx: int) -> None:
            self._acquire_inflight_permit("validation")
            try:
                single_prompt_batch = val_batch.slice(prompt_idx, prompt_idx + 1)
                per_prompt_results[prompt_idx] = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=single_prompt_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    generation_config=generation_config,
                    max_seq_len=max_seq_len,
                    max_rollout_turns=None,
                    greedy=False,
                )
            except BaseException as e:
                worker_errors[prompt_idx] = e
            finally:
                self._release_inflight_permit("validation")

        workers = []
        for prompt_idx in range(num_prompts):
            worker = _threading.Thread(
                target=_worker, args=(prompt_idx,), daemon=True
            )
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()

        for prompt_idx, err in enumerate(worker_errors):
            if err is not None:
                raise RuntimeError(
                    f"Validation rollout for prompt {prompt_idx} failed"
                ) from err

        ordered_results = [r for r in per_prompt_results if r is not None]
        return merge_async_nemo_gym_rollout_results(
            ordered_results, tokenizer=self.tokenizer
        )

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

                        # Release reservation when FIRST prompt group for this target is successfully buffered
                        if self.lag_mode == "forced" and prompt_idx == 0:
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
            if self.lag_mode == "forced" and target_weight_version is not None:
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
                self._release_inflight_permit("training")
            except Exception:
                import traceback

                traceback.print_exc()
