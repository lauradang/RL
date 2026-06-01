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
from collections import Counter
from typing import Any, Iterable, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import (
    LagMode,
    ReplayBufferProtocol,
    validate_lag_mode,
)


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int, lag_mode: LagMode = "forced"):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.lag_mode = validate_lag_mode(lag_mode)
        self.trajectories: list[dict[str, Any]] = []
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions: list[
            int
        ] = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions: list[
            Optional[int]
        ] = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: Optional[int] = None,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training.
                Required in forced-lag mode and ignored in unforced-lag mode.
        """
        if self.lag_mode == "forced" and target_weight_version is None:
            raise ValueError(
                "ReplayBuffer in forced-lag mode requires `target_weight_version`."
            )

        stored_target_weight_version = (
            target_weight_version if self.lag_mode == "forced" else None
        )

        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(stored_target_weight_version)

            if self.lag_mode == "forced":
                assert stored_target_weight_version is not None
                self.last_target_weight_already_generated = max(
                    self.last_target_weight_already_generated,
                    stored_target_weight_version,
                )
                print(
                    "🔍 ReplayBuffer.add: "
                    f"size={len(self.trajectories)}, "
                    f"weight_version={weight_version}, "
                    f"target_weight_version={stored_target_weight_version}, "
                    f"versions={self.trajectory_versions}, "
                    f"targets={self.target_weight_versions}, "
                    "last_target_weight_already_generated="
                    f"{self.last_target_weight_already_generated}"
                )
            else:
                print(
                    f"🔍 ReplayBuffer(unforced).add: size={len(self.trajectories)}, "
                    f"weight_version={weight_version}, "
                    f"versions_in_buffer={self.trajectory_versions}"
                )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": []
            if self.lag_mode == "unforced"
            else self.target_weight_versions,
            "max_size": self.max_size,
        }

    def get_last_target_weight_already_generated(self) -> int:
        if self.lag_mode == "unforced":
            return -1
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        if self.lag_mode == "unforced":
            return set()
        with self._lock:
            return {v for v in self.target_weight_versions if v is not None}

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.target_weight_versions.pop(idx)
            self.trajectories.pop(idx)

    def _evict_stale(self, min_valid_version: int) -> None:
        """Evict stale trajectories."""
        stale_indices = [
            i for i, v in enumerate(self.trajectory_versions) if v < min_valid_version
        ]
        if not stale_indices:
            return

        self._remove_indices(stale_indices)
        print(
            f"🗑️ ReplayBuffer evicted {len(stale_indices)} stale trajectories "
            f"(older than weight_version {min_valid_version})"
        )

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups for the current training step.

        Forced-lag mode only returns trajectories with
        target_weight_version == current_weight_version. Unforced-lag mode
        evicts stale trajectories and returns the oldest remaining groups.
        If insufficient trajectories are available, returns None to stall
        training until enough groups are ready.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            min_valid_version = max(0, current_weight_version - max_age_steps)

            if self.lag_mode == "unforced":
                self._evict_stale(min_valid_version)
                available = len(self.trajectories)
                print(
                    f"🔍 ReplayBuffer(unforced).sample: requested={num_prompt_groups}, "
                    f"available={available}, current_weight_version={current_weight_version}, "
                    f"max_age_steps={max_age_steps}, min_valid_version={min_valid_version}"
                )
                if available < num_prompt_groups:
                    print(
                        f"   ⏸️ STALLING: have {available}, need {num_prompt_groups}"
                    )
                    return None
                selected = list(range(num_prompt_groups))
            else:
                print("🔍 ReplayBuffer sampling debug:")
                print(f"   {current_weight_version=}, {max_age_steps=}")
                print(f"   {self.trajectory_versions=}")

                version_counts = Counter(self.trajectory_versions)
                print(f"   {version_counts=}")
                print(f"   {min_valid_version=}")

                old_trajectories = [
                    v for v in self.trajectory_versions if v < min_valid_version
                ]
                if old_trajectories:
                    raise ValueError(
                        f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}"
                    )

                valid_indices = [
                    i
                    for i, v in enumerate(self.trajectory_versions)
                    if min_valid_version <= v <= current_weight_version
                ]
                print(
                    f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
                )
                if not valid_indices:
                    print("No trajectories available for sampling.")
                    return None

                if len(valid_indices) < num_prompt_groups:
                    print(
                        f"Insufficient valid groups: have {len(valid_indices)}, "
                        f"need {num_prompt_groups}. Waiting for buffer to fill."
                    )
                    return None

                intended_indices = [
                    i
                    for i in valid_indices
                    if self.target_weight_versions[i] == current_weight_version
                ]

                print(
                    f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
                )

                if len(intended_indices) < num_prompt_groups:
                    print(
                        f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories "
                        f"for step {current_weight_version}, but only "
                        f"{len(intended_indices)} are ready"
                    )
                    print(
                        "   ⏸️ Training will wait for remaining "
                        f"{num_prompt_groups - len(intended_indices)} trajectories "
                        "to be generated"
                    )
                    return None

                selected = intended_indices[:num_prompt_groups]
                print(
                    f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
                )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            if self.lag_mode == "unforced":
                print(
                    f"   ✅ Sampled FIFO counts by generation weight-version: "
                    f"{Counter(sampled_weights)}"
                )
            else:
                print(
                    f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
                )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            if self.lag_mode == "forced":
                print(
                    f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
                )

            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)
            if self.lag_mode == "unforced":
                print(
                    f"🗑️ Consumed and removed {len(selected)} groups from buffer "
                    f"(buffer size now: {len(self.trajectories)})"
                )
            else:
                print(
                    f"🗑️ Consumed and removed {len(selected)} groups from buffer, "
                    f"old buffer size: {total_trajectories}, "
                    f"new buffer size: {len(self.trajectories)}, "
                    f"new target weight versions {self.target_weight_versions}"
                )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


# WIP: DO NOT USE - This class is WIP and may be changed without notice, please DO NOT USE it.
# Will be replaced by TQReplayBuffer once TQ is ready.
@ray.remote  # pragma: no cover
class ReplayBufferNew(ReplayBufferImpl):
    """Staleness-window replay buffer.

    -- WIP: DO NOT USE --
    This class is WIP and may be changed without notice, please DO NOT USE it.

    Differences from ReplayBuffer:
    - _evict(): Stale rows (trainer_version - weight_version > max_staleness) are evicted
      at the start of every sample() call.
    - sample(): selects trajectories in freshest-first order (default) or FIFO order,
      controlled by the sample_freshest_first flag, from whatever remains in the buffer
      after eviction.

    TODO: remove when cleaning up
    - max_age_steps won't be used in ReplayBufferNew;
    - self.target_weight_versions won't be used in ReplayBufferNew and will be removed
      when cleaning up. target_weight_versions gates generation on specific trainer steps,
      which causes generation pauses; ReplayBufferNew intentionally avoids this.
    - add this class to nemo_rl/algorithms/async_utils/__init__.py
    """

    def __init__(
        self, max_size: int, max_staleness: int, sample_freshest_first: bool = True
    ):
        super().__init__(max_size)
        if max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative, got {max_staleness}")
        self.max_staleness = max_staleness
        # will move to StalenessSampler when we implement it
        self.sample_freshest_first = sample_freshest_first

    def _evict(self, current_weight_version: int) -> None:
        """Evict rows where trainer_version - weight_version > max_staleness.

        Must be called with self._lock held.
        """
        min_valid = current_weight_version - self.max_staleness
        stale = [i for i, v in enumerate(self.trajectory_versions) if v < min_valid]
        self._remove_indices(stale)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample num_prompt_groups trajectories, freshest-first.

        Will evict stale rows before sampling, so we will get [current_weight_version - self.max_staleness, current_weight_version] valid trajectories.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None.
        """
        with self._lock:
            self._evict(current_weight_version)

            if not self.trajectories:
                return None

            all_indices = range(len(self.trajectory_versions))
            if self.sample_freshest_first:
                all_indices = sorted(
                    all_indices,
                    key=lambda i: self.trajectory_versions[i],
                    reverse=True,
                )

            if len(all_indices) < num_prompt_groups:
                print(
                    f"Insufficient trajectories: have {len(all_indices)}, "
                    f"need {num_prompt_groups}. Waiting."
                )
                return None

            selected = all_indices[:num_prompt_groups]
            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )

            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }
