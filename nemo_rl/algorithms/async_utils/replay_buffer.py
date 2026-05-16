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
from typing import Any, Literal, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """Replay buffer storing per-prompt groups, supporting both forced- and unforced-lag modes.

    A single entry corresponds to 1 prompt repeated by
    ``grpo.num_generations_per_prompt`` (required to compute per-prompt
    advantages).

    Modes:

    - ``lag_mode="forced"`` (default): each entry is stamped with both a
      ``weight_version`` (the model that generated it) and a
      ``target_weight_version`` (the future training step it is reserved for).
      ``sample`` returns only entries whose ``target_weight_version`` matches
      the current trainer step, and stalls otherwise.
    - ``lag_mode="unforced"``: entries are stamped only with the generation
      ``weight_version``. ``sample`` returns the oldest valid entries in FIFO
      order; entries older than ``current_weight_version - max_age_steps`` are
      lazily evicted at sample time. Mirrors Megatron-LM's ``enforce_order=False``
      path.
    """

    def __init__(
        self,
        max_size: int,
        lag_mode: Literal["forced", "unforced"] = "forced",
    ):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if lag_mode not in ("forced", "unforced"):
            raise ValueError(
                f"lag_mode must be 'forced' or 'unforced', got {lag_mode!r}"
            )

        self.max_size = max_size
        self.lag_mode: Literal["forced", "unforced"] = lag_mode
        self._log_prefix = "" if lag_mode == "forced" else "[unforced] "

        self.trajectories: list[dict[str, Any]] = []
        self.trajectory_versions: list[int] = []
        self._lock = _threading.Lock()

        if self.lag_mode == "forced":
            # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1
            # was used for generating a trajectory and this trajectory will be used for training when
            # weight version is 4.
            self.target_weight_versions: list[int] = []
            self.last_target_weight_already_generated: int = -1

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
                Required for forced-lag mode; ignored in unforced-lag mode.
        """
        if self.lag_mode == "forced":
            return self._add_forced(trajectory, weight_version, target_weight_version)
        return self._add_unforced(trajectory, weight_version)

    def _add_forced(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: Optional[int],
    ) -> str:
        if target_weight_version is None:
            raise ValueError(
                "ReplayBuffer.add requires `target_weight_version` when lag_mode='forced'. "
                "Switch to lag_mode='unforced' for FIFO/unforced-lag mode."
            )
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, "
                f"versions={self.trajectory_versions}, "
                f"targets={self.target_weight_versions}, "
                f"last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def _add_unforced(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
    ) -> str:
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            print(
                f"🔍 {self._log_prefix}ReplayBuffer.add: size={len(self.trajectories)}, "
                f"weight_version={weight_version}, "
                f"versions_in_buffer={self.trajectory_versions}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        if self.lag_mode == "forced":
            target_weight_versions: list[int] = list(self.target_weight_versions)
        else:
            target_weight_versions = []
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": list(self.trajectory_versions),
            "target_weight_versions": target_weight_versions,
            "max_size": self.max_size,
            "lag_mode": self.lag_mode,
        }

    def get_last_target_weight_already_generated(self) -> int:
        if self.lag_mode != "forced":
            return -1
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        if self.lag_mode != "forced":
            return set()
        with self._lock:
            return set(self.target_weight_versions)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups.

        - Forced lag: only returns trajectories with
          ``target_weight_version == current_weight_version``; stalls (returns
          ``None``) otherwise so no trajectory loses its last-chance step.
        - Unforced lag: pops the oldest ``num_prompt_groups`` valid (within age
          window) trajectories in FIFO order; lazily evicts entries older than
          ``current_weight_version - max_age_steps``.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or
            ``None`` if insufficient data is available.
        """
        if self.lag_mode == "forced":
            return self._sample_forced(
                num_prompt_groups, current_weight_version, max_age_steps
            )
        return self._sample_unforced(
            num_prompt_groups, current_weight_version, max_age_steps
        )

    def _sample_forced(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
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
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # so no trajectory loses its "last chance" to be used for its intended step.
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
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            sampled_items = [self.trajectories[i] for i in selected]

            for idx in sorted(selected, reverse=True):
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
                self.trajectories.pop(idx)
            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, "
                f"old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, "
                f"new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def _sample_unforced(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            min_valid_version = max(0, current_weight_version - max_age_steps)

            evicted = 0
            while (
                self.trajectory_versions
                and self.trajectory_versions[0] < min_valid_version
            ):
                self.trajectories.pop(0)
                self.trajectory_versions.pop(0)
                evicted += 1
            if evicted:
                print(
                    f"🗑️ {self._log_prefix}ReplayBuffer evicted {evicted} stale trajectories "
                    f"(older than weight_version {min_valid_version})"
                )

            available = len(self.trajectories)
            print(
                f"🔍 {self._log_prefix}ReplayBuffer.sample: requested={num_prompt_groups}, "
                f"available={available}, current_weight_version={current_weight_version}, "
                f"max_age_steps={max_age_steps}, min_valid_version={min_valid_version}"
            )

            if available < num_prompt_groups:
                print(
                    f"   ⏸️ {self._log_prefix}STALLING: have {available}, need {num_prompt_groups}"
                )
                return None

            sampled_trajectories = self.trajectories[:num_prompt_groups]
            sampled_versions = self.trajectory_versions[:num_prompt_groups]
            del self.trajectories[:num_prompt_groups]
            del self.trajectory_versions[:num_prompt_groups]

            avg_trajectory_age = current_weight_version - (
                sum(sampled_versions) / len(sampled_versions)
            )

            print(
                f"   ✅ {self._log_prefix}Sampled FIFO counts by generation weight-version: "
                f"{Counter(sampled_versions)}"
            )
            print(
                f"📊 {self._log_prefix}Average trajectory age: {avg_trajectory_age:.2f} steps "
                f"(buffer size now: {len(self.trajectories)})"
            )

            return {
                "trajectories": sampled_trajectories,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def evict(self) -> None:
        """Evict old trajectories.

        No-op for both modes; forced-mode eviction happens on consumption,
        unforced-mode eviction happens lazily in ``sample``.
        """
        pass

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            if self.lag_mode == "forced":
                self.target_weight_versions.clear()


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass
