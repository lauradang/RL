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

from typing import Any, Optional, Protocol


class ReplayBufferProtocol(Protocol):
    """Interface for the replay buffer used in async RL training."""

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
                Required by forced-lag buffers; ignored by unforced-lag (FIFO) buffers, which
                accept `None` to support a uniform collector contract.
        """
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        ...

    def evict(self) -> None:
        """Evict old trajectories."""
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def get_target_weight_counts(self) -> dict[int, int]:
        """Return buffered trajectory counts by forced-lag target weight."""
        ...

    def get_last_consumed_target_weight_version(self) -> int:
        """Return the latest forced-lag target weight consumed by training."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...
