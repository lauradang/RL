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
    """Interface for the replay buffer used in async RL training.

    The concrete `ReplayBuffer` class supports two `lag_mode` values
    (``"forced"`` and ``"unforced"``); this protocol covers the surface area
    used by the trajectory collector and trainer in either mode.
    """

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
            target_weight_version: version of the model weights this trajectory
                is intended for training. Required when the buffer is in
                ``lag_mode="forced"``; ignored in ``lag_mode="unforced"``,
                which stamps trajectories only with ``weight_version``.
        """
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups for the current training step.

        Behaviour depends on the buffer's ``lag_mode``:

        - Forced lag: returns trajectories whose ``target_weight_version``
          matches ``current_weight_version``; otherwise stalls (returns
          ``None``) so no trajectory loses its last-chance step.
        - Unforced lag: pops the oldest ``num_prompt_groups`` trajectories in
          FIFO order, lazily evicting any older than
          ``current_weight_version - max_age_steps``.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or
            ``None`` if insufficient data is available.
        """
        ...

    def evict(self) -> None:
        """Evict old trajectories."""
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...
