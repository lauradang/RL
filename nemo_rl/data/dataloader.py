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

from collections.abc import Iterator
from typing import Any, Protocol

from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.weights import TaskDataloaderState, TaskName, TaskQuota
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CyclingDataLoader:
    """Repeat a stateful dataloader until its consumer stops."""

    def __init__(self, dataloader: StatefulDataLoader) -> None:
        self.dataloader = dataloader

    def __iter__(self) -> Iterator[BatchedDataDict]:
        consecutive_empty_epochs = 0
        while True:
            produced_this_epoch = False
            for batch in self.dataloader:
                produced_this_epoch = True
                yield batch

            if produced_this_epoch:
                consecutive_empty_epochs = 0
            else:
                consecutive_empty_epochs += 1
                if consecutive_empty_epochs >= 2:
                    raise RuntimeError(
                        "Dataloader yielded no batches for two consecutive epochs"
                    )

    def state_dict(self) -> dict[str, Any]:
        return self.dataloader.state_dict()


class FiniteDataloader(Protocol):
    """Small interface consumed by SingleController's finite rollout pump."""

    def __iter__(self) -> Iterator[BatchedDataDict]: ...

    def __len__(self) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class FiniteWeightedDataloader:
    """Yield finite mixed batches whose per-task sizes encode a task quota.

    An epoch ends when the shortest task dataloader is exhausted. A subsequent
    call to ``iter`` starts the next epoch through the underlying stateful
    dataloaders, preserving SingleController's existing finite epoch semantics.
    """

    def __init__(
        self,
        dataloaders: dict[TaskName, StatefulDataLoader],
        task_quota: TaskQuota,
    ) -> None:
        if not dataloaders:
            raise ValueError("FiniteWeightedDataloader requires at least one task")
        if set(dataloaders) != set(task_quota):
            raise ValueError(
                "Dataloader tasks must exactly match quota tasks: "
                f"dataloaders={sorted(dataloaders)}, quota={sorted(task_quota)}"
            )
        if any(count <= 0 for count in task_quota.values()):
            raise ValueError(f"Every task quota must be positive, got {task_quota}")
        self.dataloaders = dataloaders
        self.task_quota = dict(task_quota)
        self._iterators: dict[TaskName, Iterator] | None = None
        self._batches_remaining = 0
        self._batches_yielded = 0
        self._resume_next_iter = False

    def __len__(self) -> int:
        return min(len(dataloader) for dataloader in self.dataloaders.values())

    def __iter__(self) -> "FiniteWeightedDataloader":
        if self._iterators is not None:
            return self
        self._iterators = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloaders.items()
        }
        if not self._resume_next_iter:
            self._batches_yielded = 0
        self._batches_remaining = len(self) - self._batches_yielded
        self._resume_next_iter = False
        return self

    def __next__(self) -> BatchedDataDict:
        if self._iterators is None:
            self.__iter__()
        if self._batches_remaining <= 0:
            self._iterators = None
            raise StopIteration
        assert self._iterators is not None
        batches = [next(self._iterators[name]) for name in self.dataloaders]
        self._batches_remaining -= 1
        self._batches_yielded += 1
        result = BatchedDataDict.from_batches(batches)
        expected = sum(self.task_quota.values())
        if result.size != expected:
            raise RuntimeError(
                f"Expected a {expected}-prompt weighted batch, got {result.size}"
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataloaders": {
                task_name: dataloader.state_dict()
                for task_name, dataloader in self.dataloaders.items()
            },
            "batches_yielded": self._batches_yielded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        dataloader_states: TaskDataloaderState = state["dataloaders"]
        unknown = set(dataloader_states) - set(self.dataloaders)
        if unknown:
            raise ValueError(f"Saved state contains unknown tasks: {sorted(unknown)}")
        for task_name, dataloader in self.dataloaders.items():
            if task_name in dataloader_states:
                dataloader.load_state_dict(dataloader_states[task_name])
        self._batches_yielded = int(state["batches_yielded"])
        self._iterators = None
        self._batches_remaining = 0
        self._resume_next_iter = True


class MultipleDataloaderWrapper:
    """Wrapper for multiple dataloaders.

    This wrapper is used to sample data from multiple dataloaders using a custom dataloader function.

    When a single dataloader is exhausted, the data iterator must be reset in the custom dataloader function (as demonstrated in `examples/custom_dataloader/custom_dataloader.py`).
    This design ensures that the MultipleDataloaderWrapper operates as an infinite iterator, where __next__() will not raise StopIteration and __len__() is not supported.
    """

    def __init__(
        self,
        expected_num_prompts: int,
        data_config: dict,
        dataloaders: dict[str, StatefulDataLoader],
    ):
        self.expected_num_prompts = expected_num_prompts
        self.data_config = data_config
        self.dataloaders = dataloaders

        # init data iterators
        self.data_iterators = {
            task_name: iter(dataloader) for task_name, dataloader in dataloaders.items()
        }

        # custom dataloader function to decide how to sample the data from the dataloaders
        self.custom_dataloader_func = self._load_custom_dataloader_func()
        # records to pass additional information to the custom dataloader function
        self.records = {}

    def _load_custom_dataloader_func(self):
        import sys
        from pathlib import Path

        from hydra.utils import get_method

        project_root_path = Path(__file__).absolute().parents[2]
        sys.path = [str(project_root_path)] + sys.path

        return get_method(self.data_config["custom_dataloader"])

    def __iter__(self):
        return self

    def __next__(self):
        # sample data from the dataloaders
        result, self.data_iterators = self.custom_dataloader_func(
            self.data_iterators, self.dataloaders, **self.records
        )

        # check if the number of prompts is expected
        assert len(result["message_log"]) == self.expected_num_prompts, (
            f"Expected {self.expected_num_prompts} prompts, but got {len(result['message_log'])}"
        )

        # reset records
        self.records = {}

        return result

    def set_records(self, records: dict):
        """Set the records for the custom dataloader.

        Records are used to pass additional information to the custom dataloader function to decide how to sample the data from the dataloaders.
        """
        self.records.update(records)
