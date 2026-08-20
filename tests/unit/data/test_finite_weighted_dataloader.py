# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.dataloader import FiniteWeightedDataloader


def _collate(rows):
    return {
        "value": [row["value"] for row in rows],
        "task_name": [row["task_name"] for row in rows],
    }


def _loader(task, size, batch_size):
    dataset = [{"value": i, "task_name": task} for i in range(size)]
    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )


def test_finite_weighted_loader_preserves_quota_and_shortest_epoch():
    loader = FiniteWeightedDataloader(
        {"fast": _loader("fast", 12, 3), "slow": _loader("slow", 4, 1)},
        {"fast": 3, "slow": 1},
    )
    assert len(loader) == 4
    batches = list(loader)
    assert len(batches) == 4
    assert all(batch["task_name"].count("fast") == 3 for batch in batches)
    assert all(batch["task_name"].count("slow") == 1 for batch in batches)


def test_finite_weighted_loader_state_round_trip():
    loader = FiniteWeightedDataloader(
        {"a": _loader("a", 4, 1), "b": _loader("b", 4, 1)},
        {"a": 1, "b": 1},
    )
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()

    restored = FiniteWeightedDataloader(
        {"a": _loader("a", 4, 1), "b": _loader("b", 4, 1)},
        {"a": 1, "b": 1},
    )
    restored.load_state_dict(state)
    iterator = iter(restored)
    batch = next(iterator)
    assert batch["value"] == [1, 1]
    assert len(list(iterator)) == 2
