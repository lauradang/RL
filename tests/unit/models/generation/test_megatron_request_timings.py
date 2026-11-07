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

"""Per-request serving latencies read off finished Megatron engine requests."""

import asyncio
from enum import Enum
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.megatron import request_timings
from nemo_rl.models.generation.megatron.request_timings import (
    RequestTimingRecorder,
    summarize_request_timings,
)

pytestmark = pytest.mark.mcore

_Status = Enum("_Status", ["COMPLETED", "FAILED"])
_Event = Enum("_Event", ["ADD_ENGINE", "ADD_CONTEXT", "GENERATED_TOKEN"])


def _request(
    *, arrival, scheduled, ttft, generated, prompt=10, cached=4, status="COMPLETED"
):
    events = [SimpleNamespace(type=_Event.ADD_ENGINE, timestamp=arrival)]
    if scheduled is not None:
        events.append(SimpleNamespace(type=_Event.ADD_CONTEXT, timestamp=scheduled))
        # A re-admission after eviction must not move the queue time.
        events.append(
            SimpleNamespace(type=_Event.ADD_CONTEXT, timestamp=scheduled + 50)
        )
    return SimpleNamespace(
        status=_Status[status],
        event_add_engine=events[0],
        events=events,
        ttft=ttft,
        generated_tokens=list(range(generated)),
        prompt_tokens=list(range(prompt)),
        num_cached_tokens=cached,
    )


class _Engine:
    def __init__(self, results, *, coordinator=True):
        self._results = iter(results)
        self.is_mp_coordinator = coordinator

    async def async_bookkeep(self, *args, **kwargs):
        return next(self._results)


def _step(engine):
    return asyncio.run(engine.async_bookkeep())


@pytest.fixture
def frozen_wall(monkeypatch):
    monkeypatch.setattr(request_timings.time, "time", lambda: 1000.0)


def test_recorder_reads_each_finished_request(frozen_wall):
    finished = _request(arrival=990.0, scheduled=991.0, ttft=2.0, generated=5)
    failed = _request(
        arrival=990.0, scheduled=None, ttft=None, generated=0, status="FAILED"
    )
    engine = _Engine(
        [{"finished_requests": [finished, failed], "active_request_ids": []}]
    )
    recorder = RequestTimingRecorder.attach(engine)

    _step(engine)
    drained = recorder.drain()

    # queue 1 s, ttft 2 s, first token at 992, 4 more tokens by 1000 -> 2 s each.
    assert drained["rows"] == [(1.0, 2.0, 2.0, 10.0, 10, 4, 5)]
    assert drained["failed"] == 1
    assert drained["steps"] == 1
    assert recorder.drain()["rows"] == []


def test_non_coordinator_rank_records_nothing(frozen_wall):
    request = _request(arrival=990.0, scheduled=991.0, ttft=2.0, generated=5)
    engine = _Engine(
        [{"finished_requests": [request], "active_request_ids": []}],
        coordinator=False,
    )
    recorder = RequestTimingRecorder.attach(engine)

    _step(engine)

    assert recorder.drain()["rows"] == []


def test_busy_time_counts_only_gaps_with_requests_in_flight(monkeypatch):
    clock = iter([0.0, 10.0, 14.0, 20.0, 25.0])
    monkeypatch.setattr(request_timings.time, "monotonic", lambda: next(clock))
    recorder = RequestTimingRecorder(engine=None)  # window starts at 0

    # Fed directly: patching time.monotonic would also patch asyncio's clock.
    recorder._observe({"finished_requests": [], "active_request_ids": [1]})  # 10
    recorder._observe({"finished_requests": [], "active_request_ids": []})  # 14: +4
    recorder._observe({"finished_requests": [], "active_request_ids": [2]})  # 20: idle
    drained = recorder.drain()  # 25: +5 still in flight

    assert drained["busy_s"] == pytest.approx(9.0)
    assert drained["window_s"] == pytest.approx(25.0)
    assert drained["steps"] == 3


def test_single_token_requests_have_no_tpot(frozen_wall):
    prefill = _request(arrival=999.0, scheduled=999.5, ttft=1.0, generated=1)
    engine = _Engine([{"finished_requests": [prefill], "active_request_ids": []}])
    recorder = RequestTimingRecorder.attach(engine)

    _step(engine)

    metrics = summarize_request_timings({"prefill": [recorder.drain()]})
    assert metrics["prefill/ttft_s_p50"] == pytest.approx(1.0)
    assert metrics["prefill/queue_s_p50"] == pytest.approx(0.5)
    assert "prefill/tpot_s_p50" not in metrics


def test_summary_is_per_role_and_pools_engines():
    def drained(rows, busy_s, window_s=10.0):
        return {
            "rows": rows,
            "failed": 0,
            "steps": 3,
            "busy_s": busy_s,
            "window_s": window_s,
        }

    decode = [
        drained([(0.1, 0.2, 0.01, float(e2e), 100, 50, 20)], busy_s=5.0)
        for e2e in range(1, 101)
    ]
    metrics = summarize_request_timings(
        {
            "decode": decode,
            "prefill": [drained([], busy_s=1.0)],
            "unused": [],
        }
    )

    assert metrics["decode/requests"] == 100
    assert metrics["decode/engines"] == 100
    assert metrics["decode/busy_frac"] == pytest.approx(0.5)
    assert metrics["decode/e2e_s_p50"] == pytest.approx(50.5)
    assert metrics["decode/e2e_s_max"] == 100
    assert metrics["decode/prefix_cache_hit_frac"] == pytest.approx(0.5)
    assert metrics["decode/generated_tokens_per_s"] == pytest.approx(200.0)
    # Prefill engines that finished nothing still report their activity.
    assert metrics["prefill/requests"] == 0
    assert metrics["prefill/busy_frac"] == pytest.approx(0.1)
    assert not any(name.startswith("unused/") for name in metrics)
