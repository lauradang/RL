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

"""Per-request serving latencies of a Megatron inference engine.

Megatron-LM already stamps every request: ``event_add_engine`` when it reaches
the engine, an ``ADD_CONTEXT`` event when it is first scheduled into the batch,
and ``ttft`` once its first token exists. The recorder reads those stamps off
each request as the engine finishes it, so the same numbers come out of an
aggregated engine and out of either side of a prefill/decode pair:

- aggregated: ``ttft`` is queue wait plus prefill; ``tpot`` is the decode pace.
- prefill: ``ttft`` is queue wait plus prefill; the request ends at its first
  token, so it has no ``tpot``.
- decode: the request arrives as a KV handoff, so ``queue`` is the wait for the
  KV pull and admission and ``ttft`` ends at the imported first token.

Only the engine's coordinator rank records; the other ranks see the same
requests.
"""

import threading
import time
from typing import Any, Optional

import numpy as np

# Columns of one finished request, in the order ``_rows`` stores them.
REQUEST_FIELDS = (
    "queue_s",
    "ttft_s",
    "tpot_s",
    "e2e_s",
    "prompt_tokens",
    "cached_tokens",
    "generated_tokens",
)
_PERCENTILES = (50, 90, 99)
_LATENCY_FIELDS = ("queue_s", "ttft_s", "tpot_s", "e2e_s")
# A drain queues behind whatever the worker actor is running (a refit, say);
# give up rather than hold the step's logging longer than this.
DRAIN_TIMEOUT_S = 60.0


class RequestTimingRecorder:
    """Collects one row per finished request on the engine's coordinator rank.

    ``attach`` wraps the engine's ``async_bookkeep``, the step where finished
    requests come out. ``drain`` hands the rows collected since the last drain
    to the caller, together with how long the engine was busy in that window.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._rows: list[tuple[float, ...]] = []
        self._failed = 0
        self._window_start = time.monotonic()
        self._busy_s = 0.0
        self._steps = 0
        # Monotonic time of the previous bookkeeping pass, kept only while the
        # engine still had requests in flight after it (the gap is then busy time).
        self._busy_since: Optional[float] = None

    @classmethod
    def attach(cls, engine: Any) -> "RequestTimingRecorder":
        recorder = cls(engine)
        bookkeep = engine.async_bookkeep

        async def async_bookkeep(*args: Any, **kwargs: Any) -> Any:
            result = await bookkeep(*args, **kwargs)
            if getattr(engine, "is_mp_coordinator", False):
                recorder._observe(result)
            return result

        engine.async_bookkeep = async_bookkeep
        return recorder

    def _observe(self, result: Any) -> None:
        now_wall = time.time()
        now = time.monotonic()
        rows = []
        failed = 0
        for request in result["finished_requests"]:
            if request.status is not None and request.status.name == "FAILED":
                failed += 1
                continue
            rows.append(_request_row(request, now_wall))
        with self._lock:
            self._steps += 1
            if self._busy_since is not None:
                self._busy_s += now - self._busy_since
            self._busy_since = now if result["active_request_ids"] else None
            self._rows.extend(rows)
            self._failed += failed

    def drain(self) -> dict[str, Any]:
        """Rows and engine activity since the previous drain, then reset."""
        now = time.monotonic()
        with self._lock:
            busy_s = self._busy_s
            if self._busy_since is not None:
                busy_s += now - self._busy_since
                self._busy_since = now
            drained = {
                "rows": self._rows,
                "failed": self._failed,
                "steps": self._steps,
                "busy_s": busy_s,
                "window_s": now - self._window_start,
            }
            self._rows = []
            self._failed = 0
            self._steps = 0
            self._busy_s = 0.0
            self._window_start = now
        return drained


def _request_row(request: Any, now_wall: float) -> tuple[float, ...]:
    arrival = request.event_add_engine.timestamp
    scheduled = next(
        (
            event.timestamp
            for event in request.events
            if event.type.name == "ADD_CONTEXT"
        ),
        None,
    )
    ttft = request.ttft
    generated = len(request.generated_tokens)
    tpot = np.nan
    if ttft is not None and generated > 1:
        tpot = (now_wall - arrival - ttft) / (generated - 1)
    prompt = request.prompt_tokens
    return (
        np.nan if scheduled is None else scheduled - arrival,
        np.nan if ttft is None else ttft,
        tpot,
        now_wall - arrival,
        0 if prompt is None else len(prompt),
        request.num_cached_tokens,
        generated,
    )


def summarize_request_timings(
    drained_by_role: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Flat ``<role>/<metric>`` scalars from every engine's ``drain()``.

    Latencies get p50/p90/p99 and the mean; ``e2e_s_max`` and
    ``e2e_tail_ratio`` (p99 over p50) show how far the slowest requests trail.
    ``busy_frac`` is the share of the window the engines had requests in
    flight, averaged over engines.
    """
    metrics: dict[str, float] = {}
    for role, drained in drained_by_role.items():
        if not drained:
            continue
        rows = [row for engine in drained for row in engine["rows"]]
        prefix = f"{role}/"
        metrics[prefix + "requests"] = float(len(rows))
        metrics[prefix + "failed_requests"] = float(
            sum(engine["failed"] for engine in drained)
        )
        metrics[prefix + "engines"] = float(len(drained))
        metrics[prefix + "busy_frac"] = float(
            np.mean(
                [
                    engine["busy_s"] / engine["window_s"]
                    for engine in drained
                    if engine["window_s"] > 0
                ]
                or [0.0]
            )
        )
        window_s = max(engine["window_s"] for engine in drained)
        if not rows:
            continue
        table = np.asarray(rows, dtype=np.float64).reshape(-1, len(REQUEST_FIELDS))
        columns = dict(zip(REQUEST_FIELDS, table.T))
        for name in _LATENCY_FIELDS:
            values = columns[name][~np.isnan(columns[name])]
            if values.size == 0:
                continue
            for q, value in zip(_PERCENTILES, np.percentile(values, _PERCENTILES)):
                metrics[f"{prefix}{name}_p{q}"] = float(value)
            metrics[f"{prefix}{name}_mean"] = float(values.mean())
        e2e = columns["e2e_s"]
        metrics[prefix + "e2e_s_max"] = float(e2e.max())
        p50 = float(np.percentile(e2e, 50))
        if p50 > 0:
            metrics[prefix + "e2e_tail_ratio"] = float(np.percentile(e2e, 99)) / p50
        prompt_total = columns["prompt_tokens"].sum()
        metrics[prefix + "prompt_tokens_mean"] = float(columns["prompt_tokens"].mean())
        metrics[prefix + "generated_tokens_mean"] = float(
            columns["generated_tokens"].mean()
        )
        if prompt_total > 0:
            metrics[prefix + "prefix_cache_hit_frac"] = float(
                columns["cached_tokens"].sum() / prompt_total
            )
        if window_s > 0:
            metrics[prefix + "generated_tokens_per_s"] = float(
                columns["generated_tokens"].sum() / window_s
            )
    return metrics
