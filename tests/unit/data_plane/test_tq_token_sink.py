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

"""TQTokenSink / TQTokenSource against a live TQ backend.

Runs NeMo-Gym's installable conformance kit (golden call sequences →
byte-exact digests, manifests, and linearized rows) over the TransferQueue
implementations — the framework-CI half of Gym's published conformance
contract (the other half runs inside Gym itself, verifying its digest
implementation against the same golden_vectors.json) — plus the
protocol edges the kit does not cover (missing keys, stage failure shape).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSink as TokenSinkProtocol,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSource as TokenSourceProtocol,
)

from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    STAGING_FIELDS,
    ChainPrefixCache,
    TQMegatronPromptPreparer,
    TQMegatronTokenStager,
    TQTokenSink,
    TQTokenSource,
    _delta_align_minf_routing_indices,
    resolve_admission_prefix,
)
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
    fixture_names,
)

STAGING_PARTITION = "rollout_staging_test"

pytestmark = pytest.mark.nemo_gym


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.nemo_gym
def test_gym_staging_package_is_importable_in_the_nemo_gym_lane():
    """The --nemo-gym-only lane installs the extra; a missing staging package
    means the Gym pin moved off the capture branch and every capture test
    below has silently degraded to a skip."""
    import nemo_gym.token_id_capture.staging  # noqa: F401


def test_tq_sink_source_passes_gym_golden_vectors():
    """Gym publishes a fixed wire contract independent of this repo's own
    fixtures; this catches drift in Gym's digest scheme that
    token_capture_test_fixtures.py cannot, since it computes its expected
    digests by calling Gym's own digest functions."""
    from nemo_gym.token_id_capture.staging.conformance import assert_golden_vectors

    assert_golden_vectors()


@pytest.fixture()
def staging_partition(tq_client):
    tq_client.register_partition(
        partition_id=STAGING_PARTITION,
        fields=list(STAGING_FIELDS) + [ROUTED_EXPERTS_FIELD],
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    yield STAGING_PARTITION
    tq_client.clear_samples(sample_ids=None, partition_id=STAGING_PARTITION)


def test_implementations_satisfy_protocols(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    assert isinstance(sink, TokenSinkProtocol)
    assert isinstance(source, TokenSourceProtocol)


@pytest.mark.parametrize(
    "fixture_name", ["worked_example", "single_call", "mixed_weight_versions"]
)
def test_tq_sink_source_passes_conformance(tq_client, staging_partition, fixture_name):
    assert fixture_name in fixture_names()
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts(fixture_name)
    for record in records:
        assert sink.stage(record).ok
    snapshots = source.fetch([record.staging_key for record in records])
    # The source returns extras-free base snapshots; every base field (all
    # digest inputs) must round-trip byte-exactly.
    assert [snapshot.model_dump() for snapshot in snapshots] == [
        record.model_dump(exclude={"extras"}) for record in records
    ]


def test_fetch_missing_key_raises_keyerror(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError):
        source.fetch(["ghost_rollout/ghost_call"])


def test_fetch_for_finalization_is_small_typed_and_identity_preserving(
    tq_client, staging_partition
):
    class RecordingClient:
        def __init__(self, client):
            self.client = client
            self.select_fields = None

        def get_samples(self, **kwargs):
            self.select_fields = list(kwargs["select_fields"])
            return self.client.get_samples(**kwargs)

    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    assert sink.stage(records[0]).ok
    recording_client = RecordingClient(tq_client)
    source = TQTokenSource(recording_client, staging_partition=staging_partition)

    fetched = source.fetch_for_finalization([records[0].staging_key])

    assert recording_client.select_fields == STAGING_FIELDS
    assert "routed_experts" not in recording_client.select_fields
    assert len(fetched) == 1
    assert fetched[0].staging_key == records[0].staging_key
    assert fetched[0].snapshot.model_call_id == records[0].model_call_id
    assert fetched[0].routed_len == 0
    assert fetched[0].fragment is None
    # The snapshot is a normally validated base model, never model_construct'd.
    from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

    assert type(fetched[0].snapshot) is StagedCallBaseSnapshot
    assert not hasattr(fetched[0].snapshot, "extras")


def test_fetch_for_finalization_rejects_duplicate_request_keys(
    tq_client, staging_partition
):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError, match="duplicate keys"):
        source.fetch_for_finalization(["r/c", "r/c"])


def test_stage_failure_reports_not_raises(staging_partition):
    class ExplodingClient:
        def put_samples(self, **kwargs):
            raise RuntimeError("controller down")

    sink = TQTokenSink(ExplodingClient(), staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    result = sink.stage(records[0])
    assert not result.ok
    assert result.staging_key == records[0].staging_key
    assert "controller down" in (result.error or "")


def test_sink_clear_drops_rows(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    for record in records:
        assert sink.stage(record).ok
    keys = [record.staging_key for record in records]
    assert len(source.fetch(keys)) == len(keys)
    sink.clear(keys)
    with pytest.raises(KeyError):
        source.fetch(keys)


def test_fetch_prefix_token_ids_empty(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    assert source.fetch_prefix_token_ids([]) == []


def test_fetch_prefix_token_ids_single_key(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    record = records[0]
    assert sink.stage(record).ok
    result = source.fetch_prefix_token_ids([record.staging_key])
    assert result == record.token_ids_delta


def test_fetch_prefix_token_ids_three_keys_concatenates(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("worked_example")
    for record in records:
        assert sink.stage(record).ok
    keys = [record.staging_key for record in records]
    result = source.fetch_prefix_token_ids(keys)
    expected = [t for record in records for t in record.token_ids_delta]
    assert result == expected


def test_fetch_prefix_token_ids_missing_key_raises_keyerror(
    tq_client, staging_partition
):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError):
        source.fetch_prefix_token_ids(["ghost_rollout/ghost_call"])


def test_fetch_prefix_token_ids_rejects_duplicates(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError, match="duplicates"):
        source.fetch_prefix_token_ids(["r/c", "r/c"])


def test_megatron_stager_writes_canonical_row_and_returns_coords(
    tq_client, staging_partition
):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    payload = SimpleNamespace(
        prompt_token_ids=[10, 11],
        generated_token_ids=[12, 13],
        generated_log_probs=[-0.25, -0.5],
        routing_indices=torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
            dtype=torch.int32,
        ),
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )

    result = stager.stage(
        "minf-response-1",
        payload,
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["staging_key"] == "minf-r0/c1"
    assert coords["weight_version"] == 7
    assert coords["disposition"] == "staged"
    [snapshot] = TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
        ["minf-r0/c1"]
    )
    assert snapshot.token_ids_delta == [10, 11, 12, 13]
    assert snapshot.token_mask_delta == [0.0, 0.0, 1.0, 1.0]
    assert snapshot.generation_log_probs_delta == [0.0, 0.0, -0.25, -0.5]
    [fetched] = TQTokenSource(
        tq_client, staging_partition=staging_partition
    ).fetch_for_finalization(["minf-r0/c1"], include_route_fragments=True)
    assert fetched.routed_len == 4
    assert fetched.fragment is not None
    assert fetched.fragment.routes.tolist() == [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        [[-1, -1], [-1, -1]],
    ]
    [without_routes] = TQTokenSource(
        tq_client, staging_partition=staging_partition
    ).fetch_for_finalization(["minf-r0/c1"])
    # routed_len is transport metadata carried even when the route payload is
    # left in TQ (deferred finalization); only the fragment is omitted.
    assert without_routes.routed_len == 4
    assert without_routes.fragment is None


@pytest.mark.parametrize(
    ("routes", "total_tokens", "prev_len", "match"),
    [
        pytest.param(
            torch.zeros((2, 2), dtype=torch.int32),
            3,
            0,
            r"shape \[tokens, layers, topk\]",
            id="rank",
        ),
        pytest.param(
            torch.zeros((1, 1, 2), dtype=torch.int32),
            3,
            0,
            "one row for every non-final token",
            id="row-count",
        ),
        pytest.param(
            torch.zeros((2, 0, 2), dtype=torch.int32),
            3,
            0,
            "dimensions must be positive",
            id="zero-layers",
        ),
        pytest.param(
            torch.zeros((2, 1, 0), dtype=torch.int32),
            3,
            0,
            "dimensions must be positive",
            id="zero-topk",
        ),
        pytest.param(
            torch.zeros((2, 1, 2), dtype=torch.int32),
            3,
            4,
            "prev_len must be in",
            id="prev-len",
        ),
    ],
)
def test_delta_align_minf_routing_indices_rejects_malformed_routes(
    routes, total_tokens, prev_len, match
):
    with pytest.raises(ValueError, match=match):
        _delta_align_minf_routing_indices(
            routes,
            total_tokens=total_tokens,
            prev_len=prev_len,
        )


def test_delta_align_minf_routing_indices_checks_model_route_dims():
    routes = torch.zeros((3, 4, 2), dtype=torch.int32)

    aligned = _delta_align_minf_routing_indices(
        routes, total_tokens=4, prev_len=0, expected_route_dims=(4, 2)
    )
    assert tuple(aligned.shape) == (4, 4, 2)

    # One pipeline stage's worth of layers instead of the whole model's.
    with pytest.raises(ValueError, match=r"got \(4, 2\), expected \(8, 2\)"):
        _delta_align_minf_routing_indices(
            routes, total_tokens=4, prev_len=0, expected_route_dims=(8, 2)
        )


def test_megatron_stager_rejects_misaligned_routes(tq_client, staging_partition):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )

    result = stager.stage(
        "minf-response-1",
        SimpleNamespace(
            prompt_token_ids=[10, 11],
            generated_token_ids=[12],
            generated_log_probs=[-0.25],
            routing_indices=torch.tensor([[[1, 2]]], dtype=torch.int32),
        ),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )

    assert result is not None
    assert (
        result.response_metadata["ng_commit_coords"]["disposition"] == "capture_failed"
    )


def test_megatron_stager_delta_aligns_token_in_routes(tq_client, staging_partition):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition),
        require_routed_experts=True,
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=3,
        mode="token_in",
        required_prefix_token_ids=[10, 11, 12],
        parent_chain_hash="00" * 32,
    )
    routes = torch.arange(5 * 2 * 2, dtype=torch.int32).reshape(5, 2, 2)

    result = stager.stage(
        "minf-response-2",
        SimpleNamespace(
            prompt_token_ids=[10, 11, 12, 13],
            generated_token_ids=[14, 15],
            generated_log_probs=[-0.25, -0.5],
            routing_indices=routes,
        ),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    [fetched] = TQTokenSource(
        tq_client, staging_partition=staging_partition
    ).fetch_for_finalization(["minf-r0/c2"], include_route_fragments=True)
    assert fetched.snapshot.token_ids_delta == [13, 14, 15]
    assert fetched.routed_len == 3
    assert fetched.fragment is not None
    assert fetched.fragment.routes.tolist() == [
        routes[3].tolist(),
        routes[4].tolist(),
        [[-1, -1], [-1, -1]],
    ]


def test_megatron_stager_requires_routes_when_router_replay_is_enabled(
    tq_client, staging_partition, caplog
):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition),
        require_routed_experts=True,
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )

    with caplog.at_level("ERROR", logger="nemo_rl.data_plane.tq_token_sink"):
        result = stager.stage(
            "minf-response-1",
            SimpleNamespace(
                prompt_token_ids=[10],
                generated_token_ids=[11],
                generated_log_probs=[-0.1],
                routing_indices=None,
            ),
            finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
            offload_params={"ng_capture": admission.model_dump(mode="json")},
        )

    assert result is not None
    assert (
        result.response_metadata["ng_commit_coords"]["disposition"] == "capture_failed"
    )
    # Any exception inside stage() poisons the call identically; pin the cause.
    assert "carries no routing_indices" in caplog.text


@pytest.mark.parametrize("prefix_source", ["staging_chain", "capture_admission"])
def test_megatron_prompt_preparer_splices_resolved_prefix(
    tq_client, staging_partition, prefix_source
):
    admission_kwargs = {}
    if prefix_source == "staging_chain":
        stager = TQMegatronTokenStager(
            TQTokenSink(tq_client, staging_partition=staging_partition)
        )
        root = nemo_gym.CaptureAdmission(
            rollout_id="minf-r0", model_call_id="c1", mode="text"
        )
        root_result = stager.stage(
            "minf-response-1",
            SimpleNamespace(
                prompt_token_ids=[10, 11],
                generated_token_ids=[12, 99],
                generated_log_probs=[-0.25, -0.5],
            ),
            finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
            offload_params={"ng_capture": root.model_dump(mode="json")},
        )
        assert root_result is not None
        root_coords = root_result.response_metadata["ng_commit_coords"]
        admission_kwargs = {
            "staging_chain": [root_coords["staging_key"]],
            "parent_chain_hash": root_coords["chain_hash"],
        }
    else:
        admission_kwargs = {
            "required_prefix_token_ids": [10, 11, 12, 99],
            "parent_chain_hash": _digest("chain:c1"),
        }

    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=4,
        mode="token_in",
        **admission_kwargs,
    )
    preparer = TQMegatronPromptPreparer(
        TQTokenSource(tq_client, staging_partition=staging_partition)
    )

    result = preparer.prepare_prompt(
        [80, 81, 99, 20, 21],
        offload_params={
            "ng_capture": admission.model_dump(mode="json"),
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [80, 81, 99],
            PREFIX_EOS_TOKEN_ID_FIELD: 99,
        },
    )

    assert result.prompt == [10, 11, 12, 99, 20, 21]
    assert result.offload_params is not None
    assert result.offload_params["ng_capture"]["required_prefix_token_ids"] == [
        10,
        11,
        12,
        99,
    ]


def test_megatron_stager_stamps_admission_epoch_when_request_spans_refit(
    tq_client, staging_partition, caplog
):
    """A request straddling a refit is stamped with its admission epoch, not masked.

    Mirrors vLLM, which freezes the version at begin_call. The engine stamps the
    admission epoch first and appends a boundary per refit, so epochs only grow.
    """
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )
    with caplog.at_level("WARNING", logger="nemo_rl.data_plane.tq_token_sink"):
        result = stager.stage(
            "minf-response-1",
            SimpleNamespace(
                prompt_token_ids=[10],
                generated_token_ids=[11, 12],
                generated_log_probs=[-0.1, -0.2],
            ),
            finished_metadata=SimpleNamespace(policy_epoch=[(0, 7), (1, 8), (2, 9)]),
            offload_params={"ng_capture": admission.model_dump(mode="json")},
        )
    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert coords["weight_version"] == 7
    assert stager.epoch_span_count == 1
    assert any("spans policy epochs [7, 8, 9]" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "missing_field",
    ["prompt_token_ids", "generated_token_ids", "generated_log_probs"],
)
def test_megatron_stager_poisons_malformed_payloads_with_capture_failed(
    tq_client, staging_partition, missing_field
):
    """Extraction errors return ``capture_failed`` coords, not ``None``.

    Gym maps returned failed coords to ``worker_capture_failed`` (as for
    vLLM); a ``None`` result would instead surface as
    ``worker_response_missing_commit_coordinates``.
    """
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    fields = {
        "prompt_token_ids": [10, 11],
        "generated_token_ids": [12, 13],
        "generated_log_probs": [-0.25, -0.5],
    }
    del fields[missing_field]
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )

    result = stager.stage(
        "minf-response-1",
        SimpleNamespace(**fields),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "capture_failed"
    assert coords["weight_version"] == 7
    with pytest.raises(KeyError):
        TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
            ["minf-r0/c1"]
        )


@pytest.mark.parametrize(
    ("with_capture_metadata", "policy_epoch"),
    [
        pytest.param(False, [(0, 7)], id="missing-capture-metadata"),
        pytest.param(True, [], id="no-policy-epoch-boundaries"),
        pytest.param(True, [(0, "x")], id="invalid-policy-epoch"),
        pytest.param(True, [(0, -1)], id="negative-policy-epoch"),
    ],
)
def test_megatron_stager_declines_ineligible_requests(
    tq_client, staging_partition, with_capture_metadata, policy_epoch
):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )
    result = stager.stage(
        "minf-response-1" if with_capture_metadata else "ordinary-request",
        SimpleNamespace(
            prompt_token_ids=[10],
            generated_token_ids=[11],
            generated_log_probs=[-0.1],
        ),
        finished_metadata=SimpleNamespace(policy_epoch=policy_epoch),
        offload_params=(
            {"ng_capture": admission.model_dump(mode="json")}
            if with_capture_metadata
            else None
        ),
    )
    assert result is None


class _RecordingSource:
    """Stand-in for TQTokenSource: records fetched keys, returns 2 tokens per key."""

    def __init__(self):
        self.calls = []

    def fetch_prefix_token_ids(self, keys):
        self.calls.append(list(keys))
        return [int(k[1:]) * 10 + i for k in keys for i in range(2)]


def test_chain_prefix_cache_fetches_only_uncached_suffix():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)

    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert cache.fetch(["k1", "k2", "k3"]) == [10, 11, 20, 21, 30, 31]
    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert source.calls == [["k1", "k2"], ["k3"]]


def test_chain_prefix_cache_requires_an_installed_source():
    cache = ChainPrefixCache()
    with pytest.raises(RuntimeError, match="setup_token_capture"):
        cache.fetch(["k1"])
    source = _RecordingSource()
    cache.install(source)
    assert cache.fetch(["k1"]) == [10, 11]


def test_chain_prefix_cache_evicts_oldest_insertion_past_256_entries():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    for i in range(257):
        cache.fetch([f"k{i}"])
    # k0 was the first insertion and is gone; k1 is still a hit.
    calls_before = len(source.calls)
    cache.fetch(["k1"])
    assert len(source.calls) == calls_before
    cache.fetch(["k0"])
    assert len(source.calls) == calls_before + 1


def test_resolve_admission_prefix_dispatches_like_the_vllm_worker():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    text = SimpleNamespace(mode="text", staging_chain=[], required_prefix_token_ids=[])
    inline = SimpleNamespace(
        mode="token_in", staging_chain=[], required_prefix_token_ids=[7, 8]
    )
    chained = SimpleNamespace(
        mode="token_in", staging_chain=["k1"], required_prefix_token_ids=[]
    )

    assert resolve_admission_prefix(text, cache) == []
    assert resolve_admission_prefix(inline, cache) == [7, 8]
    assert resolve_admission_prefix(chained, cache) == [10, 11]
    assert source.calls == [["k1"]]


def test_megatron_preparer_resolves_chains_through_the_shared_cache():
    source = _RecordingSource()
    preparer = TQMegatronPromptPreparer(source)
    assert isinstance(preparer._chain_prefix, ChainPrefixCache)

    child = nemo_gym.CaptureAdmission(
        rollout_id="r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=2,
        mode="token_in",
        staging_chain=["k1"],
        parent_chain_hash="a" * 64,
    )
    grandchild = nemo_gym.CaptureAdmission(
        rollout_id="r0",
        model_call_id="c3",
        parent_call_id="c2",
        prev_len=4,
        mode="token_in",
        staging_chain=["k1", "k2"],
        parent_chain_hash="b" * 64,
    )
    for admission, prompt, template_prefix in (
        (child, [80, 99, 5], [80, 99]),
        (grandchild, [80, 81, 82, 99, 6], [80, 81, 82, 99]),
    ):
        preparer.prepare_prompt(
            prompt,
            offload_params={
                "ng_capture": admission.model_dump(mode="json"),
                PREFIX_TEMPLATE_TOKEN_IDS_FIELD: template_prefix,
                PREFIX_EOS_TOKEN_ID_FIELD: 99,
            },
        )
    # k1 was cached by the child call; the grandchild fetched only k2.
    assert source.calls == [["k1"], ["k2"]]


def test_prefix_field_keys_match_megatron_constants():
    """The endpoint writes Megatron's constants; the preparer reads NeMo-RL's copies."""
    mcore = pytest.importorskip("megatron.core.inference.inference_request")
    if not hasattr(mcore, "PREFIX_TEMPLATE_TOKEN_IDS_FIELD"):
        pytest.skip(
            "pinned megatron-core predates MInf prefix-splice metadata (Megatron-LM #7015)"
        )
    assert PREFIX_TEMPLATE_TOKEN_IDS_FIELD == mcore.PREFIX_TEMPLATE_TOKEN_IDS_FIELD
    assert PREFIX_EOS_TOKEN_ID_FIELD == mcore.PREFIX_EOS_TOKEN_ID_FIELD
