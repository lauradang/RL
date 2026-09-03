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
"""TransferQueue implementations of NeMo-Gym's token staging protocols.

``TQStagingStore`` is NeMo RL's single keyed-row transport for token custody.
The vLLM-facing ``TQTokenSink`` codec writes canonical Gym call deltas, while
the MInf-facing ``TQRequestPayloadStager`` codec writes engine-native payloads
under response UIDs. Their matching sources read the rows back for the
finalizer. This module is the only hot-path file that knows tokens live in TQ;
Gym sees opaque staging keys.

Each staged row carries three jagged columns (``token_ids_delta``,
``token_mask_delta``, ``generation_logprobs_delta``), the complete receipt
identity/lineage metadata, and all digest inputs so it round-trips to a
complete ``StagedCallSnapshot``. Masks/logprobs are float32 on the wire,
matching ``compute_staging_digest``'s float32-bit-pattern scheme, so the
finalizer's digest recomputation over fetched values is byte-exact.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import ray
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    # Deferred: nemo_gym is an optional extra absent in non-gym runs; runtime
    # uses import locally so this module (and the finalizer actor importing
    # it) stays importable without it.
    from nemo_gym.token_id_capture.staging.records import (
        StagedCallRecord,
        StagedCallSnapshot,
        StageResult,
    )

from nemo_rl.data_plane.schema import (
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)

STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
    "schema_version",
    "digest_version",
    "extras_digest_version",
    "rollout_id_utf8",
    "model_call_id_utf8",
    "parent_call_id_utf8",
    "parent_call_id_present",
    "capture_mode",
    "prev_len",
    "delta_len",
    "cum_len",
    "weight_version",
    "digest_bytes",
    "extras_digest_bytes",
    "chain_hash_bytes",
    "chain_hash_present",
    "cumulative_hash_bytes",
    "cumulative_hash_present",
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_LEN_FIELD,
]

# MInf's RequestPayloadStager writes the engine-native completion payload
# immediately, before the stripped HTTP response is released.  These columns
# intentionally live in the same partition as the canonical vLLM rows: only
# one backend is active in a run, and finalization selects the corresponding
# schema explicitly.
MINF_PAYLOAD_FIELDS = [
    "minf_prompt_token_ids",
    "minf_generated_token_ids",
    "minf_generated_logprobs",
    "minf_weight_version",
]
MINF_OPTIONAL_PAYLOAD_FIELDS = [
    "minf_prompt_logprobs",
    "minf_routing_indices",
    "minf_routing_shape",
]

_ROUTE_ENCODING_NONE = 0
_ROUTE_ENCODING_ENVELOPE = 1
_ROUTE_ENCODING_LIST = 2
_MODE_TO_CODE = {"text": 0, "token_in": 1}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_TO_CODE.items()}


def _bytes_tensor(value: bytes) -> torch.Tensor:
    """Encode non-empty bytes as one jagged TQ row."""
    if not value:
        raise ValueError("staging byte fields must be non-empty")
    return torch.tensor([list(value)], dtype=torch.uint8)


def _optional_digest_fields(value: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _bytes_tensor(bytes.fromhex(value) if value is not None else bytes(32)),
        torch.tensor([value is not None], dtype=torch.bool),
    )


@dataclass(frozen=True)
class FetchedStagedCall:
    """One explicitly identified small-column finalization fetch result."""

    staging_key: str
    snapshot: StagedCallSnapshot
    routed_len: int


@dataclass(frozen=True)
class FetchedMInfPayload:
    """One MInf request payload read from TQ by its response UID."""

    request_uid: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_log_probs: list[float]
    prompt_log_probs: list[float] | None
    routing_indices: torch.Tensor | None
    weight_version: int


def _call_dp(dp_client: Any, method_name: str, **kwargs: Any) -> Any:
    """Call a DataPlaneClient method on a local client or a Ray actor handle."""
    method = getattr(dp_client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return ray.get(remote(**kwargs))
    return method(**kwargs)


class TQStagingStore:
    """Shared keyed-row transport for all token-capture TQ codecs."""

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def put(
        self,
        key: str,
        field_dict: dict[str, torch.Tensor],
        *,
        tags: dict[str, Any] | None = None,
    ) -> None:
        _call_dp(
            self._dp_client,
            "put_samples",
            sample_ids=[key],
            partition_id=self._staging_partition,
            fields=TensorDict(field_dict, batch_size=[1]),
            tags=[tags or {}],
        )

    def get(self, keys: list[str], *, select_fields: list[str]) -> TensorDict:
        return _call_dp(
            self._dp_client,
            "get_samples",
            sample_ids=list(keys),
            partition_id=self._staging_partition,
            select_fields=list(select_fields),
        )

    def clear(self, keys: list[str]) -> None:
        if not keys:
            return
        _call_dp(
            self._dp_client,
            "clear_samples",
            sample_ids=list(keys),
            partition_id=self._staging_partition,
        )


class TQRequestPayloadStager:
    """MInf ``RequestPayloadStager`` backed by the shared TQ row store.

    MInf's protocol has no failure return value.  Consequently ``stage``
    deliberately propagates TQ failures: the engine must not acknowledge a
    stripped response unless its payload is durable.
    """

    def __init__(
        self,
        store: TQStagingStore,
        *,
        weight_version: int = 0,
        weight_version_fn: Callable[[str], int] | None = None,
    ) -> None:
        self._store = store
        self._weight_version = 0
        self._weight_version_fn = weight_version_fn
        self._lock = threading.Lock()
        self.set_weight_version(weight_version)

    def set_weight_version(self, version: int) -> None:
        if type(version) is not int or version < 0:
            raise ValueError(
                f"rollout weight version must be a non-negative int, got {version!r}"
            )
        with self._lock:
            self._weight_version = version

    def stage(self, uid: str, payload: Any) -> None:
        if not isinstance(uid, str) or not uid:
            raise ValueError("MInf request UID must be a non-empty string")
        prompt_token_ids = getattr(payload, "prompt_token_ids", None)
        generated_token_ids = getattr(payload, "generated_token_ids", None)
        generated_log_probs = getattr(payload, "generated_log_probs", None)
        if prompt_token_ids is None:
            raise ValueError("MInf offloaded payload carries no prompt_token_ids")
        if generated_token_ids is None:
            raise ValueError("MInf offloaded payload carries no generated_token_ids")
        if generated_log_probs is None:
            raise ValueError("MInf offloaded payload carries no generated_log_probs")
        if len(generated_token_ids) != len(generated_log_probs):
            raise ValueError(
                "MInf generated token and log-probability lengths differ: "
                f"{len(generated_token_ids)} != {len(generated_log_probs)}"
            )

        prompt_log_probs = getattr(payload, "prompt_log_probs", None)
        routing_indices = getattr(payload, "routing_indices", None)
        routing = None
        routing_shape = None
        if routing_indices is not None:
            routing = torch.as_tensor(routing_indices)
            if routing.dim() != 3:
                raise ValueError(
                    "MInf routing_indices must have shape [tokens, layers, topk], "
                    f"got {tuple(routing.shape)}"
                )
            expected_routes = len(prompt_token_ids) + len(generated_token_ids) - 1
            if routing.shape[0] != expected_routes:
                raise ValueError(
                    "MInf routing_indices must contain one row for every "
                    "non-final token: "
                    f"got {routing.shape[0]}, expected {expected_routes}"
                )
            if routing.shape[1] <= 0 or routing.shape[2] <= 0:
                raise ValueError(
                    "MInf routing_indices layer and top-k dimensions must be positive"
                )
            routing_shape = tuple(int(size) for size in routing.shape)
            routing = routing.to(dtype=torch.int32).reshape(-1)

        with self._lock:
            weight_version = (
                self._weight_version_fn(uid)
                if self._weight_version_fn is not None
                else self._weight_version
            )
            if type(weight_version) is not int or weight_version < 0:
                raise ValueError(
                    "MInf payload weight version must be a non-negative int, "
                    f"got {weight_version!r}"
                )
            fields = {
                "minf_prompt_token_ids": torch.tensor(
                    [list(prompt_token_ids)], dtype=torch.int64
                ),
                "minf_generated_token_ids": torch.tensor(
                    [list(generated_token_ids)], dtype=torch.int64
                ),
                "minf_generated_logprobs": torch.tensor(
                    [list(generated_log_probs)], dtype=torch.float32
                ),
                "minf_weight_version": torch.tensor(
                    [weight_version], dtype=torch.int64
                ),
            }
            if prompt_log_probs is not None:
                fields["minf_prompt_logprobs"] = torch.tensor(
                    [list(prompt_log_probs)], dtype=torch.float32
                )
            if routing is not None and routing_shape is not None:
                fields["minf_routing_indices"] = routing.unsqueeze(0)
                fields["minf_routing_shape"] = torch.tensor(
                    [routing_shape], dtype=torch.int64
                )
            self._store.put(
                uid,
                fields,
                tags={
                    "payload_kind": "minf_request",
                    "request_uid": uid,
                    "weight_version": weight_version,
                },
            )


class TQMInfPayloadSource:
    """Fetch MInf request payloads from TQ by response UID."""

    def __init__(self, store: TQStagingStore) -> None:
        self._store = store

    def fetch(
        self, request_uids: list[str], *, include_routing_indices: bool = False
    ) -> list[FetchedMInfPayload]:
        if not request_uids:
            return []
        if len(set(request_uids)) != len(request_uids):
            raise KeyError("MInf payload request contains duplicate UIDs")
        select_fields = list(MINF_PAYLOAD_FIELDS)
        if include_routing_indices:
            select_fields.extend(["minf_routing_indices", "minf_routing_shape"])
        try:
            rows = self._store.get(request_uids, select_fields=select_fields)
        except Exception as error:  # noqa: BLE001 — normalize storage misses
            raise KeyError(
                f"MInf payload rows for {len(request_uids)} UIDs could not be fetched: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(request_uids):
            raise KeyError(
                f"MInf payload rows missing: requested {len(request_uids)}, got {n_rows}"
            )
        return [
            _row_to_minf_payload(uid, _select_row(rows, index))
            for index, uid in enumerate(request_uids)
        ]


class TQTokenSink:
    """Gym ``StagingSink`` over ``DataPlaneClient.put_samples``.

    ``stage`` is synchronous and returns only after TQ acknowledged the
    write, so the capture layer's fail-closed ordering (bytes durable before
    the model call is acked) holds by construction. Failures are reported in
    the ``StageResult`` — the caller decides whether the rollout poisons or
    aborts (``token_capture.on_capture_failure``).
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)

    def stage(self, record: StagedCallRecord) -> StageResult:
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import StageResult

        key = record.staging_key
        try:
            field_dict = {
                "token_ids_delta": torch.tensor(
                    [record.token_ids_delta], dtype=torch.int64
                ),
                "token_mask_delta": torch.tensor(
                    [record.token_mask_delta], dtype=torch.float32
                ),
                "generation_logprobs_delta": torch.tensor(
                    [record.generation_log_probs_delta], dtype=torch.float32
                ),
                "schema_version": torch.tensor(
                    [record.schema_version], dtype=torch.int64
                ),
                "digest_version": torch.tensor(
                    [record.digest_version], dtype=torch.int64
                ),
                "extras_digest_version": torch.tensor(
                    [record.extras_digest_version], dtype=torch.int64
                ),
                "rollout_id_utf8": _bytes_tensor(record.rollout_id.encode("utf-8")),
                "model_call_id_utf8": _bytes_tensor(
                    record.model_call_id.encode("utf-8")
                ),
                "parent_call_id_utf8": _bytes_tensor(
                    (record.parent_call_id or "\0").encode("utf-8")
                ),
                "parent_call_id_present": torch.tensor(
                    [record.parent_call_id is not None], dtype=torch.bool
                ),
                "capture_mode": torch.tensor(
                    [_MODE_TO_CODE[record.mode]], dtype=torch.int64
                ),
                "prev_len": torch.tensor([record.prev_len], dtype=torch.int64),
                "delta_len": torch.tensor([record.delta_len], dtype=torch.int64),
                "cum_len": torch.tensor([record.cum_len], dtype=torch.int64),
                "weight_version": torch.tensor(
                    [record.weight_version], dtype=torch.int64
                ),
                "digest_bytes": _bytes_tensor(bytes.fromhex(record.digest)),
                "extras_digest_bytes": _bytes_tensor(
                    bytes.fromhex(record.extras_digest)
                ),
            }
            chain_hash, chain_hash_present = _optional_digest_fields(record.chain_hash)
            cumulative_hash, cumulative_hash_present = _optional_digest_fields(
                record.cumulative_hash
            )
            field_dict.update(
                {
                    "chain_hash_bytes": chain_hash,
                    "chain_hash_present": chain_hash_present,
                    "cumulative_hash_bytes": cumulative_hash,
                    "cumulative_hash_present": cumulative_hash_present,
                }
            )
            extras_metadata = dict(record.extras) if record.extras is not None else None
            routed = (
                extras_metadata.pop("routed_experts", None)
                if extras_metadata is not None
                else None
            )
            field_dict[ROUTED_EXTRAS_METADATA_FIELD] = _bytes_tensor(
                json.dumps(
                    extras_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            routed_len = 0
            routed_encoding = _ROUTE_ENCODING_NONE
            if routed is not None:
                delta_len = len(record.token_ids_delta)
                if isinstance(routed, str):
                    from nemo_rl.utils.routed_experts_codec import (
                        decode_routed_experts,
                    )

                    dtype_name = routed.split(":", 3)[1]
                    dtype = {
                        "int8": torch.int8,
                        "int16": torch.int16,
                        "int32": torch.int32,
                    }.get(dtype_name)
                    if dtype is None:
                        raise ValueError(
                            f"unsupported routed_experts dtype {dtype_name!r}"
                        )
                    experts = decode_routed_experts(routed, dtype)
                    routed_encoding = _ROUTE_ENCODING_ENVELOPE
                else:
                    experts = torch.tensor(routed, dtype=torch.int16)
                    routed_encoding = _ROUTE_ENCODING_LIST
                if experts.dim() != 3 or experts.shape[0] != delta_len:
                    raise ValueError(
                        "routed_experts must already be delta-aligned: "
                        f"got shape {tuple(experts.shape)} for delta_len={delta_len}"
                    )
                field_dict[ROUTED_EXPERTS_FIELD] = experts.unsqueeze(0)
                routed_len = int(experts.shape[0])
            field_dict[ROUTED_EXPERTS_ENCODING_FIELD] = torch.tensor(
                [routed_encoding], dtype=torch.int64
            )
            field_dict[ROUTED_LEN_FIELD] = torch.tensor([routed_len], dtype=torch.int64)
            tags = [
                {
                    "rollout_id": record.rollout_id,
                    "model_call_id": record.model_call_id,
                    "parent_call_id": record.parent_call_id,
                    "prev_len": record.prev_len,
                    "delta_len": record.delta_len,
                    "cum_len": record.cum_len,
                    "weight_version": record.weight_version,
                    "digest": record.digest,
                    "schema_version": record.schema_version,
                }
            ]
            self._store.put(key, field_dict, tags=tags[0])
        except Exception as error:  # noqa: BLE001 — any failure must poison, not crash serving
            # The reason string is dropped downstream (_failed_coords carries
            # only the disposition) — this log line is the only place the
            # actual stage failure is visible.
            logging.getLogger(__name__).warning(
                "TQTokenSink.stage failed for %s: %s: %s",
                key,
                type(error).__name__,
                error,
            )
            return StageResult(
                ok=False, staging_key=key, error=f"{type(error).__name__}: {error}"
            )
        return StageResult(ok=True, staging_key=key)

    def clear(self, staging_keys: list[str]) -> None:
        """Drop staged rows (finalizer / eviction cleanup)."""
        self._store.clear(staging_keys)


class TQTokenSource:
    """Gym ``StagingSource`` over ``DataPlaneClient.get_samples``.

    All requested rows are fetched in a single batched ``get_samples`` call
    (TQ returns jagged delta columns as nested tensors; ``_from_wire``
    preserves the raggedness), in the order requested. A missing or
    unreadable row raises ``KeyError`` per the protocol — the finalizer maps
    that to a placeholder, never a silent skip. TQ's field-readiness check
    is all-or-nothing across a batch, so the extras fallback is batch-level:
    extras-free runs land in the base schema exactly like the old per-key
    probe, but a batch with *mixed* extras presence degrades every row to
    the base schema (worker feature-gating makes presence uniform per run).
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)
        self._staging_partition = staging_partition

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]:
        """Fetch Gym snapshots, retaining legacy optional route materialization."""
        if not staging_keys:
            return []
        try:
            # Extras are optional per row (feature-gated at the worker);
            # try the extended selection first, fall back to the base
            # schema so extras-free rows keep fetching.
            try:
                rows = self._store.get(
                    staging_keys,
                    select_fields=STAGING_FIELDS + [ROUTED_EXPERTS_FIELD],
                )
            except Exception:  # noqa: BLE001 — field-not-present probe
                rows = self._store.get(staging_keys, select_fields=STAGING_FIELDS)
        except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
            raise KeyError(
                f"staged rows for {len(staging_keys)} keys could not be "
                f"fetched from {self._staging_partition!r}: {error}"
            ) from error
        # TQ's kv path only errors when *zero* keys resolve; a partial miss
        # returns fewer rows with no error. Guard explicitly so a lost row
        # rejects the rollout as missing_staging_row instead of surfacing
        # later as a misleading digest mismatch from misaligned zipping.
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"staged rows missing: requested {len(staging_keys)} keys "
                f"from {self._staging_partition!r}, got {n_rows} rows"
            )
        # Row order mirrors the requested key order; the finalizer's digest
        # recomputation is the byte-exact backstop if that ever breaks.
        snapshots: list[StagedCallSnapshot] = []
        for index, key in enumerate(staging_keys):
            snapshot = _row_to_snapshot(_select_row(rows, index), include_routes=True)
            if snapshot.staging_key != key:
                raise KeyError(
                    f"staged row identity mismatch: requested {key!r}, got {snapshot.staging_key!r}"
                )
            snapshots.append(snapshot)
        return snapshots

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        """Bulk-fetch ordered delta chain and concatenate token_ids_delta into a prefix."""
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("prefix fetch: staging_keys contains duplicates")
        try:
            rows = self._store.get(
                staging_keys,
                select_fields=["token_ids_delta"],
            )
        except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
            raise KeyError(
                f"prefix fetch: staged rows for {len(staging_keys)} keys could "
                f"not be fetched from {self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if rows.batch_size else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"prefix fetch incomplete: requested {len(staging_keys)} keys, got {n_rows}"
            )
        result: list[int] = []
        for index in range(n_rows):
            row = _select_row(rows, index)
            delta = row["token_ids_delta"].squeeze(0).tolist()
            result.extend(int(t) for t in delta)
        return result

    def fetch_for_finalization(
        self, staging_keys: list[str]
    ) -> list[FetchedStagedCall]:
        """Fetch only digest-covered columns plus required route length metadata."""
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("finalization staging request contains duplicate keys")
        try:
            rows = self._store.get(staging_keys, select_fields=STAGING_FIELDS)
        except Exception as error:  # noqa: BLE001 — protocol maps misses to KeyError
            raise KeyError(
                f"staged rows for {len(staging_keys)} keys could not be "
                f"fetched from {self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"staged rows missing: requested {len(staging_keys)} keys "
                f"from {self._staging_partition!r}, got {n_rows} rows"
            )
        fetched: list[FetchedStagedCall] = []
        for index, key in enumerate(staging_keys):
            row = _select_row(rows, index)
            snapshot = _row_to_snapshot(row, include_routes=False)
            if snapshot.staging_key != key:
                raise KeyError(
                    f"staged row identity mismatch: requested {key!r}, got {snapshot.staging_key!r}"
                )
            fetched.append(
                FetchedStagedCall(
                    staging_key=key,
                    snapshot=snapshot,
                    routed_len=_row_scalar_int(row, ROUTED_LEN_FIELD),
                )
            )
        return fetched


def _select_row(rows: TensorDict, index: int) -> dict[str, torch.Tensor]:
    """Slice one row out of a batched fetch, restoring single-row shapes.

    ``_row_to_snapshot`` predates batching and expects each field with a
    leading batch dim of 1 (the shape a single-key ``get_samples`` returns),
    so re-add it after indexing. Indexing a nested tensor yields that row's
    dense component, which is exactly the jagged-row payload.
    """
    row: dict[str, torch.Tensor] = {}
    for field in rows.keys():
        row[str(field)] = rows.get(field)[index].unsqueeze(0)
    return row


def _row_to_minf_payload(
    request_uid: str, row: dict[str, torch.Tensor]
) -> FetchedMInfPayload:
    """Decode one raw MInf row while preserving float32 wire values."""

    def _values(name: str) -> torch.Tensor:
        value = row[name]
        tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
        return tensor.reshape(-1)

    routing_indices = None
    if "minf_routing_indices" in row or "minf_routing_shape" in row:
        if not {"minf_routing_indices", "minf_routing_shape"}.issubset(row):
            raise ValueError(
                f"MInf request {request_uid!r} has an incomplete routing payload"
            )
        shape = tuple(int(value) for value in _values("minf_routing_shape").tolist())
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError(
                f"MInf request {request_uid!r} has invalid routing shape {shape}"
            )
        flat_routing = _values("minf_routing_indices").to(dtype=torch.int32)
        expected_elements = shape[0] * shape[1] * shape[2]
        if flat_routing.numel() != expected_elements:
            raise ValueError(
                f"MInf request {request_uid!r} has {flat_routing.numel()} routing "
                f"values for shape {shape} ({expected_elements} required)"
            )
        routing_indices = flat_routing.reshape(shape)

    return FetchedMInfPayload(
        request_uid=request_uid,
        prompt_token_ids=[
            int(value) for value in _values("minf_prompt_token_ids").tolist()
        ],
        generated_token_ids=[
            int(value) for value in _values("minf_generated_token_ids").tolist()
        ],
        generated_log_probs=[
            float(value) for value in _values("minf_generated_logprobs").tolist()
        ],
        # Prompt logprobs remain durable when MInf supplies them, but Gym's
        # canonical delta builder intentionally synthesizes zeroes for prompt
        # carry positions.
        prompt_log_probs=None,
        routing_indices=routing_indices,
        weight_version=_row_scalar_int(row, "minf_weight_version"),
    )


def _row_to_snapshot(row: Any, *, include_routes: bool) -> StagedCallSnapshot:
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.records import StagedCallSnapshot

    def _leaf(name: str) -> torch.Tensor:
        value = row[name]
        tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
        return tensor.reshape(-1)

    def _text(name: str) -> str:
        return bytes(int(value) for value in _leaf(name).tolist()).decode("utf-8")

    def _digest(name: str) -> str:
        value = bytes(int(item) for item in _leaf(name).tolist())
        if len(value) != 32:
            raise ValueError(f"{name} must contain exactly 32 bytes")
        return value.hex()

    def _optional_digest(name: str, present_name: str) -> str | None:
        return _digest(name) if bool(_leaf(present_name)[0].item()) else None

    parent_call_id = (
        _text("parent_call_id_utf8")
        if bool(_leaf("parent_call_id_present")[0].item())
        else None
    )
    mode_code = int(_leaf("capture_mode")[0].item())
    try:
        mode = _CODE_TO_MODE[mode_code]
    except KeyError as error:
        raise ValueError(f"unknown capture_mode code {mode_code}") from error
    extras = json.loads(_text(ROUTED_EXTRAS_METADATA_FIELD))
    routed_encoding = int(_leaf(ROUTED_EXPERTS_ENCODING_FIELD)[0].item())
    if routed_encoding != _ROUTE_ENCODING_NONE and include_routes:
        try:
            routed = row[ROUTED_EXPERTS_FIELD]
        except KeyError as error:
            raise KeyError(
                "staged row metadata names routed_experts but its field is absent"
            ) from error
        experts = routed[0] if routed.dim() > 3 or routed.shape[0] == 1 else routed
        if extras is None:
            extras = {}
        if not isinstance(extras, dict):
            raise TypeError("staged extras metadata must decode to an object or null")
        if routed_encoding == _ROUTE_ENCODING_ENVELOPE:
            from nemo_rl.utils.routed_experts_codec import encode_routed_experts

            extras[ROUTED_EXPERTS_FIELD] = encode_routed_experts(experts)
        elif routed_encoding == _ROUTE_ENCODING_LIST:
            extras[ROUTED_EXPERTS_FIELD] = experts.tolist()
        else:
            raise ValueError(f"unknown routed_experts_encoding {routed_encoding}")
    elif routed_encoding not in (
        _ROUTE_ENCODING_NONE,
        _ROUTE_ENCODING_ENVELOPE,
        _ROUTE_ENCODING_LIST,
    ):
        raise ValueError(f"unknown routed_experts_encoding {routed_encoding}")

    values = dict(
        schema_version=int(_leaf("schema_version")[0].item()),
        digest_version=int(_leaf("digest_version")[0].item()),
        extras_digest_version=int(_leaf("extras_digest_version")[0].item()),
        rollout_id=_text("rollout_id_utf8"),
        model_call_id=_text("model_call_id_utf8"),
        parent_call_id=parent_call_id,
        mode=mode,
        prev_len=int(_leaf("prev_len")[0].item()),
        delta_len=int(_leaf("delta_len")[0].item()),
        cum_len=int(_leaf("cum_len")[0].item()),
        weight_version=int(_leaf("weight_version")[0].item()),
        digest=_digest("digest_bytes"),
        token_ids_delta=[int(t) for t in _leaf("token_ids_delta").tolist()],
        token_mask_delta=[float(m) for m in _leaf("token_mask_delta").tolist()],
        generation_log_probs_delta=[
            float(p) for p in _leaf("generation_logprobs_delta").tolist()
        ],
        extras=extras,
        extras_digest=_digest("extras_digest_bytes"),
        chain_hash=_optional_digest("chain_hash_bytes", "chain_hash_present"),
        cumulative_hash=_optional_digest(
            "cumulative_hash_bytes", "cumulative_hash_present"
        ),
    )
    if include_routes or routed_encoding == _ROUTE_ENCODING_NONE:
        return StagedCallSnapshot(**values)
    # Metadata-only deferred finalization deliberately leaves the heavy route
    # fragment in TQ. The finalizer verifies its digest binding before
    # publishing a route assembly plan.
    return StagedCallSnapshot.model_construct(**values)


def _row_scalar_int(row: Any, field_name: str) -> int:
    """Read one required scalar from a single-row TQ result."""
    value = row[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"staging field {field_name!r} must be a tensor, got {type(value).__name__}"
        )
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise TypeError(
            f"staging field {field_name!r} must use an integer dtype, got {value.dtype}"
        )
    tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
    flattened = tensor.reshape(-1)
    if flattened.numel() != 1:
        raise ValueError(
            f"staging field {field_name!r} must contain one scalar, got "
            f"shape {tuple(value.shape)}"
        )
    return int(flattened[0].item())
