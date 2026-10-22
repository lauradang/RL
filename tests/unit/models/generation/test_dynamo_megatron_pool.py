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

"""Megatron prefill/decode engines behind Dynamo: the worker pool."""

import copy
import sys
import types
from dataclasses import replace

import pytest
import ray

import nemo_rl.models.generation.dynamo.megatron_worker_pool as pool_module
from nemo_rl.models.generation.dynamo.config import DynamoConfig, DynamoWorkerLayout
from nemo_rl.models.generation.dynamo.megatron_worker_pool import (
    MegatronDynamoWorkerPool,
)
from nemo_rl.models.generation.dynamo.protocols import DynamoWorkerPool
from tests.unit.models.generation.test_dynamo_arguments import (
    MEGATRON_POLICY,
    _megatron_config,
)

_ROLES = ["prefill", "decode", "decode"]
_NODES = ["10.0.0.1", "10.0.0.1", "10.0.0.2"]
_WIDTHS = {"prefill": 2, "decode": 2}
_NOSET = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"


def _widths(width: int) -> dict[str, int]:
    """One width for both roles."""
    return {"prefill": width, "decode": width}


def _workers(roles, widths) -> list[DynamoWorkerLayout]:
    """Engines of the shared layout, one per role, each as wide as its role."""
    return [
        DynamoWorkerLayout(role=r, width=widths.get(r, 2), overrides={}) for r in roles
    ]


class _Ref:
    """Stands in for a Ray object ref; the fake ``ray.get`` unwraps it."""

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error


class _Remote:
    """A remote method: ``.remote(...)`` runs the function and wraps the result.

    Like Ray, a failure surfaces at ``ray.get``, not at ``.remote()``.
    """

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        try:
            return _Ref(self._fn(*args, **kwargs))
        except Exception as error:
            return _Ref(error=error)


class _FakeRay:
    exceptions = ray.exceptions

    def __init__(self, log):
        self.log = log

    def get(self, item, timeout=None):
        if isinstance(item, list):
            return [self.get(entry) for entry in item]
        if not isinstance(item, _Ref):
            return item
        if item.error is not None:
            raise item.error
        return item.value

    def kill(self, handle, no_restart=False):
        self.log.append(f"kill {handle.instance_id}")


class _FakePlacementGroup:
    def __init__(self, bundle_nodes):
        self.bundle_nodes = bundle_nodes
        self.bundle_count = len(bundle_nodes)
        self.node_ip = bundle_nodes[0]


class _FakeCluster:
    def __init__(self, fleet, kwargs):
        self.kwargs = kwargs
        self.log = fleet.log
        self.name = kwargs["name"].rsplit("_", 1)[-1]
        index = len(fleet.clusters)
        (width,) = kwargs["bundle_ct_per_node_list"]
        bundle_nodes = fleet.bundle_nodes.get(index, [fleet.nodes[index]] * width)
        self.placement_group = _FakePlacementGroup(bundle_nodes)
        self.placement_policy_config = None
        self.log.append(f"cluster {self.name} up")

    def get_placement_groups(self):
        return [self.placement_group]

    def shutdown(self):
        self.log.append(f"cluster {self.name} down")
        return True


class _FakeEngine:
    def __init__(self, fleet, kwargs):
        self.kwargs = kwargs
        self.log = fleet.log
        self.name = kwargs["name_prefix"].removeprefix("dynamo_")
        self.attached = None
        self.alive = True
        self.coordinator = None
        self.fail_shutdown = False
        self.unclean_shutdown = False
        self.log.append(f"engine {self.name} up")

    def attach_dynamo_adapter(self, *, parent_event_address, role):
        self.attached = (parent_event_address, role)
        self.log.append(f"attach {self.name} {role}")

    def shard_liveness_ref(self, shard_idx):
        return _Ref(self.alive)

    def dynamo_coordinator_address(self):
        return self.coordinator

    def shutdown(self):
        if self.fail_shutdown:
            raise RuntimeError(f"engine {self.name} shutdown failed")
        self.log.append(f"engine {self.name} down")
        return not self.unclean_shutdown


class _FakeSidecar:
    def __init__(self, fleet, index, options, kwargs):
        self.options = options
        self.kwargs = kwargs
        self.log = fleet.log
        self.instance_id = kwargs["instance_id"]
        self.pid = 100 + index
        self.alive = True
        self.fail_calls = False
        self.reject_attestation = False
        node_ip = options["scheduling_strategy"].placement_group.node_ip
        self.event_port = 7000 + 2 * index
        self.system_port = self.event_port + 1
        self.owner = f"owner-{self.instance_id}"
        self._meta = {
            "instance_id": self.instance_id,
            "role": kwargs["role"],
            "node_ip": node_ip,
            "parent_event_address": f"tcp://{node_ip}:{self.event_port}",
            "event_port": self.event_port,
            "system_port": self.system_port,
            "handoff_owner": self.owner,
        }
        self.metadata = _Remote(self._metadata)
        self.is_alive = _Remote(lambda: self.alive)
        self.shutdown = _Remote(self._shutdown)
        self.confirm_terminated = _Remote(self._confirm_terminated)
        self.log.append(f"sidecar {self.instance_id} up")

    def _metadata(self):
        if self.fail_calls:
            # Ray reports a constructor failure at the actor's first call.
            raise ray.exceptions.ActorDiedError()
        return {**self._meta, "process_pid": self.pid}

    def _shutdown(self):
        if self.fail_calls:
            raise RuntimeError(f"sidecar {self.instance_id} is gone")
        self.log.append(f"sidecar {self.instance_id} down")
        return True

    def _confirm_terminated(self, addresses, owner, timeout_s):
        if self.fail_calls:
            raise RuntimeError(f"sidecar {self.instance_id} is gone")
        self.log.append(f"confirm {self.instance_id} {owner} {addresses} {timeout_s}")
        return not self.reject_attestation


class _FakeAdapterActor:
    """``DynamoMegatronAdapterActor`` stand-in: ``options(...).remote(...)``."""

    def __init__(self, fleet):
        self._fleet = fleet

    def options(self, **options):
        fleet = self._fleet

        class _Bound:
            @staticmethod
            def remote(**kwargs):
                index = len(fleet.sidecars)
                sidecar = _FakeSidecar(fleet, index, options, kwargs)
                sidecar.fail_calls = fleet.fail_sidecar_at == index
                fleet.sidecars.append(sidecar)
                return sidecar

        return _Bound()


class _NodeIpTask:
    def options(self, *, scheduling_strategy):
        placement_group = scheduling_strategy.placement_group
        index = scheduling_strategy.placement_group_bundle_index
        return _Remote(lambda: placement_group.bundle_nodes[index])


class _Fleet:
    """Everything the pool touches, patched into the pool module."""

    def __init__(self, monkeypatch, nodes):
        self.nodes = nodes
        self.bundle_nodes: dict[int, list[str]] = {}
        self.log: list[str] = []
        self.clusters: list[_FakeCluster] = []
        self.engines: list[_FakeEngine] = []
        self.sidecars: list[_FakeSidecar] = []
        self.runtime_env_fqns: list[str] = []
        self.fail_sidecar_at = None
        self.ray = _FakeRay(self.log)

        def make_cluster(**kwargs):
            cluster = _FakeCluster(self, kwargs)
            self.clusters.append(cluster)
            return cluster

        def make_engine(**kwargs):
            engine = _FakeEngine(self, kwargs)
            self.engines.append(engine)
            return engine

        def init_cluster_placement_groups(cluster, config):
            cluster.placement_policy_config = config
            self.log.append(f"placement {cluster.name}")

        make_engine.init_cluster_placement_groups = init_cluster_placement_groups

        def make_runtime_env(fqn, *, extra_env_vars=None):
            self.runtime_env_fqns.append(fqn)
            return {
                "py_executable": "/venv/bin/python",
                "env_vars": {"VIRTUAL_ENV": "/venv", **(extra_env_vars or {})},
            }

        generation_module = types.ModuleType(
            "nemo_rl.models.generation.megatron.megatron_generation"
        )
        generation_module.MegatronGeneration = make_engine
        monkeypatch.setitem(sys.modules, generation_module.__name__, generation_module)
        monkeypatch.setattr(pool_module, "RayVirtualCluster", make_cluster)
        monkeypatch.setattr(
            pool_module, "DynamoMegatronAdapterActor", _FakeAdapterActor(self)
        )
        monkeypatch.setattr(pool_module, "cluster_node_ip", _NodeIpTask())
        monkeypatch.setattr(pool_module, "make_actor_runtime_env", make_runtime_env)
        monkeypatch.setattr(pool_module, "ray", self.ray)

    def pool(self, generation=None, **overrides) -> MegatronDynamoWorkerPool:
        generation = generation or _megatron_config()
        kwargs = dict(
            policy_config={**MEGATRON_POLICY, "generation": generation},
            tokenizer="tok",
            config=DynamoConfig.model_validate(generation),
            namespace="ns",
            manager_env={"ETCD_ENDPOINTS": "http://etcd:2379"},
            num_gpus_per_node=8,
            port_range_low=None,
            port_range_high=None,
            node_resource_constraints=None,
        )
        kwargs.update(overrides)
        return MegatronDynamoWorkerPool(**kwargs)


@pytest.fixture
def fleet(monkeypatch):
    return _Fleet(monkeypatch, _NODES)


def test_start_builds_one_cluster_engine_and_sidecar_per_role(fleet):
    pool = fleet.pool()
    assert isinstance(pool, DynamoWorkerPool)
    assert pool.expected_components() == {} and not pool.is_alive()

    records = pool.start(_workers(_ROLES, _WIDTHS))

    ids = ["prefill-0", "decode-0", "decode-1"]
    assert [r.instance_id for r in records] == ids
    assert [r.role for r in records] == _ROLES
    assert [r.node_ip for r in records] == _NODES
    assert all(r.world_size == 2 for r in records)
    assert [r.control for r in records] == fleet.engines
    assert pool.records == records and pool.records is not pool.records
    assert pool.expected_components() == {"prefill": 1, "backend": 2}
    assert pool.is_alive()
    # Each engine is created, then its sidecar, then attached, before the next.
    expected_log = []
    for name, role in zip(ids, _ROLES):
        expected_log += [
            f"cluster {name} up",
            f"placement {name}",
            f"engine {name} up",
            f"sidecar {name} up",
            f"attach {name} {role}",
        ]
    assert fleet.log == expected_log

    cluster = fleet.clusters[0]
    assert cluster.kwargs == {
        "name": "dynamo_ns_prefill-0",
        "bundle_ct_per_node_list": [2],
        "use_gpus": True,
        "num_gpus_per_node": 8,
        "max_colocated_worker_groups": 1,
        "port_range_low": None,
        "port_range_high": None,
        "node_resource_constraints": None,
    }
    # No role overrides: each engine builds from a copy of the shared policy.
    assert cluster.placement_policy_config == pool._policy_config
    engine = fleet.engines[0]
    assert engine.kwargs["config"] == pool._policy_config
    assert engine.kwargs["cluster"] is cluster
    assert engine.kwargs["skip_weight_load"] is True
    assert engine.kwargs["tokenizer"] == "tok"
    for engine, sidecar, role in zip(fleet.engines, fleet.sidecars, _ROLES):
        assert engine.attached == (sidecar._meta["parent_event_address"], role)
        assert sidecar.kwargs["role"] == role
    assert [s.kwargs["component"] for s in fleet.sidecars] == [
        "prefill",
        "backend",
        "backend",
    ]
    # The second sidecar on 10.0.0.1 avoids the first one's ports; the engine on
    # 10.0.0.2 starts fresh.
    first = fleet.sidecars[0]
    assert [s.kwargs["excluded_ports"] for s in fleet.sidecars] == [
        [],
        [first.event_port, first.system_port],
        [],
    ]
    # Scheduled like the vLLM sidecar: engine bundle 0, no GPU, the dynamo venv.
    options = first.options
    assert options["num_gpus"] == 0
    assert options["name"] == "ns-dynamo-adapter-prefill-0"
    strategy = options["scheduling_strategy"]
    assert strategy.placement_group is fleet.clusters[0].placement_group
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True
    assert options["runtime_env"]["env_vars"][_NOSET] == "1"
    assert fleet.runtime_env_fqns == [
        "nemo_rl.models.generation.dynamo.megatron_worker_pool."
        "DynamoMegatronAdapterActor"
    ]
    kwargs = first.kwargs
    assert kwargs["model_name"] == "Qwen/Qwen3-0.6B"
    assert kwargs["namespace"] == "ns"
    assert kwargs["manager_env"] == {"ETCD_ENDPOINTS": "http://etcd:2379"}
    assert kwargs["engine_start_timeout_s"] == 600
    assert kwargs["endpoint_types"] == ["chat", "completions"]
    assert kwargs["extra_cli_args"] == []
    assert kwargs["engine_argv"] == [
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        "1",
    ]
    assert kwargs["megatron_root"] == str(pool_module.default_megatron_root())


def _layout(engine_config: dict) -> tuple[int, int]:
    """(TP, PP) the Megatron workers of a policy config are built with."""
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    merged = merged_inference_megatron_cfg(engine_config)
    return merged["tensor_model_parallel_size"], merged["pipeline_model_parallel_size"]


@pytest.mark.parametrize(
    ("prefill", "decode", "layouts"),
    [
        # 1 prefill at TP 4 next to 2 decode at the shared TP 2.
        (
            [{"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1}],
            [],
            [(4, 1), (2, 1), (2, 1)],
        ),
        # The two decode engines differ: TP 1, then TP 2 at PP 2.
        (
            [],
            [
                {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
                {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 2},
            ],
            [(2, 1), (1, 1), (2, 2)],
        ),
    ],
    ids=["prefill_tp4", "decode_mixed"],
)
def test_each_engine_builds_with_its_own_layout(fleet, prefill, decode, layouts):
    """Cluster width, record width, sidecar argv and engine config follow the entry."""
    generation = _megatron_config()
    generation["dynamo_cfg"].update(prefill_engines=prefill, decode_engines=decode)
    pool = fleet.pool(generation=generation)
    shared = {**MEGATRON_POLICY, "generation": generation}
    config = DynamoConfig.model_validate(generation)
    widths = [tp * pp for tp, pp in layouts]

    records = pool.start(config.replica_layout(shared))

    assert [r.world_size for r in records] == widths
    assert [c.kwargs["bundle_ct_per_node_list"] for c in fleet.clusters] == [
        [width] for width in widths
    ]
    assert [_layout(e.kwargs["config"]) for e in fleet.engines] == layouts
    assert [_layout(c.placement_policy_config) for c in fleet.clusters] == layouts
    assert [s.kwargs["engine_argv"] for s in fleet.sidecars] == [
        [
            "--tensor-model-parallel-size",
            str(tp),
            "--pipeline-model-parallel-size",
            str(pp),
        ]
        for tp, pp in layouts
    ]
    # The shared policy config is untouched; each engine got its own copy.
    assert pool._policy_config == shared
    assert pool_module.role_policy_config(shared, {}) == shared
    assert pool_module.role_policy_config(shared, {}) is not shared


@pytest.mark.parametrize(
    ("mcore_generation_config", "overrides", "layout"),
    [
        # A null shared block is the policy layout plus the role's overrides.
        (None, {"tensor_model_parallel_size": 4}, (4, 1)),
        # Overrides win over the shared block's layout keys.
        (
            {"pipeline_model_parallel_size": 2},
            {"tensor_model_parallel_size": 4},
            (4, 2),
        ),
    ],
    ids=["null_shared_block", "shared_block_present"],
)
def test_role_policy_config_merges_overrides_into_a_copy(
    mcore_generation_config, overrides, layout
):
    generation = {
        **_megatron_config(),
        "mcore_generation_config": mcore_generation_config,
    }
    shared = {**MEGATRON_POLICY, "generation": generation}
    before = copy.deepcopy(shared)

    merged = pool_module.role_policy_config(shared, overrides)

    assert _layout(merged) == layout
    assert shared == before
    assert merged["generation"] is not shared["generation"]


def test_engine_wider_than_a_node_is_rejected_by_role(fleet):
    with pytest.raises(
        ValueError, match=r"decode engine spans 16 GPUs.*decode_engines"
    ):
        fleet.pool().start(_workers(_ROLES, {"prefill": 2, "decode": 16}))
    assert fleet.log == []


def test_mixed_widths_tile_pinned_nodes_in_launch_order(fleet):
    """Each engine sits on one pinned node; a full node moves to the next pin."""
    pins = [{"nvl-a": 0.001}, {"nvl-b": 0.001}]
    # 4 + 2 + 2 fills node 0; 8 + 4 + 4 needs node 1 for both decode engines.
    pool = fleet.pool(node_resource_constraints=pins)
    pool.start(_workers(_ROLES, {"prefill": 4, "decode": 2}))
    assert [c.kwargs["node_resource_constraints"] for c in fleet.clusters] == [
        [pins[0]]
    ] * 3

    fleet.clusters.clear()
    fleet.pool(node_resource_constraints=pins).start(
        _workers(_ROLES, {"prefill": 8, "decode": 4})
    )
    assert [c.kwargs["node_resource_constraints"] for c in fleet.clusters] == [
        [pins[0]],
        [pins[1]],
        [pins[1]],
    ]

    short = fleet.pool(node_resource_constraints=pins[:1])
    with pytest.raises(ValueError, match=r"widths \[8, 4, 4\] GPUs tile 2 nodes"):
        short.start(_workers(_ROLES, {"prefill": 8, "decode": 4}))


def test_megatron_root_override_is_expanded(fleet, monkeypatch):
    monkeypatch.setenv("HOME", "/home/me")
    generation = _megatron_config()
    generation["dynamo_cfg"]["megatron_root"] = "~/megatron-lm"
    pool = fleet.pool(generation=generation)
    pool.start(_workers(["prefill", "decode"], _WIDTHS))
    assert fleet.sidecars[0].kwargs["megatron_root"] == "/home/me/megatron-lm"


def test_rejections(fleet):
    with pytest.raises(ValueError, match="Aggregated Megatron engines"):
        fleet.pool().start(_workers(["aggregated"], _WIDTHS))
    with pytest.raises(ValueError, match="multi-node engines"):
        fleet.pool().start(_workers(_ROLES, _widths(16)))
    assert fleet.log == []

    vllm_generation = _megatron_config()
    vllm_generation["dynamo_cfg"]["engine"] = "vllm"
    vllm_generation["vllm_cfg"] = {
        "async_engine": True,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 512,
        "precision": "bfloat16",
        "kv_cache_dtype": "auto",
        "load_format": "auto",
        "enforce_eager": False,
        "expose_http_server": False,
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 1.0,
        "env_vars": None,
    }
    with pytest.raises(ValueError, match="engine='megatron'"):
        fleet.pool(generation=vllm_generation)

    pool = fleet.pool()
    pool.start(_workers(_ROLES, _WIDTHS))
    with pytest.raises(RuntimeError, match="already started"):
        pool.start(_workers(_ROLES, _WIDTHS))


def test_start_failure_tears_down_what_exists(fleet, caplog):
    # The third sidecar's constructor fails; Ray reports it at the first call.
    fleet.fail_sidecar_at = 2
    pool = fleet.pool()
    with caplog.at_level("ERROR"), pytest.raises(ray.exceptions.RayActorError):
        pool.start(_workers(_ROLES, _WIDTHS))
    assert pool.records == [] and pool.expected_components() == {}
    down = [line for line in fleet.log if line.endswith(" down")]
    assert set(down) == {
        "sidecar prefill-0 down",
        "sidecar decode-0 down",
        "engine prefill-0 down",
        "engine decode-0 down",
        "engine decode-1 down",
        "cluster prefill-0 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
    }
    # The sidecar that never answered is still killed.
    assert "kill decode-1" in fleet.log
    # Engines without a record are not mistaken for a broken pool.
    assert "prefill coordinator addresses" not in caplog.text


def test_engine_split_across_nodes_is_rejected(fleet):
    fleet.bundle_nodes[1] = ["10.0.0.1", "10.0.0.2"]
    pool = fleet.pool()
    with pytest.raises(RuntimeError, match=r"decode-0 spans multiple nodes"):
        pool.start(_workers(_ROLES, _WIDTHS))
    # Rejected before the engine's workers or sidecar exist.
    assert [engine.name for engine in fleet.engines] == ["prefill-0"]
    assert [sidecar.instance_id for sidecar in fleet.sidecars] == ["prefill-0"]
    assert "cluster decode-0 down" in fleet.log
    assert pool.records == []


def test_engine_clusters_inherit_port_range_and_node_pins(fleet):
    pins = [{"nvl-a": 0.001}, {"nvl-b": 0.001}]
    pool = fleet.pool(
        port_range_low=2100, port_range_high=2200, node_resource_constraints=pins
    )
    # Two 4-GPU engines per 8-GPU node: the first two take node 0's pin.
    pool.start(_workers(_ROLES, _widths(4)))
    assert [c.kwargs["node_resource_constraints"] for c in fleet.clusters] == [
        [pins[0]],
        [pins[0]],
        [pins[1]],
    ]
    assert {
        (c.kwargs["port_range_low"], c.kwargs["port_range_high"])
        for c in fleet.clusters
    } == {(2100, 2200)}

    short = fleet.pool(
        port_range_low=None, port_range_high=None, node_resource_constraints=pins[:1]
    )
    fleet.log.clear()
    with pytest.raises(ValueError, match="more than the 1 pinned inference nodes"):
        short.start(_workers(_ROLES, _widths(4)))
    assert fleet.log == []


def test_shutdown_stops_decode_attests_owners_then_prefill(fleet):
    pool = fleet.pool()
    pool.start(_workers(_ROLES, _WIDTHS))
    fleet.engines[0].coordinator = "tcp://10.0.0.1:5555"
    assert pool.prefill_coordinator_addresses() == ["tcp://10.0.0.1:5555"]
    fleet.log.clear()

    pool.shutdown()

    assert fleet.log == [
        "sidecar decode-0 down",
        "kill decode-0",
        "engine decode-0 down",
        "sidecar decode-1 down",
        "kill decode-1",
        "engine decode-1 down",
        "confirm prefill-0 owner-decode-0 ['tcp://10.0.0.1:5555'] 600.0",
        "confirm prefill-0 owner-decode-1 ['tcp://10.0.0.1:5555'] 600.0",
        "sidecar prefill-0 down",
        "kill prefill-0",
        "engine prefill-0 down",
        "cluster prefill-0 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
    ]
    assert pool.records == [] and not pool.is_alive()
    fleet.log.clear()
    pool.shutdown()
    assert fleet.log == []


def test_shutdown_skips_attestation_without_prefill_coordinator(fleet):
    pool = fleet.pool()
    pool.start(_workers(_ROLES, _WIDTHS))
    fleet.log.clear()
    pool.shutdown()
    assert not any(line.startswith("confirm") for line in fleet.log)


def test_shutdown_continues_past_sidecar_and_engine_failures(fleet):
    pool = fleet.pool()
    pool.start(_workers(_ROLES, _WIDTHS))
    fleet.engines[0].coordinator = "tcp://10.0.0.1:5555"
    fleet.sidecars[0].fail_calls = True  # the attester: confirm and shutdown raise
    fleet.sidecars[1].fail_calls = True
    fleet.engines[1].fail_shutdown = True
    fleet.log.clear()

    pool.shutdown()

    assert fleet.log == [
        "kill decode-0",
        "sidecar decode-1 down",
        "kill decode-1",
        "engine decode-1 down",
        "kill prefill-0",
        "engine prefill-0 down",
        "cluster prefill-0 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
    ]
    assert pool.records == []


def test_shutdown_attests_only_cleanly_stopped_decode_engines(fleet, caplog):
    pool = fleet.pool()
    pool.start(_workers(_ROLES, _WIDTHS))
    fleet.engines[0].coordinator = "tcp://10.0.0.1:5555"
    fleet.engines[1].unclean_shutdown = True
    fleet.sidecars[0].reject_attestation = True
    fleet.log.clear()

    with caplog.at_level("WARNING"):
        pool.shutdown()

    confirms = [line for line in fleet.log if line.startswith("confirm")]
    assert confirms == [
        "confirm prefill-0 owner-decode-1 ['tcp://10.0.0.1:5555'] 600.0"
    ]
    assert "Decode engine decode-0 did not stop cleanly" in caplog.text
    assert "did not accept decode owner owner-decode-1" in caplog.text


def test_validate_rejects_sidecar_restart_and_identity_change(fleet):
    pool = fleet.pool()
    records = pool.start(_workers(_ROLES, _WIDTHS))
    assert pool.validate(records) == records

    fleet.sidecars[1].pid += 1
    with pytest.raises(RuntimeError, match="adapter membership changed"):
        pool.validate(records)
    fleet.sidecars[1].pid -= 1

    moved = [records[0], replace(records[1], node_ip="10.0.0.9"), records[2]]
    with pytest.raises(RuntimeError, match="engine membership changed"):
        pool.validate(moved)

    fleet.engines[2].alive = False
    assert not pool.is_alive()
    with pytest.raises(RuntimeError, match="exited behind the frontend"):
        pool.validate(records)
    fleet.engines[2].alive = True
    fleet.sidecars[0].alive = False
    assert not pool.is_alive()
    fleet.sidecars[0].alive = True

    def dead_actor():
        raise ray.exceptions.RayActorError()

    fleet.sidecars[1].is_alive = _Remote(dead_actor)
    assert not pool.is_alive()
    with pytest.raises(RuntimeError, match="exited behind the frontend"):
        pool.validate(records)
