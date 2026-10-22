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

"""Driver-owned lifecycle for a fixed Dynamo deployment.

The runtime owns etcd, NATS, and the frontend, and either the fixed Ray-managed
``dynamo.vllm`` worker fleet or a ``DynamoWorkerPool`` (NeMo RL's own Megatron
engines) registered behind that frontend.
"""

import atexit
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from typing import Any, Protocol

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.dynamo.protocols import (
    DynamoWorkerPool,
    DynamoWorkerRecord,
)
from nemo_rl.models.generation.dynamo.venv import (
    get_dynamo_python,
    get_dynamo_service_executable,
)
from nemo_rl.models.generation.dynamo.worker_pool import FixedDynamoWorkerPool

LOGGER = logging.getLogger(__name__)


def _managed_namespace() -> str:
    raw = f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    namespace = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-_").lower()
    if not namespace:
        raise ValueError(f"Could not derive a valid Dynamo namespace from {raw!r}.")
    return namespace


class WorkerPoolFactory(Protocol):
    """Builds an worker pool once the runtime knows its namespace and env."""

    def __call__(
        self, *, namespace: str, manager_env: dict[str, str]
    ) -> DynamoWorkerPool: ...


# Endpoints each engine registers: the vLLM fleet serves generation and the
# RL control route; Megatron adapters serve generation only.
VLLM_ENGINE_ENDPOINTS: frozenset[str] = frozenset({"generate", "rl"})
POOL_ENGINE_ENDPOINTS: frozenset[str] = frozenset({"generate"})


def wait_for_frontend_registrations(
    *,
    base_url: str,
    namespace: str,
    expected_components: Mapping[str, int],
    expected_endpoints: Collection[str],
    expected_model: str,
    timeout_s: float,
    check_alive: Callable[[], None] | None = None,
) -> None:
    """Block until the frontend lists every expected engine and the model.

    ``/health`` lists discovery registrations. Every component in
    ``expected_components`` must show exactly its count of instance ids on each
    endpoint in ``expected_endpoints``, with the same ids across those endpoints.
    ``/v1/models`` must then list ``expected_model``: the frontend installs its
    OpenAI routes from a model watcher that can lag discovery, and exposing the
    URL before that races a transient 404. The frontend answers 503 until both
    prefill and decode are live and never falls back to aggregated serving, so
    under disaggregation every component must register.

    Args:
        base_url: Frontend root without the ``/v1`` suffix.
        namespace: Dynamo namespace the engines register under.
        expected_components: Component name to the number of engines in it.
        expected_endpoints: Endpoints each engine registers.
        expected_model: Served model name ``/v1/models`` must list.
        timeout_s: Total wait budget.
        check_alive: Raises when a process the caller owns has died.

    Raises:
        RuntimeError: When the fleet is not fully registered within ``timeout_s``.
    """
    root = base_url.rstrip("/")
    health_url = f"{root}/health"
    models_url = f"{root}/v1/models"
    endpoints = sorted(expected_endpoints)
    deadline = time.monotonic() + timeout_s
    last_counts: dict[str, dict[str, int]] = {
        component: {endpoint: 0 for endpoint in endpoints}
        for component in expected_components
    }
    last_models: set[str] = set()
    while time.monotonic() < deadline:
        if check_alive is not None:
            check_alive()
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
            continue
        # component -> endpoint -> instance ids
        observed: dict[str, dict[str, set[str]]] = {
            component: {endpoint: set() for endpoint in endpoints}
            for component in expected_components
        }
        for instance in payload.get("instances", []):
            if not isinstance(instance, dict):
                continue
            if instance.get("namespace") != namespace:
                continue
            component = instance.get("component")
            endpoint = instance.get("endpoint")
            instance_id = instance.get("instance_id")
            if (
                isinstance(component, str)
                and component in observed
                and endpoint in observed[component]
                and instance_id is not None
            ):
                observed[component][str(endpoint)].add(str(instance_id))
        last_counts = {
            component: {endpoint: len(ids) for endpoint, ids in by_endpoint.items()}
            for component, by_endpoint in observed.items()
        }
        complete = all(
            all(len(observed[component][endpoint]) == count for endpoint in endpoints)
            and len({frozenset(observed[component][e]) for e in endpoints}) == 1
            for component, count in expected_components.items()
        )
        if not complete:
            time.sleep(1)
            continue
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                models_payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
            continue
        last_models = {
            str(model["id"])
            for model in models_payload.get("data", [])
            if isinstance(model, dict) and model.get("id") is not None
        }
        if expected_model not in last_models:
            time.sleep(1)
            continue
        print(
            f"  [Dynamo] frontend ready: {dict(expected_components)} engines "
            f"serving {endpoints} for {expected_model!r}",
            flush=True,
        )
        return
    raise RuntimeError(
        "Dynamo frontend did not observe the fixed engine fleet within "
        f"{timeout_s}s: expected={dict(expected_components)} on {endpoints}, "
        f"last counts={last_counts}, expected_model={expected_model!r}, "
        f"last_models={sorted(last_models)!r}."
    )


class ManagedDynamoRuntime:
    """Own etcd, NATS, frontend, and the engines registered behind them.

    Engines are either the fixed Ray-managed ``dynamo.vllm`` fleet (the default)
    or a ``DynamoWorkerPool`` built by ``worker_pool_factory`` once the namespace
    and service environment exist. A pool's engines may register only later
    (Megatron engines come up at the first refit), so readiness is a separate
    step, :meth:`wait_until_ready`, that the vLLM path runs inside :meth:`start`
    and a pool's owner runs once its engines are serving.
    """

    def __init__(
        self,
        *,
        cluster: RayVirtualCluster,
        config: dict[str, Any],
        policy_config: Mapping[str, Any] | None = None,
        worker_pool_factory: WorkerPoolFactory | None = None,
    ) -> None:
        """Validate the fleet shape without starting anything.

        Args:
            cluster: Non-colocated inference cluster the engines run on.
            config: ``policy.generation`` (a ``DynamoConfig``).
            policy_config: Full ``policy`` config; Megatron engines take their
                width from it. None for vLLM engines.
            worker_pool_factory: Builds the worker pool; required for Megatron
                engines, absent for the vLLM fleet.

        Raises:
            ValueError: When Megatron engines come without a factory, or one
                engine does not fit on a node.
        """
        validated_config = DynamoConfig.model_validate(config)
        self._cluster = cluster
        self._config = validated_config.model_dump()
        self._dynamo_cfg = validated_config.dynamo_cfg
        self._worker_pool_factory = worker_pool_factory
        if validated_config.is_megatron_engine and worker_pool_factory is None:
            raise ValueError(
                "Megatron engines behind Dynamo require an worker pool factory"
            )
        # Every engine of the fleet in launch order, each with its own width.
        replica_layout = validated_config.replica_layout(policy_config)
        widest = max(replica_layout, key=lambda worker: worker.width)
        if widest.width > cluster.num_gpus_per_node:
            raise ValueError(
                "Managed Dynamo requires each TP/PP engine group to fit on one "
                f"node: a {widest.role} engine spans {widest.width} ranks, which "
                f"exceeds cluster.num_gpus_per_node={cluster.num_gpus_per_node}"
            )
        self._worker_layouts = validated_config.worker_layouts(
            cluster.world_size(), replica_layout
        )
        self._worker_roles = [worker.role for worker in self._worker_layouts]
        # Dynamo component names: aggregated and decode workers register as
        # "backend", prefill workers as "prefill".
        self._expected_components = dict(
            Counter(
                "prefill" if role == "prefill" else "backend"
                for role in self._worker_roles
            )
        )
        self._namespace = _managed_namespace()
        self._host = ""
        self._etcd_port = 0
        self._etcd_peer_port = 0
        self._nats_port = 0
        self._frontend_port = 0
        self._manager_env: dict[str, str] = {}
        self._started = False
        self._atexit_registered = False
        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._frontend_process: subprocess.Popen | None = None
        self._etcd_data_dir: str | None = None
        self._nats_data_dir: str | None = None
        self._pool: FixedDynamoWorkerPool | None = None
        self._worker_pool: DynamoWorkerPool | None = None
        self._fleet_ready = False

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def frontend_base_url(self) -> str:
        """Frontend root, without the ``/v1`` suffix."""
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._frontend_port}"

    @property
    def frontend_url(self) -> str:
        return f"{self.frontend_base_url}/v1"

    @property
    def fleet_ready(self) -> bool:
        """Whether the frontend has listed every engine and the model."""
        return self._fleet_ready

    def expected_registrations(self) -> tuple[dict[str, int], frozenset[str]]:
        """What the frontend must list before the fleet counts as ready.

        Returns:
            The component counts and the endpoints each engine registers; the
            arguments :func:`wait_for_frontend_registrations` needs, so a copy
            of the owner in another process can run the same wait.
        """
        if self._worker_pool is not None:
            return self._worker_pool.expected_components(), POOL_ENGINE_ENDPOINTS
        return dict(self._expected_components), VLLM_ENGINE_ENDPOINTS

    def start(self) -> None:
        """Start the complete managed service fleet."""
        if self._started:
            raise RuntimeError("Managed Dynamo runtime is already started")
        # Managed services use new process groups and would survive an
        # exception that escapes GRPO setup. Keep an interpreter-exit fallback
        # in addition to the normal explicit shutdown path.
        atexit.register(self.shutdown)
        self._atexit_registered = True
        self._host = _get_node_ip_local()
        used_ports: set[int] = set()

        def allocate_port(*, low: int, high: int) -> int:
            port = _get_free_port_local(
                low,
                high,
                max_retries=None,
                excluded_ports=used_ports,
            )
            used_ports.add(port)
            return port

        self._etcd_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._etcd_peer_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._nats_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._frontend_port = allocate_port(
            low=DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
        )
        self._manager_env = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            # Dynamo 1.3.0's legacy tool jail rebuilds response chunks with
            # nvext=None. Its qwen3/deepseek v2 path preserves engine_data,
            # which NeMo-Gym needs for exact token IDs and log probabilities.
            "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
            "DYN_RL_INIT_WEIGHTS_TIMEOUT_S": str(self._dynamo_cfg.control_timeout_s),
            # Matches the trainer side and VllmGeneration for non-colocated refit
            # (NeMo-RL issue #564).
            "NCCL_CUMEM_ENABLE": "1",
        }
        print(
            f"  [Dynamo] managed environment={redact_environment(self._manager_env)!r}",
            flush=True,
        )
        try:
            self._start_etcd()
            self._start_nats()
            if self._worker_pool_factory is not None:
                self._worker_pool = self._worker_pool_factory(
                    namespace=self._namespace, manager_env=dict(self._manager_env)
                )
                self._worker_pool.start(self._worker_layouts)
                self._start_frontend()
                # The engines register when they come up, which for Megatron
                # engines is the first refit; the owner calls wait_until_ready
                # then.
                self._started = True
                return
            self._pool = FixedDynamoWorkerPool(
                cluster=self._cluster,
                config=self._config,
                namespace=self._namespace,
                engine_world_size=self._vllm_engine_world_size(),
                worker_roles=self._worker_roles,
                # Only the prefill hop is KV-routed under disaggregation.
                prefill_kv_events=self._dynamo_cfg.frontend_args.router_mode == "kv",
                manager_env=self._manager_env,
                startup_timeout_s=self._dynamo_cfg.startup_timeout_s,
            )
            self._pool.start()
            self._start_frontend()
            self._started = True
            self.wait_until_ready()
        except Exception:
            self.shutdown()
            raise

    def _vllm_engine_world_size(self) -> int:
        """The one width every Dynamo-managed vLLM worker has.

        ``DynamoConfig`` rejects per-worker layouts for vLLM, so every worker
        has the same width; this guards that invariant rather than handling a
        mixed fleet.
        """
        widths = {worker.width for worker in self._worker_layouts}
        assert len(widths) == 1, (
            f"Dynamo-managed vLLM workers all share one width, got {sorted(widths)}"
        )
        return next(iter(widths))

    def wait_until_ready(self) -> None:
        """Wait for every engine to register and the model to be routable.

        Idempotent: returns at once after the first successful wait.
        """
        if self._fleet_ready:
            return
        expected_components, expected_endpoints = self.expected_registrations()
        wait_for_frontend_registrations(
            base_url=f"http://127.0.0.1:{self._frontend_port}",
            namespace=self._namespace,
            expected_components=expected_components,
            expected_endpoints=expected_endpoints,
            expected_model=str(self._config["model_name"]),
            timeout_s=self._dynamo_cfg.startup_timeout_s,
            check_alive=self._check_fleet_alive,
        )
        self._fleet_ready = True

    def _check_fleet_alive(self) -> None:
        """Raise if an engine, a worker, or the frontend process has died."""
        if self._worker_pool is not None:
            if not self._worker_pool.is_alive():
                raise RuntimeError(
                    "An engine behind the Dynamo frontend exited while the "
                    "frontend was waiting for model registration."
                )
        elif self._pool is None or not self._pool.is_alive():
            raise RuntimeError(
                "A Ray-managed Dynamo vLLM worker exited while the frontend "
                "was waiting for model registration."
            )
        if (
            self._frontend_process is not None
            and self._frontend_process.poll() is not None
        ):
            raise RuntimeError(
                f"Dynamo frontend exited with code {self._frontend_process.returncode}."
            )

    @staticmethod
    def _stop_process(
        process: subprocess.Popen | None, label: str, timeout_s: float = 15
    ) -> None:
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        print(f"  [Dynamo] {label} stopped pid={process.pid}", flush=True)

    def _service_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("DYN_", "ETCD_", "NATS_"))
        }
        env.update(self._manager_env)
        return env

    def _frontend_env(self) -> dict[str, str]:
        env = self._service_env()
        frontend_args = self._dynamo_cfg.frontend_args
        env["DYN_TOKENIZER"] = frontend_args.tokenizer
        if frontend_args.tokenizer_cache:
            env["DYN_TOKENIZER_CACHE"] = "1"
            env["DYN_TOKENIZER_CACHE_BYTES"] = str(frontend_args.tokenizer_cache_bytes)
        return env

    def _start_etcd(self) -> None:
        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_etcd_")
        peer_url = f"http://{self._host}:{self._etcd_peer_port}"
        command = [
            get_dynamo_service_executable("etcd"),
            "--listen-client-urls",
            f"http://0.0.0.0:{self._etcd_port}",
            "--advertise-client-urls",
            f"http://{self._host}:{self._etcd_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{self._etcd_peer_port}",
            "--initial-advertise-peer-urls",
            peer_url,
            "--initial-cluster",
            f"default={peer_url}",
            "--data-dir",
            self._etcd_data_dir,
        ]
        self._etcd_process = subprocess.Popen(
            command, env=self._service_env(), start_new_session=True
        )
        self._wait_for_etcd()
        print(f"  [Dynamo] etcd ready on {self._host}:{self._etcd_port}", flush=True)

    def _wait_for_etcd(self) -> None:
        url = f"http://127.0.0.1:{self._etcd_port}/health"
        deadline = time.monotonic() + min(self._dynamo_cfg.startup_timeout_s, 60)
        while time.monotonic() < deadline:
            if self._etcd_process is not None and self._etcd_process.poll() is not None:
                raise RuntimeError(
                    f"etcd exited with code {self._etcd_process.returncode}."
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        raise RuntimeError(f"etcd did not become healthy at {url}.")

    def _start_nats(self) -> None:
        self._nats_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_nats_")
        self._nats_process = subprocess.Popen(
            [
                get_dynamo_service_executable("nats-server"),
                "-js",
                "-sd",
                self._nats_data_dir,
                "-p",
                str(self._nats_port),
            ],
            env=self._service_env(),
            start_new_session=True,
        )
        self._wait_for_port(self._nats_port, "NATS", self._nats_process)
        print(f"  [Dynamo] NATS ready on {self._host}:{self._nats_port}", flush=True)

    def _wait_for_port(
        self, port: int, label: str, process: subprocess.Popen | None = None
    ) -> None:
        deadline = time.monotonic() + min(self._dynamo_cfg.startup_timeout_s, 60)
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"{label} exited with code {process.returncode}.")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{label} did not open port {port}.")

    def _start_frontend(self) -> None:
        argv = build_dynamo_frontend_argv(
            host="0.0.0.0",
            port=self._frontend_port,
            namespace=self._namespace,
            dynamo_cfg=self._dynamo_cfg,
        )
        dynamo_python = get_dynamo_python()
        command = [
            dynamo_python,
            "-m",
            "dynamo.frontend",
            *argv,
        ]
        print(
            f"  [Dynamo] launching frontend argv={redact_argv(command)!r}", flush=True
        )
        frontend_env = self._frontend_env()
        dynamo_venv = os.path.dirname(os.path.dirname(dynamo_python))
        frontend_env["VIRTUAL_ENV"] = dynamo_venv
        frontend_env["UV_PROJECT_ENVIRONMENT"] = dynamo_venv
        tokenizer_env = {
            key: value
            for key, value in frontend_env.items()
            if key.startswith("DYN_TOKENIZER")
        }
        print(
            f"  [Dynamo] frontend tokenizer environment={tokenizer_env!r}",
            flush=True,
        )
        self._frontend_process = subprocess.Popen(
            command, env=frontend_env, start_new_session=True
        )

    def refit_workers(self) -> list[dict[str, Any]]:
        self._assert_services_alive()
        if self._pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._pool.refit_workers()

    def validate_workers(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._assert_services_alive()
        if self._pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._pool.validate(expected)

    def worker_records(self) -> list[DynamoWorkerRecord]:
        """Records of the pool's engines, in role order."""
        self._assert_services_alive()
        if self._worker_pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._worker_pool.records

    def validate_worker_records(
        self, expected: list[DynamoWorkerRecord]
    ) -> list[DynamoWorkerRecord]:
        """Fail if the pool's live engines no longer match ``expected``."""
        self._assert_services_alive()
        if self._worker_pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._worker_pool.validate(expected)

    def _assert_services_alive(self) -> None:
        for label, process in (
            ("etcd", self._etcd_process),
            ("NATS", self._nats_process),
            ("frontend", self._frontend_process),
        ):
            if process is None or process.poll() is not None:
                code = None if process is None else process.returncode
                raise RuntimeError(
                    f"Managed Dynamo {label} is not alive (code={code})."
                )

    def shutdown(self) -> None:
        """Best-effort, idempotent teardown of every owned resource."""
        try:
            self._stop_process(self._frontend_process, "frontend")
        except Exception:
            LOGGER.exception("Failed to stop the managed Dynamo frontend")
        self._frontend_process = None
        pool = self._pool
        self._pool = None
        if pool is not None:
            try:
                pool.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the managed Dynamo worker pool")
        worker_pool = self._worker_pool
        self._worker_pool = None
        if worker_pool is not None:
            try:
                worker_pool.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the Dynamo worker pool")
        try:
            self._stop_process(self._nats_process, "NATS")
        except Exception:
            LOGGER.exception("Failed to stop managed Dynamo NATS")
        self._nats_process = None
        try:
            self._stop_process(self._etcd_process, "etcd")
        except Exception:
            LOGGER.exception("Failed to stop managed Dynamo etcd")
        self._etcd_process = None
        if self._etcd_data_dir is not None:
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None
        if self._nats_data_dir is not None:
            shutil.rmtree(self._nats_data_dir, ignore_errors=True)
            self._nats_data_dir = None
        if self._atexit_registered:
            atexit.unregister(self.shutdown)
            self._atexit_registered = False
        self._started = False
        self._fleet_ready = False
