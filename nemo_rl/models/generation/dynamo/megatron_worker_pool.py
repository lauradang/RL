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

"""Adapter sidecars for NeMo-RL-owned Megatron engines behind a Dynamo frontend.

One engine is one ``MegatronGeneration`` over its own worker group, the same
dedicated inference policy the Megatron backend already builds. Next to each
engine runs an adapter sidecar: Megatron-LM's Dynamo adapter in external-launch
mode, which registers the engine with the frontend, forwards requests to the
engine's coordinator, and carries the KV handoff between prefill and decode.
NeMo RL creates the engine process group; the sidecar only waits for the
engine's rank zero to report readiness on the parent event socket it binds.
"""

import copy
import ctypes
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import (
    DynamoConfig,
    DynamoWorkerLayout,
    DynamoWorkerRole,
)
from nemo_rl.models.generation.dynamo.protocols import DynamoWorkerRecord
from nemo_rl.utils.venvs import make_actor_runtime_env

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from nemo_rl.models.generation.megatron.megatron_generation import (
        MegatronGeneration,
    )
    from nemo_rl.models.policy import PolicyConfig

LOGGER = logging.getLogger(__name__)

_ADAPTER_MODULE = "megatron.inference.integrations.dynamo"
_ADAPTER_ACTOR_FQN = (
    "nemo_rl.models.generation.dynamo.megatron_worker_pool.DynamoMegatronAdapterActor"
)
# Dynamo component each engine role registers under.
_COMPONENT_BY_ROLE: dict[str, str] = {"prefill": "prefill", "decode": "backend"}
_HANDOFF_RECOVERY_MODULE = "megatron.inference.integrations.dynamo.handoff_recovery"
# Binding the event socket happens before the adapter touches the engine, so it
# never needs the full engine start timeout.
_EVENT_SOCKET_WAIT_CAP_S = 120.0
_PR_SET_PDEATHSIG = 1
# Loaded once at import time: the forked child between fork and exec must not
# dlopen or allocate, only make the one prctl call.
_LIBC = (
    ctypes.CDLL("libc.so.6", use_errno=True)
    if sys.platform.startswith("linux")
    else None
)

# Flags the worker pool owns: every flag build_megatron_adapter_argv emits, plus
# the local-launch flags external mode must not see. A user-supplied value would
# change which process builds the engine (launch mode, rank count), register the
# adapter on a different request plane, discovery backend or endpoint than the
# managed frontend uses, or break the event and ownership contract between the
# sidecar and the engine NeMo RL started.
_RESERVED_ADAPTER_FLAGS = frozenset(
    {
        "--endpoint",
        "--endpoint-types",
        "--discovery-backend",
        "--request-plane",
        "--event-plane",
        "--engine-start-timeout",
        "--engine-launch-mode",
        "--nproc-per-node",
        "--parent-event-host",
        "--parent-event-port",
        "--coordinator-host",
        "--coordinator-port",
        "--handoff-owner",
        "--role",
        "--component",
        "--model",
        "--served-model-name",
        "--namespace",
        "--worker-id-file",
        "--megatron-root",
    }
)
# Package the adapter module lives in, relative to the Megatron-LM tree.
_ADAPTER_PACKAGE_DIR = Path("megatron", "inference", "integrations", "dynamo")


def _die_with_parent() -> None:  # pragma: no cover
    """Ask the kernel to SIGTERM this child when its parent thread exits.

    Runs in the forked child before ``exec``. Linux only; elsewhere a no-op.
    The signal is tied to the thread that forked, which is the actor's main
    thread, so a dead sidecar actor takes its adapter process down with it.
    """
    if _LIBC is None:
        return
    if _LIBC.prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM)) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def default_megatron_root() -> Path:
    """Megatron-LM nested in the Megatron-Bridge submodule."""
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "3rdparty"
        / "Megatron-Bridge-workspace"
        / "Megatron-Bridge"
        / "3rdparty"
        / "Megatron-LM"
    )


def check_megatron_root(megatron_root: str, *, node_ip: str) -> None:
    """Fail fast unless ``megatron_root`` contains Megatron-LM's Dynamo adapter.

    Without this check a tree that lacks the adapter only surfaces as the
    adapter process exiting with code 1, with the ``ModuleNotFoundError`` left
    in the actor log.

    Args:
        megatron_root: Megatron-LM tree the adapter imports from.
        node_ip: Node the check runs on, for the error message.

    Raises:
        FileNotFoundError: When ``megatron_root`` is not a directory, or has no
            ``megatron/inference/integrations/dynamo`` package.
    """
    root = Path(megatron_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"dynamo_cfg.megatron_root {megatron_root!r} is not a directory "
            f"on {node_ip}"
        )
    if not (root / _ADAPTER_PACKAGE_DIR).is_dir():
        raise FileNotFoundError(
            f"dynamo_cfg.megatron_root {megatron_root!r} on {node_ip} has no "
            f"{_ADAPTER_PACKAGE_DIR.as_posix()}, so it cannot run Megatron's "
            "Dynamo adapter. Set dynamo_cfg.megatron_root to a Megatron-LM "
            "checkout that includes the Dynamo integration (a null value "
            "resolves to the Megatron-LM nested in Megatron-Bridge, which must "
            "then include it)."
        )


def build_megatron_adapter_argv(
    *,
    role: str,
    component: str,
    model_name: str,
    namespace: str,
    parent_event_host: str,
    parent_event_port: int,
    coordinator_host: str,
    megatron_root: str,
    engine_start_timeout_s: float,
    handoff_owner: str,
    endpoint_types: Sequence[str],
    engine_argv: Sequence[str],
    extra_cli_args: Sequence[str] = (),
) -> list[str]:
    """Arguments for ``python -m megatron.inference.integrations.dynamo``.

    External launch mode: the adapter binds the parent event socket at
    ``parent_event_host:parent_event_port`` and waits for an engine it did not
    start. The adapter's parser still requires a Megatron argument list after
    ``--``; in external mode that list is only recorded, never executed, so the
    engine's layout flags are passed there for the record.

    Args:
        role: ``prefill`` or ``decode``.
        component: Dynamo component the engine registers under.
        model_name: Model the frontend serves and the adapter advertises.
        namespace: Dynamo namespace shared by the frontend and every engine.
        parent_event_host: Interface the adapter binds its event socket on.
        parent_event_port: Port of that event socket.
        coordinator_host: Value for the adapter's ``--coordinator-host``. Its
            parser requires the flag for prefill and decode roles, but in
            external mode the adapter never uses it: it reaches the coordinator
            at the address the engine reports in its readiness message.
        megatron_root: Megatron-LM tree the adapter imports from.
        engine_start_timeout_s: How long the adapter waits for engine readiness.
        handoff_owner: Unique launch-attempt ID; only decode adapters pass it.
        endpoint_types: Endpoint kinds the adapter exposes, e.g. ``completions``.
        engine_argv: Megatron layout flags recorded after ``--``.
        extra_cli_args: User flags inserted before ``--``.

    Returns:
        The adapter's command-line arguments, module name excluded.

    Raises:
        ValueError: On an unsupported role, a user flag the pool owns, a user
            ``--`` separator, a user flag given twice, or an empty
            ``engine_argv``.
    """
    if role not in ("prefill", "decode"):
        raise ValueError(f"role must be 'prefill' or 'decode', got {role!r}")
    user_flags: set[str] = set()
    for arg in extra_cli_args:
        flag = str(arg).split("=", 1)[0]
        # argparse accepts unambiguous prefixes, so a prefix of an owned flag
        # would set it just the same.
        if len(flag) >= 3 and any(
            reserved.startswith(flag) for reserved in _RESERVED_ADAPTER_FLAGS
        ):
            raise ValueError(
                "dynamo_cfg.worker_args.extra_cli_args may not set the adapter "
                f"flag {flag}; the Megatron worker pool owns it"
            )
        if flag.startswith("--") and flag != "--":
            # argparse keeps the last value, so a repeated flag silently drops
            # the earlier one.
            if flag in user_flags:
                raise ValueError(
                    "dynamo_cfg.worker_args.extra_cli_args sets the adapter flag "
                    f"{flag} more than once"
                )
            user_flags.add(flag)
    if "--" in extra_cli_args:
        raise ValueError(
            "dynamo_cfg.worker_args.extra_cli_args may not contain '--'; the "
            "Megatron engine is launched by NeMo RL, not by the adapter"
        )
    if not engine_argv:
        raise ValueError("engine_argv must describe the engine layout")
    argv = [
        "--role",
        role,
        "--component",
        component,
        "--endpoint",
        "generate",
        "--endpoint-types",
        # The adapter accepts only "chat", "completions" or "chat,completions".
        ",".join(kind for kind in ("chat", "completions") if kind in endpoint_types),
        "--model",
        model_name,
        "--served-model-name",
        model_name,
        "--namespace",
        namespace,
        "--discovery-backend",
        "etcd",
        "--request-plane",
        "tcp",
        "--event-plane",
        "nats",
        "--engine-launch-mode",
        "external",
        "--parent-event-host",
        parent_event_host,
        "--parent-event-port",
        str(parent_event_port),
        "--coordinator-host",
        coordinator_host,
        "--megatron-root",
        megatron_root,
        "--engine-start-timeout",
        str(float(engine_start_timeout_s)),
    ]
    if role == "decode":
        argv.extend(["--handoff-owner", handoff_owner])
    argv.extend(str(arg) for arg in extra_cli_args)
    argv.append("--")
    argv.extend(str(arg) for arg in engine_argv)
    return argv


def role_policy_config(
    policy_config: Mapping[str, Any], overrides: Mapping[str, int]
) -> dict[str, Any]:
    """The policy config one worker builds its engine from.

    A deep copy of ``policy_config`` with ``overrides`` (the worker's entry in
    ``dynamo_cfg.prefill_engines`` / ``decode_engines``) merged into
    ``generation.mcore_generation_config``, which is where the dedicated
    inference policy takes its layout from. ``policy_config`` is left as is.

    Args:
        policy_config: The full ``policy`` config.
        overrides: Layout keys from ``DynamoConfig.layout_overrides``; empty
            for a worker that keeps the shared layout.

    Returns:
        A fresh config whose engine layout is the shared one plus ``overrides``.
    """
    config = copy.deepcopy(dict(policy_config))
    if overrides:
        generation = dict(config["generation"])
        generation["mcore_generation_config"] = {
            **(generation.get("mcore_generation_config") or {}),
            **overrides,
        }
        config["generation"] = generation
    return config


def engine_layout_argv(
    policy_config: Mapping[str, Any], overrides: Mapping[str, int] | None = None
) -> list[str]:
    """Megatron layout flags recorded after ``--`` for one engine.

    Taken from the same merged inference layout the engine workers are built
    with, so the record matches the engine the adapter attaches to.

    Args:
        policy_config: The full ``policy`` config.
        overrides: The worker's layout keys, from ``DynamoConfig.layout_overrides``;
            ``None`` or empty records the shared layout.
    """
    # Imported here: the Megatron config module pulls in the policy config types.
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    megatron_cfg = merged_inference_megatron_cfg(
        policy_config,  # type: ignore[arg-type]
        overrides=overrides,
    )
    argv = [
        "--tensor-model-parallel-size",
        str(int(megatron_cfg["tensor_model_parallel_size"])),
        "--pipeline-model-parallel-size",
        str(int(megatron_cfg["pipeline_model_parallel_size"])),
    ]
    expert = int(megatron_cfg["expert_model_parallel_size"])
    if expert > 1:
        argv.extend(["--expert-model-parallel-size", str(expert)])
    return argv


@ray.remote(num_cpus=0)
class DynamoMegatronAdapterActor:  # pragma: no cover
    """Own one Megatron Dynamo adapter subprocess next to an engine's rank zero.

    Scheduled on the engine's first placement-group bundle with no GPU, so the
    parent event socket it binds is on the node rank zero connects from. The
    adapter reaches the engine's coordinator at the address rank zero reports
    when it is ready; the node IP passed as ``--coordinator-host`` only
    satisfies the adapter's parser in external mode.

    The vLLM sidecar registers its child with a cleanup reservation so a dead
    actor leaves no detached engine. This actor instead has the kernel deliver
    SIGTERM to the adapter when the actor process dies (parent-death signal, a
    Linux feature; a no-op elsewhere), which covers the same orphan case.
    """

    def __init__(
        self,
        *,
        instance_id: str,
        role: str,
        component: str,
        model_name: str,
        namespace: str,
        megatron_root: str,
        manager_env: dict[str, str],
        engine_start_timeout_s: float,
        endpoint_types: list[str],
        extra_cli_args: list[str],
        engine_argv: list[str],
        excluded_ports: list[int],
    ) -> None:
        self._instance_id = instance_id
        self._role = role
        self._node_ip = _get_node_ip_local()
        self._handoff_owner = uuid.uuid4().hex
        self._process: subprocess.Popen | None = None
        used = set(excluded_ports)
        self._event_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        used.add(self._event_port)
        self._system_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        check_megatron_root(megatron_root, node_ip=self._node_ip)
        # This actor runs in the dynamo extra's venv (actor_environments.py), the
        # same venv the vLLM sidecar launches from, so the adapter and the
        # frontend resolve one ai-dynamo install.
        dynamo_python = sys.executable
        dynamo_venv = sys.prefix
        env = build_managed_worker_env(
            base_env=os.environ,
            configured_env={},
            manager_env={
                **manager_env,
                "DYN_SYSTEM_PORT": str(self._system_port),
                # The sidecar never touches a GPU; the engine ranks own them.
                "CUDA_VISIBLE_DEVICES": "",
                "VIRTUAL_ENV": dynamo_venv,
                "UV_PROJECT_ENVIRONMENT": dynamo_venv,
            },
        )
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (megatron_root, env.get("PYTHONPATH", "")) if path
        )
        argv = build_megatron_adapter_argv(
            role=role,
            component=component,
            model_name=model_name,
            namespace=namespace,
            parent_event_host=self._node_ip,
            parent_event_port=self._event_port,
            coordinator_host=self._node_ip,
            megatron_root=megatron_root,
            engine_start_timeout_s=engine_start_timeout_s,
            handoff_owner=self._handoff_owner,
            endpoint_types=endpoint_types,
            engine_argv=engine_argv,
            extra_cli_args=extra_cli_args,
        )
        command = [dynamo_python, "-m", _ADAPTER_MODULE, *argv]
        relevant_env = {
            key: value
            for key, value in env.items()
            if key.startswith(("DYN_", "ETCD_", "NATS_", "PYTHONPATH"))
        }
        print(
            f"  [Dynamo:{instance_id}] launching adapter argv={redact_argv(command)!r} "
            f"env={redact_environment(relevant_env)!r} "
            f"parent_event={self.parent_event_address}",
            flush=True,
        )
        self._command = command
        self._env = env
        for port_name, port in (
            ("parent event port", self._event_port),
            ("DYN_SYSTEM_PORT", self._system_port),
        ):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(
                    f"{port_name} {port} is unavailable for the Dynamo Megatron "
                    f"adapter of {instance_id} on {self._node_ip}."
                ) from exc
        try:
            self._process = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
                preexec_fn=_die_with_parent,
            )
            self._wait_for_event_port(engine_start_timeout_s)
        except Exception:
            self._stop_process()
            raise

    @property
    def parent_event_address(self) -> str:
        """``tcp://host:port`` the engine's rank zero reports readiness to."""
        host = f"[{self._node_ip}]" if ":" in self._node_ip else self._node_ip
        return f"tcp://{host}:{self._event_port}"

    def _wait_for_event_port(self, timeout_s: float) -> None:
        wait_s = min(timeout_s, _EVENT_SOCKET_WAIT_CAP_S)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Dynamo Megatron adapter for {self._instance_id} exited with "
                    f"code {process.returncode} before binding its event socket."
                )
            try:
                # The adapter binds the event socket on the node IP it was
                # given, not on loopback.
                with socket.create_connection(
                    (self._node_ip, self._event_port), timeout=1
                ):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(
            f"Dynamo Megatron adapter for {self._instance_id} did not bind its "
            f"event socket on {self._node_ip}:{self._event_port} within {wait_s}s "
            f"(the engine start timeout, capped at {_EVENT_SOCKET_WAIT_CAP_S}s for "
            "this step)."
        )

    def metadata(self) -> dict[str, Any]:
        """Identity and addresses of the running adapter."""
        process = self._process
        if process is None:
            raise RuntimeError(
                f"Dynamo Megatron adapter for {self._instance_id} is not running."
            )
        return {
            "instance_id": self._instance_id,
            "role": self._role,
            "node_ip": self._node_ip,
            "parent_event_address": self.parent_event_address,
            "system_port": self._system_port,
            "event_port": self._event_port,
            "handoff_owner": self._handoff_owner,
            "process_pid": process.pid,
        }

    def is_alive(self) -> bool:
        """Whether the adapter subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    def confirm_terminated(
        self, coordinator_addresses: list[str], owner: str, timeout_s: float
    ) -> bool:
        """Tell every prefill coordinator that a decode attempt is gone for good.

        Runs Megatron's supervisor hook in the adapter environment so the
        prefill engines release the KV blocks pinned for that owner. Only valid
        after the decode adapter and every decode rank have exited. The hook
        imports Megatron core, so ``timeout_s`` must cover several seconds of
        interpreter startup on top of the coordinator round trips; the hook
        itself does not retry an unreachable coordinator, so callers that need
        the attestation to land retry this call.

        Returns:
            True when every coordinator accepted the attestation; False (after
            logging) when the hook failed or timed out.
        """
        if not coordinator_addresses:
            return True
        command = [self._command[0], "-m", _HANDOFF_RECOVERY_MODULE]
        for address in coordinator_addresses:
            command.extend(["--coordinator-address", address])
        command.extend(["--confirm-terminated-owner", owner])
        try:
            result = subprocess.run(
                command,
                env=self._env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            LOGGER.error("handoff recovery for owner %s exceeded %ss", owner, timeout_s)
            return False
        if result.returncode != 0:
            LOGGER.error(
                "handoff recovery for owner %s failed: stdout=%r stderr=%r",
                owner,
                result.stdout,
                result.stderr,
            )
            return False
        return True

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self._process = None

    def shutdown(self) -> bool:
        """Stop the adapter: SIGTERM its process group, SIGKILL after 15 s."""
        process = self._process
        if process is None:
            return True
        self._stop_process()
        print(
            f"  [Dynamo:{self._instance_id}] adapter stopped pid={process.pid}",
            flush=True,
        )
        return True


@ray.remote(num_cpus=0)
def cluster_node_ip() -> str:  # pragma: no cover
    """IP of the node a placement-group bundle lives on."""
    return _get_node_ip_local()


class MegatronDynamoWorkerPool:
    """``DynamoWorkerPool`` of NeMo-RL-owned Megatron engines plus adapter sidecars.

    Every engine is a dedicated ``MegatronGeneration`` over its own one-node
    virtual cluster, built with ``skip_weight_load=True`` so its inference engine
    comes up at the first weight refit. Every engine has the layout the policy
    config gives plus its own entry in ``dynamo_cfg.prefill_engines`` /
    ``decode_engines``, so any two engines may differ in TP and PP (the KV
    handoff reshards between them). An engine's role also picks the Dynamo
    component it registers under, the engine variant and KV-transfer agents its
    ranks bring up when the adapter attaches, and its place in the teardown
    order.
    """

    def __init__(
        self,
        *,
        policy_config: Mapping[str, Any],
        tokenizer: "PreTrainedTokenizerBase",
        config: DynamoConfig,
        namespace: str,
        manager_env: Mapping[str, str],
        num_gpus_per_node: int,
        port_range_low: int | None,
        port_range_high: int | None,
        node_resource_constraints: list[dict[str, float]] | None,
    ) -> None:
        """Remember what every engine and sidecar is built from.

        Args:
            policy_config: The full ``policy`` config; each engine is built from
                it with its role's layout overrides applied.
            tokenizer: Tokenizer handed to every engine.
            config: Validated generation config with ``dynamo_cfg.engine='megatron'``.
            namespace: Dynamo namespace shared by the frontend and every engine.
            manager_env: etcd / NATS connection variables for the sidecars.
            num_gpus_per_node: GPUs per node; one engine fits on one node.
            port_range_low: Lower bound of every engine's master-port range,
                as the shared inference cluster was given; ``None`` keeps the
                ``RayVirtualCluster`` default.
            port_range_high: Upper bound (exclusive) of that range; ``None``
                keeps the ``RayVirtualCluster`` default.
            node_resource_constraints: The shared inference cluster's per-node
                NVLink-domain pins, or ``None`` when it has none. Engines tile
                those nodes in role order, so each engine inherits the pin of
                the inference node it takes the place of.

        Raises:
            ValueError: When ``config`` does not select Megatron engines.
        """
        if not config.is_megatron_engine:
            raise ValueError(
                "MegatronDynamoWorkerPool requires dynamo_cfg.engine='megatron'"
            )
        # The engines take the policy config as-is; MegatronGeneration types it.
        self._policy_config = cast("PolicyConfig", policy_config)
        self._tokenizer = tokenizer
        self._config = config
        self._namespace = namespace
        self._manager_env = dict(manager_env)
        self._num_gpus_per_node = num_gpus_per_node
        self._port_range_low = port_range_low
        self._port_range_high = port_range_high
        self._node_resource_constraints = node_resource_constraints
        megatron_root = config.dynamo_cfg.megatron_root
        self._megatron_root = str(
            Path(megatron_root).expanduser()
            if megatron_root
            else default_megatron_root()
        )
        self._roles: list[DynamoWorkerRole] = []
        self._records: list[DynamoWorkerRecord] = []
        self._clusters: list[RayVirtualCluster] = []
        self._engines: list["MegatronGeneration"] = []
        self._adapters: list[ray.actor.ActorHandle] = []
        self._adapter_metadata: list[dict[str, Any]] = []

    @property
    def records(self) -> list[DynamoWorkerRecord]:
        """Records returned by :meth:`start`, in engine order."""
        return list(self._records)

    def expected_components(self) -> dict[str, int]:
        """Dynamo component name to the number of engines registering under it."""
        counts: dict[str, int] = {}
        for role in self._roles:
            component = _COMPONENT_BY_ROLE[role]
            counts[component] = counts.get(component, 0) + 1
        return counts

    def start(self, workers: Sequence[DynamoWorkerLayout]) -> list[DynamoWorkerRecord]:
        """Bring up one engine and one sidecar per worker, in order.

        Args:
            workers: Role, width, and layout overrides per engine, from
                ``DynamoConfig.worker_layouts``; the order is the refit rank
                order.

        Returns:
            One record per engine, in ``workers`` order.

        Raises:
            ValueError: On an aggregated role, an engine wider than a node, or
                more engines than the node pins hold.
            RuntimeError: When the pool is already started, or an engine's
                GPUs land on more than one node.
        """
        if self._engines or self._adapters:
            raise RuntimeError("MegatronDynamoWorkerPool is already started")
        workers = list(workers)
        roles = [worker.role for worker in workers]
        unsupported = sorted({role for role in roles if role not in _COMPONENT_BY_ROLE})
        if unsupported:
            raise ValueError(
                "Megatron engines behind Dynamo are prefill or decode engines; "
                f"got {unsupported}. Aggregated Megatron engines are not supported "
                "(use policy.generation.backend=megatron)."
            )
        for worker in workers:
            if worker.width > self._num_gpus_per_node:
                raise ValueError(
                    f"A Megatron {worker.role} engine spans {worker.width} GPUs "
                    f"but nodes have {self._num_gpus_per_node}; multi-node "
                    "engines are not supported behind Dynamo. Lower "
                    f"policy.generation.dynamo_cfg.{worker.role}_engines or "
                    "mcore_generation_config tensor_model_parallel_size x "
                    "pipeline_model_parallel_size."
                )
        # Which pinned inference node each engine takes, when the shared
        # inference cluster has node pins. Engines tile the pinned nodes in
        # launch order, each engine wholly on one node: an engine that does not
        # fit in the GPUs the earlier engines left on the current node starts
        # the next one. With one width this is engine_index // engines_per_node.
        constraints = self._node_resource_constraints
        pin_index_by_engine: list[int] = []
        node_index = 0
        used_on_node = 0
        for worker in workers:
            width = worker.width
            if used_on_node + width > self._num_gpus_per_node:
                node_index += 1
                used_on_node = 0
            pin_index_by_engine.append(node_index)
            used_on_node += width
        nodes_needed = node_index + 1
        if constraints is not None and nodes_needed > len(constraints):
            widths = [worker.width for worker in workers]
            raise ValueError(
                f"{len(workers)} Megatron engines of widths {widths} GPUs tile "
                f"{nodes_needed} nodes of {self._num_gpus_per_node} GPUs, more "
                f"than the {len(constraints)} pinned inference nodes."
            )
        self._roles = roles
        try:
            # Imported here: this module is also the sidecar actor's module and
            # is imported inside the Dynamo venv, which never builds an engine
            # and need not load the Megatron generation backend and its
            # generation-config chain.
            from nemo_rl.models.generation.megatron.megatron_generation import (
                MegatronGeneration,
            )

            # Builds the dynamo extra's venv on every node when uv-managed, the
            # same venv the vLLM sidecar launches from.
            runtime_env = make_actor_runtime_env(
                _ADAPTER_ACTOR_FQN,
                extra_env_vars={"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"},
            )
            ports_by_node: dict[str, set[int]] = {}
            role_counts: dict[str, int] = {}
            for engine_index, worker in enumerate(workers):
                role = worker.role
                ordinal = role_counts.get(role, 0)
                role_counts[role] = ordinal + 1
                instance_id = f"{role}-{ordinal}"
                width = worker.width
                # Each engine is built from the policy config with its own
                # layout overrides, and its sidecar records the same layout.
                engine_policy_config = cast(
                    "PolicyConfig",
                    role_policy_config(self._policy_config, worker.overrides),
                )
                cluster = RayVirtualCluster(
                    name=f"dynamo_{self._namespace}_{instance_id}",
                    bundle_ct_per_node_list=[width],
                    use_gpus=True,
                    num_gpus_per_node=self._num_gpus_per_node,
                    max_colocated_worker_groups=1,
                    port_range_low=self._port_range_low,
                    port_range_high=self._port_range_high,
                    node_resource_constraints=(
                        None
                        if constraints is None
                        else [constraints[pin_index_by_engine[engine_index]]]
                    ),
                )
                self._clusters.append(cluster)
                # Reserved the way the engine will reserve them (always PACK,
                # since Dynamo engines are never colocated; idempotent), so a
                # split engine fails before any worker is scheduled.
                MegatronGeneration.init_cluster_placement_groups(
                    cluster, engine_policy_config
                )
                node_ip = self._engine_node_ip(cluster, instance_id)
                engine = MegatronGeneration(
                    config=engine_policy_config,
                    tokenizer=self._tokenizer,
                    cluster=cluster,
                    name_prefix=f"dynamo_{instance_id}",
                    skip_weight_load=True,
                )
                self._engines.append(engine)
                metadata = self._start_adapter(
                    instance_id=instance_id,
                    role=role,
                    cluster=cluster,
                    node_ip=node_ip,
                    runtime_env=runtime_env,
                    engine_argv=engine_layout_argv(
                        self._policy_config, overrides=worker.overrides
                    ),
                    ports_by_node=ports_by_node,
                )
                engine.attach_dynamo_adapter(
                    parent_event_address=metadata["parent_event_address"], role=role
                )
                self._records.append(
                    DynamoWorkerRecord(
                        role=role,
                        world_size=width,
                        instance_id=instance_id,
                        node_ip=metadata["node_ip"],
                        control=engine,
                    )
                )
                print(
                    f"  [Dynamo:{instance_id}] engine ready for refit on "
                    f"{metadata['node_ip']} ({width} ranks), adapter "
                    f"owner={metadata['handoff_owner']}",
                    flush=True,
                )
        except Exception:
            self.shutdown()
            raise
        return self.records

    def _engine_node_ip(self, cluster: RayVirtualCluster, instance_id: str) -> str:
        """Node every bundle of an engine's cluster sits on.

        ``PACK`` is best effort: on a fragmented or partly busy cluster Ray may
        spread an engine's bundles over several nodes, and its tensor-parallel
        group would then silently run across them.
        """
        refs = [
            cluster_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                )
            ).remote()
            for placement_group in cluster.get_placement_groups()
            for bundle_index in range(placement_group.bundle_count)
        ]
        node_ips = set(ray.get(refs))
        if len(node_ips) != 1:
            raise RuntimeError(
                f"Megatron engine {instance_id} spans multiple nodes "
                f"{sorted(node_ips)}. Multi-node engines are not supported "
                "behind Dynamo."
            )
        return next(iter(node_ips))

    def _start_adapter(
        self,
        *,
        instance_id: str,
        role: str,
        cluster: RayVirtualCluster,
        node_ip: str,
        runtime_env: dict[str, Any],
        engine_argv: list[str],
        ports_by_node: dict[str, set[int]],
    ) -> dict[str, Any]:
        """Schedule the sidecar on the engine's first bundle and return its metadata."""
        placement_group = cluster.get_placement_groups()[0]
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        # Ports are node-local: a later sidecar on a node avoids the ports the
        # earlier sidecars on that node reported.
        used_ports = ports_by_node.setdefault(node_ip, set())
        worker_args = self._config.dynamo_cfg.worker_args
        adapter = DynamoMegatronAdapterActor.options(
            num_gpus=0,
            runtime_env=runtime_env,
            scheduling_strategy=strategy,
            name=f"{self._namespace}-dynamo-adapter-{instance_id}",
        ).remote(
            instance_id=instance_id,
            role=role,
            component=_COMPONENT_BY_ROLE[role],
            model_name=str(self._policy_config["model_name"]),
            namespace=self._namespace,
            megatron_root=self._megatron_root,
            manager_env=self._manager_env,
            engine_start_timeout_s=self._config.dynamo_cfg.startup_timeout_s,
            endpoint_types=list(worker_args.endpoint_types),
            extra_cli_args=list(worker_args.extra_cli_args),
            engine_argv=list(engine_argv),
            excluded_ports=sorted(used_ports),
        )
        # Appended before the first call so a failed launch is still torn down.
        self._adapters.append(adapter)
        metadata = dict(ray.get(adapter.metadata.remote()))
        self._adapter_metadata.append(metadata)
        used_ports.update({int(metadata["event_port"]), int(metadata["system_port"])})
        return metadata

    def is_alive(self) -> bool:
        """Whether every sidecar process and every engine's rank zero is alive."""
        if not self._records:
            return False
        # A dead actor must read as "not alive", not raise, so validate() can
        # report it.
        try:
            adapters_alive = ray.get(
                [adapter.is_alive.remote() for adapter in self._adapters]
            )
            if not all(adapters_alive):
                return False
            engines_alive = ray.get(
                [engine.shard_liveness_ref(0) for engine in self._engines]
            )
        except ray.exceptions.RayActorError:
            return False
        return all(engines_alive)

    def validate(
        self, expected: Sequence[DynamoWorkerRecord]
    ) -> list[DynamoWorkerRecord]:
        """Fail if the fleet died, a sidecar restarted, or identities changed."""
        if not self.is_alive():
            raise RuntimeError(
                "A Megatron engine or its Dynamo adapter exited behind the frontend."
            )
        current = ray.get([adapter.metadata.remote() for adapter in self._adapters])
        recorded_pids = [m["process_pid"] for m in self._adapter_metadata]
        if [m["process_pid"] for m in current] != recorded_pids:
            raise RuntimeError(
                "Dynamo adapter membership changed after collective initialization."
            )
        identities = [
            (r.role, r.world_size, r.instance_id, r.node_ip) for r in expected
        ]
        live = [(r.role, r.world_size, r.instance_id, r.node_ip) for r in self._records]
        if identities != live:
            raise RuntimeError(
                "Megatron engine membership changed after collective "
                f"initialization: expected={identities}, current={live}."
            )
        return self.records

    def prefill_coordinator_addresses(self) -> list[str]:
        """Coordinator addresses of the prefill engines that are listening."""
        addresses = []
        # After a partial start there are engines without a record yet; none of
        # them has a coordinator.
        engines = self._engines[: len(self._records)]
        for record, engine in zip(self._records, engines, strict=True):
            if record.role != "prefill":
                continue
            address = engine.dynamo_coordinator_address()
            if address:
                addresses.append(address)
        return addresses

    def shutdown(self) -> None:
        """Decode first, then attest their termination to prefill, then prefill.

        Prefill engines keep KV blocks pinned for decode attempts they have not
        heard back from. Once every decode rank is gone, their owner IDs are
        reported to each prefill coordinator so the pins are released before
        the prefill engines themselves go down. A decode engine that did not
        stop cleanly is not attested. Best effort and idempotent: every failure
        is logged and the teardown continues.
        """
        indexed = list(enumerate(self._records))
        decode = [i for i, record in indexed if record.role == "decode"]
        prefill = [i for i, record in indexed if record.role == "prefill"]
        stopped_decode = []
        for index in decode:
            if self._stop_engine(index):
                stopped_decode.append(index)
            else:
                LOGGER.warning(
                    "Decode engine %s did not stop cleanly; its handoff owner is "
                    "not attested as terminated",
                    self._records[index].instance_id,
                )
        if stopped_decode and prefill:
            self._confirm_decode_terminated(
                attester=self._adapters[prefill[0]], decode_indices=stopped_decode
            )
        for index in prefill:
            self._stop_engine(index)
        # Engines and sidecars that failed before their record existed.
        for index in range(
            len(self._records), max(len(self._engines), len(self._adapters))
        ):
            self._stop_engine(index)
        for cluster in self._clusters:
            try:
                cluster.shutdown()
            except Exception:
                LOGGER.exception("Could not remove an engine placement group")
        self._roles = []
        self._records.clear()
        self._engines.clear()
        self._adapters.clear()
        self._adapter_metadata.clear()
        self._clusters.clear()

    def _confirm_decode_terminated(
        self, *, attester: ray.actor.ActorHandle, decode_indices: list[int]
    ) -> None:
        try:
            addresses = self.prefill_coordinator_addresses()
        except Exception:
            LOGGER.exception("Could not read prefill coordinator addresses")
            return
        if not addresses:
            LOGGER.info(
                "No prefill coordinator is listening (expected before the first "
                "refit); skipping handoff attestation for %d decode engines",
                len(decode_indices),
            )
            return
        timeout_s = self._config.dynamo_cfg.control_timeout_s
        for index in decode_indices:
            owner = self._adapter_metadata[index].get("handoff_owner")
            if not owner:
                continue
            try:
                accepted = ray.get(
                    attester.confirm_terminated.remote(addresses, owner, timeout_s),
                    timeout=timeout_s + 30,
                )
            except Exception:
                LOGGER.exception("Could not attest decode owner %s termination", owner)
                continue
            if not accepted:
                LOGGER.warning(
                    "Prefill coordinators did not accept decode owner %s "
                    "termination; its pinned KV blocks stay until the prefill "
                    "engines stop (see the prefill adapter's log)",
                    owner,
                )

    def _stop_engine(self, index: int) -> bool:
        """Stop one engine's sidecar and policy; whether both stopped cleanly."""
        stopped = True
        if index < len(self._adapters):
            adapter = self._adapters[index]
            try:
                ray.get(adapter.shutdown.remote(), timeout=30)
            except Exception:
                LOGGER.exception("Could not stop a Dynamo Megatron adapter")
                stopped = False
            try:
                ray.kill(adapter, no_restart=True)
            except Exception:
                LOGGER.exception("Could not kill a Dynamo Megatron adapter actor")
        if index < len(self._engines):
            try:
                engine_stopped = self._engines[index].shutdown()
            except Exception:
                LOGGER.exception("Could not stop a Megatron engine")
                stopped = False
            else:
                if not engine_stopped:
                    LOGGER.error("A Megatron engine did not shut down cleanly")
                    stopped = False
        return stopped
