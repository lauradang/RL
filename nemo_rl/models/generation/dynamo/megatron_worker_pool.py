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
from typing import Any

import ray

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)

LOGGER = logging.getLogger(__name__)

_ADAPTER_MODULE = "megatron.inference.integrations.dynamo"
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


def engine_layout_argv(policy_config: Mapping[str, Any]) -> list[str]:
    """Megatron layout flags recorded after ``--`` for one engine.

    Taken from the same merged inference layout the engine workers are built
    with, so the record matches the engine the adapter attaches to.
    """
    # Imported here: the Megatron config module pulls in the policy config types.
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    megatron_cfg = merged_inference_megatron_cfg(policy_config)  # type: ignore[arg-type]
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
