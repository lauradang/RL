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

"""Adapter sidecar argv for Megatron prefill/decode engines behind Dynamo."""

import pytest

from nemo_rl.models.generation.dynamo.megatron_worker_pool import (
    _RESERVED_ADAPTER_FLAGS,
    build_megatron_adapter_argv,
    check_megatron_root,
    default_megatron_root,
    engine_layout_argv,
)

ENGINE_ARGV = ["--tensor-model-parallel-size", "2"]


def _argv(**overrides) -> list[str]:
    kwargs = dict(
        role="decode",
        component="backend",
        model_name="Qwen/Qwen3-0.6B",
        namespace="nemo-rl-7",
        parent_event_host="10.0.0.5",
        parent_event_port=4321,
        coordinator_host="10.0.0.5",
        megatron_root="/opt/megatron-lm",
        engine_start_timeout_s=600,
        handoff_owner="owner-abc",
        endpoint_types=["completions"],
        engine_argv=ENGINE_ARGV,
    )
    kwargs.update(overrides)
    return build_megatron_adapter_argv(**kwargs)


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("role", "component", "owner_flag"),
    [("decode", "backend", True), ("prefill", "prefill", False)],
)
def test_adapter_argv_is_external_mode_for_each_role(role, component, owner_flag):
    argv = _argv(role=role, component=component, extra_cli_args=["--drain-timeout=60"])
    head = argv[: argv.index("--")]
    owned = {
        "--role": role,
        "--component": component,
        "--endpoint": "generate",
        "--endpoint-types": "completions",
        "--model": "Qwen/Qwen3-0.6B",
        "--served-model-name": "Qwen/Qwen3-0.6B",
        "--namespace": "nemo-rl-7",
        "--discovery-backend": "etcd",
        "--request-plane": "tcp",
        "--event-plane": "nats",
        "--engine-launch-mode": "external",
        "--parent-event-host": "10.0.0.5",
        "--parent-event-port": "4321",
        "--coordinator-host": "10.0.0.5",
        "--megatron-root": "/opt/megatron-lm",
        "--engine-start-timeout": "600.0",
    }
    assert {flag: _flag(head, flag) for flag in owned} == owned
    assert ("--handoff-owner" in head) is owner_flag
    if owner_flag:
        assert _flag(head, "--handoff-owner") == "owner-abc"
    assert "--nproc-per-node" not in head
    # User flags come last, after the pool-owned ones.
    assert head[-1] == "--drain-timeout=60"
    # The adapter's parser requires an engine argv after '--' even though
    # external mode never launches it.
    assert argv[argv.index("--") + 1 :] == ENGINE_ARGV


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"extra_cli_args": ["--engine-launch-mode", "local"]}, "owns it"),
        ({"extra_cli_args": ["--engine-launch-mode=local"]}, "owns it"),
        ({"extra_cli_args": ["--nproc-per-node", "2"]}, "owns it"),
        ({"extra_cli_args": ["--parent-event-port=1"]}, "owns it"),
        ({"extra_cli_args": ["--coordinator-host", "h"]}, "owns it"),
        ({"extra_cli_args": ["--handoff-owner", "x"]}, "owns it"),
        ({"extra_cli_args": ["--model=other"]}, "owns it"),
        ({"extra_cli_args": ["--worker-id-file", "/x"]}, "owns it"),
        # The frontend and the adapter must agree on these planes and endpoint.
        ({"extra_cli_args": ["--request-plane", "nats"]}, "owns it"),
        ({"extra_cli_args": ["--discovery-backend=kubernetes"]}, "owns it"),
        ({"extra_cli_args": ["--event-plane", "zmq"]}, "owns it"),
        ({"extra_cli_args": ["--endpoint", "other"]}, "owns it"),
        ({"extra_cli_args": ["--endpoint-types", "chat"]}, "owns it"),
        ({"extra_cli_args": ["--engine-start-timeout=5"]}, "owns it"),
        # argparse keeps the last of a repeated flag.
        (
            {"extra_cli_args": ["--drain-timeout=1", "--drain-timeout", "2"]},
            "more than once",
        ),
        # argparse accepts unambiguous prefixes of a flag.
        ({"extra_cli_args": ["--handoff", "x"]}, "owns it"),
        ({"extra_cli_args": ["--parent-event-po=1"]}, "owns it"),
        ({"extra_cli_args": ["--", "--load", "/x"]}, "may not contain '--'"),
        ({"role": "aggregated"}, "role must be"),
        ({"engine_argv": []}, "engine_argv"),
    ],
)
def test_adapter_argv_rejects_launch_and_ownership_overrides(overrides, match):
    with pytest.raises(ValueError, match=match):
        _argv(**overrides)


def test_every_emitted_adapter_flag_is_reserved() -> None:
    # A flag the pool emits but does not reserve could be overridden by a user
    # flag appended after it.
    for role in ("decode", "prefill"):
        argv = _argv(role=role)
        head = argv[: argv.index("--")]
        emitted = {token for token in head if token.startswith("--")}
        assert emitted <= _RESERVED_ADAPTER_FLAGS


def test_check_megatron_root_requires_the_dynamo_integration(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        check_megatron_root(str(tmp_path / "missing"), node_ip="10.0.0.5")
    # A Megatron-LM tree that predates the Dynamo integration.
    (tmp_path / "megatron" / "inference").mkdir(parents=True)
    with pytest.raises(
        FileNotFoundError, match="has no megatron/inference/integrations/dynamo"
    ):
        check_megatron_root(str(tmp_path), node_ip="10.0.0.5")
    (tmp_path / "megatron" / "inference" / "integrations" / "dynamo").mkdir(
        parents=True
    )
    check_megatron_root(str(tmp_path), node_ip="10.0.0.5")


def test_endpoint_types_are_emitted_in_the_adapter_order() -> None:
    # The adapter's parser accepts only "chat", "completions" or
    # "chat,completions", so an unordered config list is canonicalised.
    argv = _argv(endpoint_types=["completions", "chat"])
    assert _flag(argv, "--endpoint-types") == "chat,completions"
    assert _flag(_argv(endpoint_types=["chat"]), "--endpoint-types") == "chat"


def test_engine_layout_argv_uses_merged_inference_layout() -> None:
    policy = {
        "model_name": "Qwen/Qwen3-0.6B",
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "generation": {
            "backend": "dynamo",
            # The same overrides the engine workers are built with win over
            # policy.megatron_cfg.
            "mcore_generation_config": {"tensor_model_parallel_size": 4},
        },
    }
    assert engine_layout_argv(policy) == [
        "--tensor-model-parallel-size",
        "4",
        "--pipeline-model-parallel-size",
        "1",
    ]
    policy["generation"]["mcore_generation_config"]["expert_model_parallel_size"] = 2
    assert engine_layout_argv(policy)[-2:] == ["--expert-model-parallel-size", "2"]


def test_default_megatron_root_is_the_bridge_nested_tree() -> None:
    assert default_megatron_root().parts[-3:] == (
        "Megatron-Bridge",
        "3rdparty",
        "Megatron-LM",
    )
