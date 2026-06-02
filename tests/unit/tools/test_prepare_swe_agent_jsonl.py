import json

import pytest

from tools.nemo_gym.prepare_swe_agent_jsonl import (
    NormalizeError,
    normalize_file,
    normalize_row,
)


def test_normalize_row_adds_agent_ref_model_and_instance_dict():
    row = {
        "responses_create_params": {
            "model": "old-model",
            "metadata": {
                "instance_id": "repo__pkg-1",
                "dataset_name": "SWE-Gym/SWE-Gym",
                "split": "train",
                "golden_patch": "diff --git a/file.py b/file.py\n",
                "test_patch": "diff --git a/test.py b/test.py\n",
            },
        }
    }

    normalized, warnings = normalize_row(
        row, agent_ref_name="swe_agents_train", model_name="new-model", line_number=1
    )

    assert warnings == []
    assert normalized["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "swe_agents_train",
    }
    params = normalized["responses_create_params"]
    assert params["model"] == "new-model"
    instance_dict = json.loads(params["metadata"]["instance_dict"])
    assert instance_dict["instance_id"] == "repo__pkg-1"
    assert instance_dict["patch"] == "diff --git a/file.py b/file.py\n"
    assert instance_dict["test_patch"] == "diff --git a/test.py b/test.py\n"


def test_normalize_row_preserves_existing_instance_dict_shape():
    existing_instance_dict = {
        "instance_id": "repo__pkg-2",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "patch": "diff --git a/src.py b/src.py\n",
        "test_patch": "diff --git a/test_src.py b/test_src.py\n",
        "repo": "owner/repo",
    }
    row = {
        "agent_ref": {"type": "responses_api_agents", "name": "old_agent"},
        "responses_create_params": {
            "model": "old-model",
            "metadata": {"instance_dict": json.dumps(existing_instance_dict)},
        },
    }

    normalized, warnings = normalize_row(
        row, agent_ref_name="swe_agents_val", model_name="new-model", line_number=7
    )

    assert warnings == []
    assert normalized["agent_ref"]["name"] == "swe_agents_val"
    instance_dict = json.loads(
        normalized["responses_create_params"]["metadata"]["instance_dict"]
    )
    assert instance_dict == existing_instance_dict


def test_normalize_row_warns_for_missing_required_swe_fields():
    row = {
        "responses_create_params": {
            "metadata": {
                "instance_id": "repo__pkg-3",
                "dataset_name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
                "golden_patch": "diff --git a/src.py b/src.py\n",
            }
        }
    }

    normalized, warnings = normalize_row(
        row, agent_ref_name="swe_agents_val", model_name="new-model", line_number=3
    )

    assert warnings == ["line 3: instance_dict has empty required SWE fields: test_patch"]
    instance_dict = json.loads(
        normalized["responses_create_params"]["metadata"]["instance_dict"]
    )
    assert instance_dict["patch"] == "diff --git a/src.py b/src.py\n"
    assert instance_dict["test_patch"] == ""


def test_normalize_file_fails_on_malformed_jsonl(tmp_path):
    input_path = tmp_path / "bad.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text('{"responses_create_params": {}}\nnot-json\n')

    with pytest.raises(NormalizeError, match="line 2: invalid JSON"):
        normalize_file(
            input_path,
            output_path,
            agent_ref_name="swe_agents_train",
            model_name="new-model",
        )


def test_normalize_file_strict_swe_fields_fails_on_missing_required_field(tmp_path):
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "responses_create_params": {
                    "metadata": {
                        "instance_id": "repo__pkg-4",
                        "dataset_name": "SWE-Gym/SWE-Gym",
                        "split": "train",
                    }
                }
            }
        )
        + "\n"
    )

    with pytest.raises(NormalizeError, match="patch, test_patch"):
        normalize_file(
            input_path,
            output_path,
            agent_ref_name="swe_agents_train",
            model_name="new-model",
            strict_swe_fields=True,
        )
