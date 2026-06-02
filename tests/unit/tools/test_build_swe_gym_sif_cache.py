import json

import pytest

from tools.nemo_gym.build_swe_gym_sif_cache import (
    SifCacheError,
    iter_sif_build_specs,
    make_docker_id,
    make_sif_build_spec,
    shard_specs,
)


def _normalized_row(instance_id: str) -> str:
    return (
        json.dumps(
            {
                "responses_create_params": {
                    "metadata": {
                        "instance_dict": json.dumps({"instance_id": instance_id})
                    }
                }
            }
        )
        + "\n"
    )


def test_make_docker_id_uses_swegym_s_separator_and_lowercases():
    assert (
        make_docker_id(
            "Project-MONAI__MONAI-3837",
            id_replacement="_s_",
            preserve_case=False,
        )
        == "project-monai_s_monai-3837"
    )


def test_make_sif_build_spec_matches_published_swegym_image_shape(tmp_path):
    spec = make_sif_build_spec(
        "getmoto__moto-7365",
        output_dir=tmp_path,
        image_namespace="xingyaoww/sweb.eval.x86_64",
        image_tag="latest",
        filename_prefix="xingyaoww_sweb.eval.x86_64",
        id_replacement="_s_",
        preserve_case=False,
    )

    assert (
        spec.image_uri
        == "docker://xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest"
    )
    assert spec.sif_path == tmp_path / "xingyaoww_sweb.eval.x86_64.getmoto_s_moto-7365.sif"


def test_iter_sif_build_specs_reads_normalized_rows_and_deduplicates(tmp_path):
    input_path = tmp_path / "data.jsonl"
    input_path.write_text(
        _normalized_row("getmoto__moto-7365")
        + _normalized_row("getmoto__moto-7365")
        + _normalized_row("python__mypy-123")
    )

    specs = iter_sif_build_specs(input_path, output_dir=tmp_path / "sifs")

    assert [spec.instance_id for spec in specs] == [
        "getmoto__moto-7365",
        "python__mypy-123",
    ]


def test_iter_sif_build_specs_fails_on_malformed_jsonl(tmp_path):
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(_normalized_row("getmoto__moto-7365") + "not-json\n")

    with pytest.raises(SifCacheError, match="line 2: invalid JSON"):
        iter_sif_build_specs(input_path, output_dir=tmp_path / "sifs")


def test_shard_specs_uses_zero_based_modulo_shards(tmp_path):
    specs = [
        make_sif_build_spec(
            f"repo__pkg-{index}",
            output_dir=tmp_path,
            image_namespace="xingyaoww/sweb.eval.x86_64",
            image_tag="latest",
            filename_prefix="xingyaoww_sweb.eval.x86_64",
            id_replacement="_s_",
            preserve_case=False,
        )
        for index in range(5)
    ]

    sharded = shard_specs(specs, shard_index=1, num_shards=2)

    assert [spec.instance_id for spec in sharded] == ["repo__pkg-1", "repo__pkg-3"]
