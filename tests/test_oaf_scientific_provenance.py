
import json
from pathlib import Path

import pytest

from biotoxbridge.oaf.scientific import scientific_stages
from biotoxbridge.provenance import verify_run_manifest
from biotoxbridge.oaf.scientific_snapshots import publish_snapshot


def setup_run(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("chemical,score\nA,1.0\n")

    config = {
        "target": {"raw_file": str(source)},
        "tox21_project": {"model_set_id": "tox21-test-v1"},
        "evaluation": {
            "metrics": ["mae"],
            "bootstrap_iterations": 100,
            "bootstrap_seed": 42,
        },
    }

    config_file = tmp_path / "oaf_config.json"
    config_file.write_text(json.dumps(config))

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Simulate the inventory frozen by the Tox21 scoring stage.
    # These tests invoke experiment_reporting directly.
    inventory_source = Path(
        "data/reference/tox21/model_inventory.json"
    )
    frozen_inventory = run_dir / "tox21_model_inventory.json"

    frozen_inventory.write_bytes(inventory_source.read_bytes())


    publish_snapshot(
        run_dir,
        "chemistry_resolution",
        {"resolved": [], "missing": []},
    )

    publish_snapshot(
        run_dir,
        "tox21_scoring",
        {
            "model_set_id": "tox21-test-v1",
            "endpoints": [],
            "scored": [],
        },
    )
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "split_manifest.json").write_text(
        '{"train_dtxsids":["A"],"test_dtxsids":["B"]}\n'
    )



    value = {
        "train_count": 1,
        "test_count": 1,
        "excluded_count": 0,
        "models": {
            "A": {
                "predictions": [
                    {
                        "dtxsid": "B",
                        "y_true": 1.0,
                        "y_pred": 1.2,
                        "residual": -0.2,
                    }
                ]
            }
        },
        "metrics": {"A": {"mae": 0.2}},
        "comparison": {},
    }

    return config, config_file, run_dir, source, value


def test_reporting_generates_verified_manifest(tmp_path):
    config, config_file, run_dir, source, value = setup_run(tmp_path)

    stages = scientific_stages(
        config,
        run_dir,
        config_path=config_file,
    )

    result = stages["experiment_reporting"](value)

    assert Path(result["run_manifest"]).is_file()

    manifest = verify_run_manifest(run_dir)

    assert manifest["schema_version"] == 1
    assert manifest["model_metadata"]["models"] == ["A"]
    assert manifest["model_metadata"]["tox21_model_set_id"] == "tox21-test-v1"
    assert "toxicity_source" in manifest["inputs"]
    assert "split_manifest" in manifest["artifacts"]


def test_reporting_replay_preserves_manifest(tmp_path):
    config, config_file, run_dir, source, value = setup_run(tmp_path)

    stages = scientific_stages(
        config,
        run_dir,
        config_path=config_file,
    )

    stages["experiment_reporting"](value)
    before = (run_dir / "run_manifest.json").read_bytes()

    stages["experiment_reporting"](value)

    assert (run_dir / "run_manifest.json").read_bytes() == before


def test_reporting_detects_source_tampering(tmp_path):
    config, config_file, run_dir, source, value = setup_run(tmp_path)

    stages = scientific_stages(
        config,
        run_dir,
        config_path=config_file,
    )

    stages["experiment_reporting"](value)

    source.write_text("chemical,score\nA,9.0\n")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        stages["experiment_reporting"](value)
