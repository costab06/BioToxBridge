"""Bind existing BioToxBridge scientific stages to OAF's JSON stage contract.

Inputs are JSON mappings. Configuration and output directories are supplied by
this factory, never by untrusted event payloads. No OAF import is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from biotoxbridge import stages as scientific
from biotoxbridge.oaf.scientific_snapshots import publish_snapshot
from biotoxbridge.features.tox21_inventory import verify_tox21_inventory


def scientific_stages(
    config: dict,
    run_dir: str | Path,
    *,
    interim_dir: str | Path | None = None,
    config_path: str | Path | None = None,
):
    """Return allowlisted scientific stages for one isolated experiment run.

    Each stage receives one JSON-compatible mapping and returns a mapping.
    The caller must create a separate mapping for each run. The resolver cache
    defaults to the run directory; an explicit cache location may be supplied.
    """
    run_dir = Path(run_dir)
    audit = run_dir / 'audit'
    interim = Path(interim_dir) if interim_dir is not None else run_dir / 'interim'
    processed = run_dir / 'processed'
    splits = run_dir / 'splits'
    for directory in (audit, interim, processed, splits):
        directory.mkdir(parents=True, exist_ok=True)

    inventory_source = Path("data/reference/tox21/model_inventory.json")
    frozen_inventory = run_dir / "tox21_model_inventory.json"

    def freeze_inventory() -> Path:
        from biotoxbridge.provenance import sha256

        if frozen_inventory.exists():
            if not frozen_inventory.is_file():
                raise ValueError("Frozen inventory is not a regular file")
            if not inventory_source.is_file():
                raise FileNotFoundError(inventory_source)
            if sha256(frozen_inventory) != sha256(inventory_source):
                raise ValueError("Frozen Tox21 inventory conflicts with source")
            return frozen_inventory

        content = inventory_source.read_bytes()

        # Exclusive creation prevents silent replacement.
        with frozen_inventory.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        return frozen_inventory

    def mapping(value: Any, *fields: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError('Scientific stage input must be a JSON object')
        for field in fields:
            if field not in value:
                raise ValueError(f'Missing scientific input: {field}')
        return value

    def target(value: Any) -> dict:
        mapping(value)
        return {'target': scientific.build_toxicity_target(config, audit)}

    def resolve(value: Any) -> dict:
        value = mapping(value, 'target')
        resolved, missing = scientific.resolve_chemistry(value['target'], interim, audit)
        publish_snapshot(
            run_dir,
            "chemistry_resolution",
            {
                "resolved": resolved,
                "missing": missing,
            },
        )
        return {'target': value['target'], 'resolved': resolved, 'missing': missing}

    def group(value: Any) -> dict:
        value = mapping(value, 'target', 'resolved')
        enriched, failed = scientific.group_structures(value['resolved'], config)
        return {**value, 'enriched': enriched, 'group_failed': failed}

    def score(value: Any) -> dict:
        value = mapping(value, 'target', 'enriched')

        tox21_config = config["tox21_project"]

        inventory_path = freeze_inventory()

        verify_tox21_inventory(
            inventory_path=inventory_path,
            weights_dir="/opt/neural/data/models",
            model_set_id=tox21_config["model_set_id"],
            endpoints=tox21_config["endpoints"],
        )

        scored = scientific.score_tox21(
            value['enriched'], config, audit
        )

        publish_snapshot(
            run_dir,
            "tox21_scoring",
            {
                "model_set_id": tox21_config["model_set_id"],
                "endpoints": tox21_config["endpoints"],
                "scored": scored,
            },
        )

        return {**value, 'scored': scored}

    def master(value: Any) -> dict:
        value = mapping(value, 'target', 'scored')
        records, excluded = scientific.assemble_master(
            value['target'], value['scored'], processed, audit)
        if not records:
            raise ValueError('Empty master dataset — check audit logs')
        return {'master': records, 'excluded': excluded}

    def split(value: Any) -> dict:
        value = mapping(value, 'master')
        train, test = scientific.freeze_split(value['master'], config, splits)
        return {'train': train, 'test': test, 'excluded': value.get('excluded', [])}

    from biotoxbridge.oaf.modeling import train_models, evaluate_models, report_models

    def model_training(value: Any) -> dict:
        value = mapping(value, 'train', 'test')
        return train_models(value, config)

    def model_evaluation(value: Any) -> dict:
        value = mapping(value, 'models', 'train_count', 'test_count')
        return evaluate_models(value, config)

    def experiment_reporting(value: Any) -> dict:
        value = mapping(value, 'models', 'metrics', 'comparison')

        result = report_models(value, run_dir)

        if config_path is None:
            return result

        from biotoxbridge.provenance import (
            write_run_manifest,
            verify_run_manifest,
        )

        config_file = Path(config_path)

        if not config_file.is_file():
            raise FileNotFoundError(
                f"Frozen experiment configuration missing: {config_file}"
            )

        split_file = splits / "split_manifest.json"

        if not split_file.is_file():
            raise FileNotFoundError(
                f"Frozen split manifest missing: {split_file}"
            )

        raw_file = Path(config["target"]["raw_file"])

        model_metadata = {
            "models": list(value["models"]),
            "tox21_model_set_id": config["tox21_project"]["model_set_id"],
            "train_count": value["train_count"],
            "test_count": value["test_count"],
            "excluded_count": value["excluded_count"],
            "evaluation": config["evaluation"],
        }
        snapshot_dir = run_dir / "snapshots"

        snapshot_files = {
            "chemistry_resolution": (
                    snapshot_dir / "chemistry_resolution.json"
            ),
            "tox21_scoring": (
                    snapshot_dir / "tox21_scoring.json"
            ),
        }

        for name, path in snapshot_files.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required scientific snapshot missing: {name}: {path}"
                )

        write_run_manifest(
            run_dir=run_dir,
            config_path=config_file,
            source_manifest={},

            artifact_paths={
                **result["report_files"],
                "split_manifest": split_file,
                **snapshot_files,
            },

            input_paths={
                "toxicity_source": raw_file,
                "tox21_model_inventory": frozen_inventory,
            },
            model_metadata=model_metadata,
        )

        verify_run_manifest(run_dir)

        return {
            **result,
            "run_manifest": str(run_dir / "run_manifest.json"),
        }


    return {
        'toxicity_target': target,
        'chemistry_resolution': resolve,
        'structure_grouping': group,
        'tox21_scoring': score,
        'master_assembly': master,
        'dataset_split': split,
        'model_training': model_training,
        'model_evaluation': model_evaluation,
        'experiment_reporting': experiment_reporting,
    }
