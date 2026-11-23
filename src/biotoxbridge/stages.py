"""Callable scientific preparation stages for BioToxBridge.

No OAF dependency: these functions are ordinary Python and can later be
invoked by an OAF event handler. Each stage receives explicit inputs and
returns its results; the orchestrator owns run identity and publication.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_toxicity_target(cfg: dict, audit_dir: Path) -> list[dict]:
    tc = cfg["target"]

    source = tc.get("source", "toxrefdb_3_0")
    if source == "toxcast_aeid252_frozen_v1":
        from biotoxbridge.targets.toxcast_frozen_target import load_frozen_target
        records = load_frozen_target(tc["target_csv"], tc["manifest_json"])
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "toxcast_target_selection.json").write_text(
            json.dumps({"source": source, "target_csv": tc["target_csv"],
                        "manifest_json": tc["manifest_json"], "chemicals": len(records)}, indent=2),
            encoding="utf-8",
        )
        return records
    if source != "toxrefdb_3_0":
        raise ValueError(f"Unsupported toxicity target source: {source!r}")
    from biotoxbridge.targets.toxref_target import build_target
    return build_target(
        pod_csv=tc["raw_file"], effect_substring=tc["effect_substring"],
        species=tc["species"], route=tc["route"],
        study_types=tuple(tc["study_types"]),
        duration_preference=tuple(tc["duration_preference"]),
        pod_type=tc["pod_type"],
        qualifier_exclude=tuple(tc.get("qualifier_exclude", [">"])),
        audit_csv=audit_dir / "toxref_target.csv",
    )

def resolve_chemistry(target: list[dict], interim_dir: Path, audit_dir: Path):
    from biotoxbridge.chemistry.resolver import resolve, write_audit
    resolved, missing = resolve(
        [r["dtxsid"] for r in target], cache_csv=interim_dir / "dsstox_cache.csv"
    )
    write_audit(resolved, missing,
                resolved_path=audit_dir / "resolver_resolved.csv",
                missing_path=audit_dir / "resolver_missing.csv")
    return resolved, missing


def group_structures(resolved: list[dict], cfg: dict):
    from biotoxbridge.chemistry.structure_groups import assign_groups
    return assign_groups(resolved, smiles_field=cfg["chemistry"]["structure_field"])


def score_tox21(enriched: list[dict], cfg: dict, audit_dir: Path) -> list[dict]:
    from biotoxbridge.features.tox21_adapter import Tox21Adapter
    tp = cfg["tox21_project"]
    adapter = Tox21Adapter(base_url=tp["nnsd_client_url"], endpoints=tp["endpoints"])
    scored = adapter.score_records(enriched, smiles_field="canonical_smiles")
    ref_csv = Path("data/reference/legacy_pipeline/tox21_predictions.csv")
    if ref_csv.exists():
        try:
            report = adapter.regression_test(scored, str(ref_csv), tol=0.01)
        except (KeyError, ValueError) as exc:
            report = {"status": "NOT_COMPARABLE", "all_pass": False,
                      "reason": str(exc)}
        (audit_dir / "tox21_regression.json").write_text(json.dumps(report, indent=2))
        print(f"  Tox21 regression test: {report['status']}")
    else:
        report = {"status": "NOT_COMPARABLE", "all_pass": False,
                  "reason": f"Reference file not found: {ref_csv}"}
        (audit_dir / "tox21_regression.json").write_text(json.dumps(report, indent=2))
        print("  Tox21 regression test: NOT_COMPARABLE")
    return scored


def assemble_master(target: list[dict], scored: list[dict],
                    processed_dir: Path, audit_dir: Path):
    # Classification and regression outputs must never share paths.
    if target and all(type(r.get("label")) is int and r["label"] in (0, 1) for r in target):
        from biotoxbridge.datasets.toxcast_master import build_classification_master
        return build_classification_master(
            target_records=target, scored_records=scored,
            output_csv=processed_dir / "toxcast_aeid252_master.csv",
            audit_csv=audit_dir / "toxcast_aeid252_master_exclusions.csv",
        )
    if any("label" in r for r in target):
        raise ValueError("Mixed or invalid classification target records")
    from biotoxbridge.datasets.master import build_master
    return build_master(
        target_records=target, scored_records=scored,
        output_csv=processed_dir / "liver_master.csv",
        audit_csv=audit_dir / "master_exclusions.csv",
    )


def freeze_split(master: list[dict], cfg: dict, splits_dir: Path):
    from biotoxbridge.splitting.splitter import make_split, load_split
    sp = cfg["split"]
    make_split(master, splits_dir=splits_dir,
               test_fraction=sp["test_fraction"], seed=sp["seed"],
               group_field=sp["group_by"])
    return load_split(master, splits_dir / "split_manifest.json")
