import csv
import math
import pytest
from biotoxbridge.datasets.toxcast_master import build_classification_master, FIELDS
from biotoxbridge.features.tox21_adapter import TOX21_SCORE_FIELDS
from biotoxbridge.stages import assemble_master


def _scored(dtxsid="DTXSID1", **updates):
    row = dict(dtxsid=dtxsid, canonical_smiles="CCO", structure_group="group1", **{f: 0.2 for f in TOX21_SCORE_FIELDS})
    row.update(updates)
    return row


def _target(dtxsid="DTXSID1", label=1):
    return dict(dtxsid=dtxsid, label=label, mc6_flags="13,14", source_record_count=2, has_low_replication=1)


def test_classification_schema_and_flags(tmp_path):
    output = tmp_path / "master.csv"
    rows, audit = build_classification_master([_target()], [_scored()], output_csv=output)
    assert len(rows) == 1 and not audit
    assert rows[0]["label"] == 1 and rows[0]["mc6_flags"] == "13,14"
    with output.open(newline="") as f:
        assert csv.DictReader(f).fieldnames == FIELDS


def test_invalid_score_excluded(tmp_path):
    bad = _scored()
    bad[TOX21_SCORE_FIELDS[0]] = math.nan
    rows, audit = build_classification_master([_target()], [bad], audit_csv=tmp_path / "audit.csv")
    assert not rows and audit[0]["exclusion_reason"] == "invalid_tox21_scores"


def test_duplicate_target_rejected():
    with pytest.raises(ValueError, match="Duplicate target"):
        build_classification_master([_target(), _target()], [_scored()])


def test_stage_uses_distinct_paths(tmp_path):
    rows, audit = assemble_master([_target()], [_scored()], tmp_path, tmp_path)
    assert len(rows) == 1
    assert (tmp_path / "toxcast_aeid252_master.csv").exists()
    assert not (tmp_path / "liver_master.csv").exists()


def test_stage_rejects_mixed_labels(tmp_path):
    with pytest.raises(ValueError, match="Mixed or invalid"):
        assemble_master([_target(), {"dtxsid": "DTXSID2"}], [], tmp_path, tmp_path)
