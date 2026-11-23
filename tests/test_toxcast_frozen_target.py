import csv
import hashlib
import json
from pathlib import Path

import pytest

from biotoxbridge.targets.toxcast_frozen_target import load_frozen_target
from biotoxbridge.stages import build_toxicity_target


def fixture(tmp_path):
    csv_path = tmp_path / 'toxcast_aeid252_targets.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dsstox_substance_id', 'label', 'source_record_count', 'mc6_flags'])
        writer.writeheader()
        writer.writerows([{'dsstox_substance_id': 'DTXSID001', 'label': '1', 'source_record_count': '2', 'mc6_flags': '13,14'},
                          {'dsstox_substance_id': 'DTXSID002', 'label': '0', 'source_record_count': '1', 'mc6_flags': '14'}])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'aeid': 252, 'retained_chemicals': 2, 'active_chemicals': 1,
        'inactive_chemicals': 1, 'output_files': {csv_path.name: hashlib.sha256(csv_path.read_bytes()).hexdigest()}}))
    return csv_path, manifest


def test_target_stage(tmp_path):
    target, manifest = fixture(tmp_path)
    cfg = {'target': {'source': 'toxcast_aeid252_frozen_v1', 'target_csv': str(target), 'manifest_json': str(manifest)}}
    rows = build_toxicity_target(cfg, tmp_path / 'audit')
    assert [r['dtxsid'] for r in rows] == ['DTXSID001', 'DTXSID002']
    assert [r['label'] for r in rows] == [1, 0]
    assert (tmp_path / 'audit/toxcast_target_selection.json').exists()


def test_modified_csv_fails_closed(tmp_path):
    target, manifest = fixture(tmp_path)
    target.write_text(target.read_text().replace('DTXSID002', 'DTXSID003'))
    with pytest.raises(ValueError, match='SHA-256'):
        load_frozen_target(target, manifest)


def test_unknown_source_fails(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        build_toxicity_target({'target': {'source': 'other'}}, tmp_path)
