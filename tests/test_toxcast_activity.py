import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from biotoxbridge.targets.toxcast_activity import FIELDS, MEMBER, extract_target


def test_extract_preserves_raw_and_audits_duplicates(tmp_path):
    archive = tmp_path / "sample.zip"
    rows = [
        {"aeid": "252", "aenm": "BSK_hDFCGF_Proliferation", "dsstox_substance_id": "DTXSID123", "hitcall": "0.95", "mc6_flags": "9,20", "m4id": "1"},
        {"aeid": "252", "aenm": "BSK_hDFCGF_Proliferation", "dsstox_substance_id": "DTXSID123", "hitcall": "0.05", "mc6_flags": "", "m4id": "2"},
        {"aeid": "252", "aenm": "BSK_hDFCGF_Proliferation", "dsstox_substance_id": "NA", "hitcall": "0", "m4id": "3"},
        {"aeid": "158", "aenm": "BSK_3C_Proliferation", "dsstox_substance_id": "DTXSID456", "hitcall": "1", "m4id": "4"},
    ]
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    with ZipFile(archive, "w") as z:
        z.writestr(MEMBER, stream.getvalue())
    out = tmp_path / "output"
    summary = extract_target(archive, out)
    assert summary["counts"]["selected_rows"] == 3
    assert summary["unique_valid_chemicals"] == 1
    assert summary["counts"]["chemicals_with_differing_hitcalls"] == 1
    assert summary["counts"]["missing_or_invalid_dtxsid"] == 1
    with (out / "toxcast_activity_records.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    with pytest.raises(FileExistsError):
        extract_target(archive, out)
