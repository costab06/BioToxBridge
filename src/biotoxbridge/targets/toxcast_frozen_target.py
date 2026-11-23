"""Load and validate a frozen ToxCast AEID 252 chemical-level target."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_frozen_target(target_csv: str | Path, manifest_json: str | Path) -> list[dict]:
    target_csv, manifest_json = Path(target_csv), Path(manifest_json)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    if manifest.get("aeid") != 252:
        raise ValueError("Expected ToxCast AEID 252 manifest")
    expected = manifest.get("output_files", {}).get(target_csv.name)
    if not expected:
        raise ValueError(f"Manifest has no SHA-256 for {target_csv.name}")
    actual = hashlib.sha256(target_csv.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Frozen target SHA-256 mismatch: {target_csv}")

    with target_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dsstox_substance_id", "label", "source_record_count", "mc6_flags"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing target columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)

    records = []
    seen = set()
    for row in rows:
        dtxsid = row["dsstox_substance_id"].strip()
        if not dtxsid or dtxsid in seen:
            raise ValueError(f"Missing or duplicate DTXSID: {dtxsid!r}")
        seen.add(dtxsid)
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid binary label for {dtxsid}: {row['label']!r}")
        count = int(row["source_record_count"])
        if count < 1:
            raise ValueError(f"Invalid source record count for {dtxsid}")
        records.append({
            "dtxsid": dtxsid,
            "label": int(row["label"]),
            "source_record_count": count,
            "mc6_flags": row["mc6_flags"],
            "has_quality_flags": int(row.get("has_quality_flags") or 0),
            "has_low_replication": int(row.get("has_low_replication") or 0),
            "has_low_concentration_count": int(row.get("has_low_concentration_count") or 0),
            "has_other_flags": int(row.get("has_other_flags") or 0),
        })

    counts = Counter(r["label"] for r in records)
    for key, actual_count in (
        ("retained_chemicals", len(records)),
        ("active_chemicals", counts[1]),
        ("inactive_chemicals", counts[0]),
    ):
        if manifest.get(key) != actual_count:
            raise ValueError(f"Manifest {key} mismatch: {manifest.get(key)} != {actual_count}")
    return sorted(records, key=lambda r: r["dtxsid"])
