"""Build a separate AEID 252 classification master without changing liver regression."""
from __future__ import annotations
import csv
import math
from pathlib import Path

from biotoxbridge.features.tox21_adapter import TOX21_SCORE_FIELDS

IDENTITY = ("dtxsid", "preferred_name", "casrn", "canonical_smiles", "inchikey", "structure_group")
TARGET = ("label", "source_record_count", "mc6_flags", "has_quality_flags", "has_low_replication", "has_low_concentration_count", "has_other_flags")
FIELDS = list(IDENTITY + TARGET) + list(TOX21_SCORE_FIELDS)
AUDIT_FIELDS = ("dtxsid", "exclusion_reason", "missing_fields", "invalid_fields")


def build_classification_master(target_records, scored_records, output_csv=None, audit_csv=None):
    """Join by DTXSID, fail closed on invalid scores and ungrouped structures."""
    targets = {}
    for row in target_records:
        key = row["dtxsid"]
        if key in targets:
            raise ValueError(f"Duplicate target DTXSID: {key}")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"Invalid binary label for {key}")
        targets[key] = row
    scored = {}
    for row in scored_records:
        key = row["dtxsid"]
        if key in scored:
            raise ValueError(f"Duplicate scored DTXSID: {key}")
        scored[key] = row
    master, audit = [], []
    for key, target in sorted(targets.items()):
        if key not in scored:
            audit.append(dict(dtxsid=key, exclusion_reason="no_structure_or_tox21_scores"))
            continue
        s = scored[key]
        if not s.get("canonical_smiles") or not s.get("structure_group"):
            audit.append(dict(dtxsid=key, exclusion_reason="missing_structure_or_group"))
            continue
        missing = [f for f in TOX21_SCORE_FIELDS if s.get(f) is None or str(s[f]).strip() == ""]
        if missing:
            audit.append(dict(dtxsid=key, exclusion_reason="missing_tox21_scores", missing_fields=",".join(missing)))
            continue
        numeric, invalid = {}, []
        for f in TOX21_SCORE_FIELDS:
            try:
                n = float(s[f])
                if not math.isfinite(n):
                    raise ValueError("nonfinite")
                numeric[f] = n
            except (TypeError, ValueError, OverflowError):
                invalid.append(f)
        if invalid:
            audit.append(dict(dtxsid=key, exclusion_reason="invalid_tox21_scores", invalid_fields=",".join(invalid)))
            continue
        row = {f: (s.get(f) or target.get(f, "")) for f in IDENTITY}
        row.update({f: target.get(f, "") for f in TARGET})
        row.update(numeric)
        master.append(row)
    if output_csv is not None:
        _write(Path(output_csv), FIELDS, master)
    if audit_csv is not None:
        _write(Path(audit_csv), AUDIT_FIELDS, audit)
    return master, audit


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
