"""Auditable, unlabeled ToxCast assay extraction from an invitrodb summary ZIP.

This module does not infer binary labels or exclude records based on MC6 flags.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

MEMBER = "mc5-6_winning_model_fits-flags_invitrodbv4_3_AUG2024.csv"
FIELDS = (
    "spid", "chid", "casn", "chnm", "dsstox_substance_id", "code",
    "aeid", "aenm", "m4id", "m5id", "modl", "hitc", "hitcall",
    "fitc", "mc6_flags", "flag.length", "conc_min", "conc_max",
    "nconc", "npts", "nrep", "ac50", "bmd", "coff", "bmr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value.upper() in {"NA", "N/A", "NULL"} else value


def extract_target(
    archive: Path,
    output_dir: Path,
    *,
    aeid: int = 252,
    member: str = MEMBER,
) -> dict:
    """Extract all records for an AEID; retain duplicates and unclassified calls.

    Returns the audit summary. Existing outputs are not overwritten.
    """
    archive = Path(archive)
    output_dir = Path(output_dir)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "toxcast_activity_records.csv"
    chemicals_path = output_dir / "toxcast_chemical_audit.csv"
    summary_path = output_dir / "toxcast_extraction_summary.json"

    counters = Counter()
    by_chemical = defaultdict(lambda: {"rows": 0, "hitcalls": set(), "flags": set(), "m4ids": set()})
    endpoint_names = set()

    with ZipFile(archive) as z:
        info = z.getinfo(member)
        with z.open(member) as raw, records_path.open("x", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            missing = sorted(set(FIELDS) - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"Missing source columns: {missing}")
            writer = csv.DictWriter(dst, fieldnames=FIELDS + ("identifier_status",))
            writer.writeheader()
            for row in reader:
                counters["source_rows_scanned"] += 1
                if _clean(row["aeid"]) != str(aeid):
                    continue
                counters["selected_rows"] += 1
                endpoint_names.add(_clean(row["aenm"]))
                chemical = _clean(row["dsstox_substance_id"])
                valid = chemical.startswith("DTXSID") and chemical[6:].isdigit()
                status = "valid_dtxsid" if valid else "missing_or_invalid_dtxsid"
                counters[status] += 1
                clean_row = {field: _clean(row[field]) for field in FIELDS}
                writer.writerow({**clean_row, "identifier_status": status})
                if not valid:
                    continue
                entry = by_chemical[chemical]
                entry["rows"] += 1
                entry["hitcalls"].add(clean_row["hitcall"])
                if clean_row["mc6_flags"]:
                    entry["flags"].update(x.strip() for x in clean_row["mc6_flags"].split(",") if x.strip())
                    counters["flagged_valid_rows"] += 1
                if clean_row["m4id"]:
                    entry["m4ids"].add(clean_row["m4id"])

    with chemicals_path.open("x", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=(
            "dsstox_substance_id", "record_count", "distinct_hitcall_count",
            "hitcall_values", "distinct_m4id_count", "flag_codes", "duplicate_status",
        ))
        writer.writeheader()
        for chemical, entry in sorted(by_chemical.items()):
            count = entry["rows"]
            if count > 1:
                counters["chemicals_with_multiple_records"] += 1
            if len(entry["hitcalls"]) > 1:
                counters["chemicals_with_differing_hitcalls"] += 1
            writer.writerow({
                "dsstox_substance_id": chemical,
                "record_count": count,
                "distinct_hitcall_count": len(entry["hitcalls"]),
                "hitcall_values": ";".join(sorted(entry["hitcalls"])),
                "distinct_m4id_count": len(entry["m4ids"]),
                "flag_codes": ";".join(sorted(entry["flags"])),
                "duplicate_status": "multiple_records" if count > 1 else "single_record",
            })

    summary = {
        "aeid": aeid,
        "endpoint_names_in_source": sorted(endpoint_names),
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": _sha256(archive),
        "source_member": member,
        "source_member_uncompressed_bytes": info.file_size,
        "source_member_crc32": f"{info.CRC:08x}",
        "source_release_label": "AUG2024 (from member filename; not independently verified)",
        "annotation_release_compatibility": "unverified",
        "classification_policy": "none; raw hitcalls retained",
        "quality_flag_policy": "none; raw flags retained",
        "unique_valid_chemicals": len(by_chemical),
        "counts": dict(sorted(counters.items())),
        "outputs": {
            "records": records_path.name,
            "chemical_audit": chemicals_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/toxcast_v4_3/INVITRODB_SUMMARY.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/toxcast_aeid252_extraction_v1"))
    parser.add_argument("--aeid", type=int, default=252)
    args = parser.parse_args()
    print(json.dumps(extract_target(args.archive, args.output_dir, aeid=args.aeid), indent=2))


if __name__ == "__main__":
    main()
