
"""ToxCast endpoint discovery.

Exploratory screening only. Hit-call thresholds are provisional.
No labels or exclusions are frozen by this module.
"""

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


MEMBER = (
    "mc5-6_winning_model_fits-flags_"
    "invitrodbv4_3_AUG2024.csv"
)


def classify(hitcall):
    try:
        value = float(hitcall)
    except (TypeError, ValueError):
        return "missing"

    if not 0 <= value <= 1:
        return "missing"
    if value >= 0.9:
        return "active"
    if value <= 0.1:
        return "inactive"
    return "ambiguous"


def discover(archive: Path):
    # Per endpoint, retain classifications for each chemical.
    observations = defaultdict(lambda: defaultdict(set))
    names = {}
    flags = Counter()
    total = Counter()

    with ZipFile(archive) as z:
        with z.open(MEMBER) as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(
                    raw, encoding="utf-8-sig", newline=""
                )
            )

            for row in reader:
                aeid = row["aeid"]
                chemical = row["dsstox_substance_id"].strip()

                if not aeid:
                    continue

                names[aeid] = row["aenm"]
                total[aeid] += 1

                if not chemical or chemical.upper() == "NA":
                    continue

                observations[aeid][chemical].add(
                    classify(row["hitcall"])
                )

                if row["mc6_flags"] not in ("", "NA"):
                    flags[aeid] += 1

    results = []

    for aeid, chemicals in observations.items():
        counts = Counter()

        for calls in chemicals.values():
            if len(calls) == 1:
                counts[next(iter(calls))] += 1
            else:
                counts["conflicting"] += 1

        usable = counts["active"] + counts["inactive"]

        results.append({
            "aeid": aeid,
            "name": names.get(aeid, ""),
            "chemicals": len(chemicals),
            "active": counts["active"],
            "inactive": counts["inactive"],
            "ambiguous": counts["ambiguous"],
            "missing": counts["missing"],
            "conflicting": counts["conflicting"],
            "usable": usable,
            "active_fraction": (
                counts["active"] / usable if usable else 0
            ),
            "records": total[aeid],
            "flagged_records": flags[aeid],
        })

    return sorted(
        results,
        key=lambda r: min(r["active"], r["inactive"]),
        reverse=True,
    )


if __name__ == "__main__":
    archive = Path(
        "data/raw/toxcast_v4_3/INVITRODB_SUMMARY.zip"
    )

    results = discover(archive)

    print(
        f"{'AEID':>7} {'ACTIVE':>8} {'INACTIVE':>9} "
        f"{'USABLE':>9} {'ACT%':>7}  NAME"
    )

    candidates = [
        row for row in results
        if not row["name"].upper().startswith("TOX21_")
           and row["active"] >= 300
           and row["inactive"] >= 300
    ]

    print(f"\nNon-TOX21 candidate endpoints: {len(candidates)}")

    for row in candidates[:40]:
        print(
            f"{row['aeid']:>7} "
            f"{row['active']:>8,} "
            f"{row['inactive']:>9,} "
            f"{row['usable']:>9,} "
            f"{100 * row['active_fraction']:>6.1f}%  "
            f"{row['name']}"
        )
