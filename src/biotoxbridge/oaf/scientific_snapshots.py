"""Immutable snapshots of external scientific stage results."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from biotoxbridge.oaf.modeling import finite_json


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            finite_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def publish_snapshot(
    run_dir: str | Path,
    name: str,
    value: object,
) -> dict:
    """Publish a snapshot once; reject conflicting replays."""

    if not name or not all(
        c.isascii() and (c.isalnum() or c in "_-")
        for c in name
    ):
        raise ValueError("Invalid snapshot name")

    directory = Path(run_dir) / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{name}.json"
    content = canonical_bytes(value)

    try:
        with path.open("xb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(
                f"Scientific snapshot conflict: {name}"
            )

    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def load_snapshot(
    run_dir: str | Path,
    name: str,
) -> object:
    """Load a previously published scientific snapshot."""

    if not name or not all(
        c.isascii() and (c.isalnum() or c in "_-")
        for c in name
    ):
        raise ValueError("Invalid snapshot name")

    path = Path(run_dir) / "snapshots" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
