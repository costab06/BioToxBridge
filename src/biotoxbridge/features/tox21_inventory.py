"""Verify frozen Tox21 model inventory against local weight files."""

import hashlib
import json
from pathlib import Path

from biotoxbridge.features.tox21_adapter import _ENDPOINT_TO_MODEL


def verify_tox21_inventory(
    inventory_path: str | Path,
    weights_dir: str | Path,
    model_set_id: str,
    endpoints: list[str],
) -> dict:
    inventory_path = Path(inventory_path)
    weights_dir = Path(weights_dir)

    inventory = json.loads(inventory_path.read_text())

    if inventory["model_set_id"] != model_set_id:
        raise ValueError("Tox21 model-set identifier mismatch")

    if len(endpoints) != len(set(endpoints)):
        raise ValueError("Duplicate configured Tox21 endpoints")

    try:
        expected = {_ENDPOINT_TO_MODEL[ep] for ep in endpoints}
    except KeyError as exc:
        raise ValueError(
            f"Unknown configured Tox21 endpoint: {exc}"
        ) from exc

    models = inventory["models"]
    names = [model["name"] for model in models]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate Tox21 inventory models")

    if set(names) != expected:
        raise ValueError(
            f"Tox21 inventory mismatch: "
            f"missing={sorted(expected - set(names))}, "
            f"unexpected={sorted(set(names) - expected)}"
        )

    for model in models:
        path = weights_dir / f"{model['name']}.mtx"

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.stat().st_size != model["bytes"]:
            raise ValueError(f"Tox21 model size mismatch: {model['name']}")

        digest = hashlib.sha256()

        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != model["sha256"]:
            raise ValueError(f"Tox21 model checksum mismatch: {model['name']}")

        if model.get("trained") is not True:
            raise ValueError(f"Untrained Tox21 model: {model['name']}")

    return inventory
