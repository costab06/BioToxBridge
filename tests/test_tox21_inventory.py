
import hashlib
import json

import pytest

from biotoxbridge.features.tox21_inventory import verify_tox21_inventory


MODEL_SET = "tox21_continuous_v1"
ENDPOINTS = ["NR-AhR", "NR-AR"]


@pytest.fixture
def inventory_fixture(tmp_path):
    weights = tmp_path / "models"
    weights.mkdir()

    models = []
    for name in ("nr_ahr", "nr_ar"):
        content = f"weights-for-{name}".encode()
        (weights / f"{name}.mtx").write_bytes(content)

        models.append({
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "trained": True,
        })

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "model_set_id": MODEL_SET,
        "models": models,
    }))

    return inventory, weights


def verify(inventory, weights, endpoints=ENDPOINTS, model_set=MODEL_SET):
    return verify_tox21_inventory(
        inventory, weights, model_set, endpoints
    )


def test_valid_inventory(inventory_fixture):
    inventory, weights = inventory_fixture
    result = verify(inventory, weights)
    assert len(result["models"]) == 2


def test_modified_weights_rejected(inventory_fixture):
    inventory, weights = inventory_fixture
    path = weights / "nr_ahr.mtx"
    path.write_bytes(b"X" * path.stat().st_size)

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(inventory, weights)


def test_missing_weights_rejected(inventory_fixture):
    inventory, weights = inventory_fixture
    (weights / "nr_ahr.mtx").unlink()

    with pytest.raises(FileNotFoundError):
        verify(inventory, weights)


def test_wrong_model_set_rejected(inventory_fixture):
    inventory, weights = inventory_fixture

    with pytest.raises(ValueError, match="model-set"):
        verify(inventory, weights, model_set="wrong_model_set")


def test_unknown_endpoint_rejected(inventory_fixture):
    inventory, weights = inventory_fixture

    with pytest.raises(ValueError, match="Unknown"):
        verify(inventory, weights, endpoints=["UNKNOWN"])


def test_duplicate_inventory_model_rejected(inventory_fixture):
    inventory, weights = inventory_fixture
    data = json.loads(inventory.read_text())
    data["models"].append(data["models"][0])
    inventory.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Duplicate"):
        verify(inventory, weights)


def test_untrained_model_rejected(inventory_fixture):
    inventory, weights = inventory_fixture
    data = json.loads(inventory.read_text())
    data["models"][0]["trained"] = False
    inventory.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Untrained"):
        verify(inventory, weights)
