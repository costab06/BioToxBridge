import pytest

from biotoxbridge.oaf.scientific_snapshots import (
    load_snapshot,
    publish_snapshot,
)


def test_snapshot_roundtrip(tmp_path):
    value = {
        "resolved": [
            {"dtxsid": "D1", "smiles": "CCO"}
        ],
        "missing": [],
    }

    record = publish_snapshot(
        tmp_path, "chemistry_resolution", value
    )

    assert record["sha256"]
    assert load_snapshot(
        tmp_path, "chemistry_resolution"
    ) == value


def test_identical_replay(tmp_path):
    value = {"scored": [{"dtxsid": "D1", "score": 0.5}]}

    first = publish_snapshot(
        tmp_path, "tox21_scoring", value
    )
    second = publish_snapshot(
        tmp_path, "tox21_scoring", value
    )

    assert first == second


def test_conflicting_replay_rejected(tmp_path):
    publish_snapshot(
        tmp_path, "tox21_scoring", {"score": 0.5}
    )

    with pytest.raises(ValueError, match="conflict"):
        publish_snapshot(
            tmp_path, "tox21_scoring", {"score": 0.9}
        )


def test_invalid_snapshot_name(tmp_path):
    with pytest.raises(ValueError):
        publish_snapshot(
            tmp_path, "../escape", {"value": 1}
        )
