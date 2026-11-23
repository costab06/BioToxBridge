import pytest

from biotoxbridge.oaf.lifecycle import Lifecycle, LifecycleStep
from biotoxbridge.oaf.esol_lifecycle_orchestration import next_request
from biotoxbridge.oaf.coordinator import (
    STAGES,
    TOXICOLOGY_LIFECYCLE,
)


def test_esol_acquire_transition():
    assert next_request(
        "esol_acquire",
        {"sha256": "abc123"},
        {"seed": 42},
    ) == (
        "esol_train",
        {"dataset_sha256": "abc123", "seed": 42},
    )


def test_esol_train_transition():
    assert next_request(
        "esol_train",
        {"candidate_version": "v1"},
        {"policy": {"max_mae": 1.0}},
    ) == (
        "esol_promote",
        {"candidate_version": "v1", "max_mae": 1.0},
    )


def test_terminal_stage():
    assert next_request(
        "esol_promote", {}, {}
    ) == (None, None)


def test_unknown_stage():
    with pytest.raises(ValueError):
        next_request("unknown", {}, {})


def test_duplicate_stage_rejected():
    with pytest.raises(ValueError):
        Lifecycle((
            LifecycleStep("a", None),
            LifecycleStep("a", None),
        ))


def test_missing_transition_rejected():
    with pytest.raises(ValueError):
        Lifecycle((
            LifecycleStep("a", "b"),
            LifecycleStep("b", None),
        ))


def test_unknown_destination_rejected():
    with pytest.raises(ValueError):
        Lifecycle((
            LifecycleStep("a", "missing", lambda output, config: {}),
        ))


def test_terminal_builder_rejected():
    with pytest.raises(ValueError):
        Lifecycle((
            LifecycleStep("a", None, lambda output, config: {}),
        ))

def test_toxicology_lifecycle_stage_order():
    assert TOXICOLOGY_LIFECYCLE.stages == STAGES


def test_toxicology_lifecycle_transitions():
    for index, stage in enumerate(STAGES[:-1]):
        assert TOXICOLOGY_LIFECYCLE.next_stage(stage) == STAGES[index + 1]


def test_toxicology_lifecycle_terminal():
    assert TOXICOLOGY_LIFECYCLE.next_stage(STAGES[-1]) is None


def test_toxicology_lifecycle_unknown_stage():
    import pytest

    with pytest.raises(ValueError):
        TOXICOLOGY_LIFECYCLE.next_stage("unknown")