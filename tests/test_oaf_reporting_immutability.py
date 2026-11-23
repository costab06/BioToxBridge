
import copy
import pytest

from biotoxbridge.oaf.modeling import report_models


def sample_result():
    return {
        "train_count": 10,
        "test_count": 2,
        "excluded_count": 0,
        "metrics": {"A": {"mae": 0.5}},
        "comparison": {},
        "models": {
            "A": {
                "predictions": [
                    {
                        "dtxsid": "D1",
                        "y_true": 1.0,
                        "y_pred": 1.5,
                        "residual": -0.5,
                    }
                ]
            }
        },
    }


def test_report_replay_is_identical(tmp_path):
    value = sample_result()

    first = report_models(value, tmp_path)
    second = report_models(value, tmp_path)

    assert first == second
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions.csv").is_file()


def test_changed_metrics_rejected(tmp_path):
    value = sample_result()
    report_models(value, tmp_path)

    changed = copy.deepcopy(value)
    changed["metrics"]["A"]["mae"] = 0.9

    with pytest.raises(ValueError, match="conflict"):
        report_models(changed, tmp_path)


def test_changed_predictions_rejected(tmp_path):
    value = sample_result()
    report_models(value, tmp_path)

    changed = copy.deepcopy(value)
    changed["models"]["A"]["predictions"][0]["y_pred"] = 2.0

    with pytest.raises(ValueError, match="conflict"):
        report_models(changed, tmp_path)


def test_existing_report_not_modified_on_conflict(tmp_path):
    value = sample_result()
    report_models(value, tmp_path)

    metrics_before = (tmp_path / "metrics.json").read_bytes()
    predictions_before = (tmp_path / "predictions.csv").read_bytes()

    changed = copy.deepcopy(value)
    changed["models"]["A"]["predictions"][0]["y_pred"] = 9.0

    with pytest.raises(ValueError):
        report_models(changed, tmp_path)

    assert (tmp_path / "metrics.json").read_bytes() == metrics_before
    assert (tmp_path / "predictions.csv").read_bytes() == predictions_before
