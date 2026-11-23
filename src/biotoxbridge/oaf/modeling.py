"""Existing BioToxBridge model projection/training/evaluation under JSON OAF contracts."""
from __future__ import annotations

import csv
import json
import math
import io
from pathlib import Path

MODEL_IDS = {"structure_only": "A", "predicted_bioactivity": "B",
             "structure_plus_predicted": "D"}


def finite_json(value):
    """Make numpy scalars and undefined statistics compatible with immutable JSON."""
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def model_ids(config):
    requested = config.get('models', list(MODEL_IDS))
    ids = [MODEL_IDS.get(x, x) for x in requested]
    if not ids or len(ids) != len(set(ids)) or any(x not in ('A', 'B', 'D') for x in ids):
        raise ValueError('Expected unique supported models A/B/D')
    return ids


def train_models(value, config):
    """Project and train using existing functions; return JSON predictions in original units."""
    from biotoxbridge.datasets.projectors import project_A, project_B, project_D
    from biotoxbridge.training.trainer import train_model

    train, test = value['train'], value['test']
    if not train or not test:
        raise ValueError('Both frozen partitions must be nonempty')
    fp = config['chemistry']['fingerprint']
    projectors = {'A': lambda: project_A(train, test, radius=fp['radius'], n_bits=fp['bits']),
                  'B': lambda: project_B(train, test),
                  'D': lambda: project_D(train, test, radius=fp['radius'], n_bits=fp['bits'])}
    results = {}
    for mid in model_ids(config):
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te, meta = projectors[mid]()
        fitted, info = train_model(X_tr, y_tr, model_id=mid)
        predicted = fitted.predict(X_te)
        scaler = meta['y_scaler']
        y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(predicted.reshape(-1, 1)).ravel()
        if len(ids_te) != len(test) or len(ids_tr) != len(train):
            raise ValueError('Projector changed frozen partition size')
        results[mid] = {'training': finite_json(info),
                        'predictions': [{'dtxsid': str(sid), 'y_true': float(y),
                                         'y_pred': float(p), 'residual': float(y-p)}
                                        for sid, y, p in zip(ids_te, y_true, y_pred)]}
    return {'models': finite_json(results), 'train_count': len(train),
            'test_count': len(test), 'excluded_count': len(value.get('excluded', []))}


def evaluate_models(value, config):
    import numpy as np
    from biotoxbridge.evaluation.evaluator import evaluate, bootstrap_ci, compare_models

    metrics = {}
    for mid, model in value['models'].items():
        predictions = model['predictions']
        y_true = np.asarray([r['y_true'] for r in predictions], dtype=float)
        y_pred = np.asarray([r['y_pred'] for r in predictions], dtype=float)
        result = evaluate(y_true, y_pred, label=f'Model {mid}')
        ci = bootstrap_ci(y_true, y_pred,
                          n_iter=config['evaluation']['bootstrap_iterations'],
                          seed=config['evaluation']['bootstrap_seed'])
        metrics[mid] = finite_json({**result, 'ci': ci})
    plain = {mid: {k: v for k, v in entry.items() if k != 'ci'}
             for mid, entry in metrics.items()}
    return {**value, 'metrics': metrics,
            'comparison': finite_json(compare_models(plain))}


def _publish_immutable(path: Path, content: bytes) -> None:
    """Publish once; accept identical replay, reject conflicting content."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("xb") as f:
            f.write(content)
            f.flush()
            import os
            os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Report artifact conflict: {path}")


def report_models(value, run_dir):
    """Persist immutable human-readable scientific outputs."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_dir / "metrics.json"
    predictions = run_dir / "predictions.csv"

    metrics_content = (
        json.dumps(
            value["metrics"],
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "dtxsid", "model", "y_true", "y_pred", "residual"
        ),
    )
    writer.writeheader()

    for mid, result in value["models"].items():
        for row in result["predictions"]:
            writer.writerow({"model": mid, **row})

    predictions_content = buffer.getvalue().encode("utf-8")

    # Validate existing files before publishing either output.
    for path, content in (
        (metrics, metrics_content),
        (predictions, predictions_content),
    ):
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Report artifact conflict: {path}")

    _publish_immutable(metrics, metrics_content)
    _publish_immutable(predictions, predictions_content)

    return {
        "train_count": value["train_count"],
        "test_count": value["test_count"],
        "excluded_count": value["excluded_count"],
        "metrics": value["metrics"],
        "comparison": value["comparison"],
        "report_files": {
            "metrics": str(metrics),
            "predictions": str(predictions),
        },
    }