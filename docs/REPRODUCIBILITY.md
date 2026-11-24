# Reproducing the published experiments

The public source release does not yet distribute the frozen experimental inputs. Do not substitute newly generated splits or folds and describe the resulting metrics as exact reproduction.

## Required inputs

| Experiment | Required location (relative to repository root) |
|---|---|
| Liver LOAEL | `runs/equivalence-direct-004/processed/liver_master.csv` |
| Liver LOAEL | `runs/equivalence-direct-004/splits/split_manifest.json` |
| AEID252 | `data/processed/toxcast_aeid252_v1/` development inputs |
| AEID252 | `data/processed/toxcast_aeid252_v1/cv_folds/` frozen folds |

Distribution locations, exact AEID252 input inventory, SHA-256 checksums, and redistribution permissions remain to be established. The Apache-2.0 code license does not automatically cover third-party datasets or model artifacts. Frozen Tox21 predictions are not observed assay measurements.

## Execution

Install from the repository root with `python -m pip install -e '.[test,torch]'`. Inspect the two configurations in `config/experiments/` before running.

```bash
python -m biotoxbridge.experiments.runner config/experiments/liver_loael_unified.yaml --dry-run
python -m biotoxbridge.experiments.runner config/experiments/aeid252_unified.yaml --dry-run
```

Dry runs describe commands; they do not validate input availability. After obtaining the exact frozen inputs, repeat without `--dry-run` and run `python scripts/audit_loael_release.py`. Preserve execution manifests and metrics. AEID252 reports development cross-validation, not independent holdout validation. OAF is optional for these commands.
