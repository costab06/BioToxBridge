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

## Validated reproduction record

A fresh public checkout at commit `400e565`, followed by the standalone logistic runner at `b4b4cac`, reproduced all 24 published metrics to four decimal places. The standalone suite reported 181 passed, 1 skipped and 1 deselected. The LOAEL audit passed all 18 checks. Fifty frozen input integrity checks passed in the original data workspace.

The validated environment used Python 3.12.13, an NVIDIA GeForce RTX 4060, driver 580.82.07 and PyTorch 2.15.0+cu132. Clear inherited PYTHONPATH before installing and running standalone experiments. The planned release assets include the validated dependency inventory and frozen input archive; availability remains pending publication.

Run the additional logistic baseline from the repository root:

```bash
python experiments/toxcast_aeid252/logistic/run_logistic.py
```

| Representation | Logistic mean fold AP | Logistic pooled AUC | Small NN mean fold AP | Small NN pooled AUC |
|---|---:|---:|---:|---:|
| A | 0.5766 | 0.6365 | 0.5876 | 0.6605 |
| B | 0.6428 | 0.7217 | 0.6327 | 0.6957 |
| D | 0.6163 | 0.6791 | 0.6638 | 0.7303 |

Logistic C selection and reporting reuse the same five development folds; these are model-selection estimates, not unbiased nested-CV estimates. Frozen predicted Tox21 features are reused, not regenerated from original model weights.

## Optional OAF integration validation

In a separate environment with OpenAgentFabric v0.2.2 (commit `ec73d1a`), Redis 7 on localhost port 6381, and OAF dependencies installed, all 13 submission tests and the opt-in ESOL crash-recovery test passed. The latter completed in 64.40 seconds and verified SIGKILL recovery, immutable acquisition artifacts, durable OAF records, lifecycle completion and zero pending messages.

```bash
python -m pytest -v tests/test_esol_submission.py
BIOTOXBRIDGE_RUN_CRASH_TEST=1 python -m pytest -v -s tests/test_esol_crash_recovery_integration.py
```

These tests require OAF framework on PYTHONPATH and access to the ESOL dataset. They validate the ESOL execution lifecycle, not OAF-native execution of the toxicity experiments. Use an isolated test Redis instance where possible. No OAF source changes were required.
