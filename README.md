# BioToxBridge

**Testing whether predicted bioactivity adds useful information to molecular structure representations.**

BioToxBridge contains two chemical-toxicity experiments and a shared experiment execution contract. The experiments deliberately include both a negative result and a positive development-set result. This is a reproducibility-oriented research demonstration, **not** a validated toxicity prediction product.

## Scientific questions

| Experiment | Outcome | Interpretation |
|---|---|---|
| Liver LOAEL regression | No demonstrated meaningful improvement | The combined fingerprint + Tox21 representation slightly reduces MAE and RMSE, but test R² remains negative and rank correlation worsens relative to fingerprints alone. |
| ToxCast AEID252 binary assay activity | Positive development cross-validation result | Combined features (D_small) reached mean fold average precision **0.6638**, versus **0.6428** for the Tox21-only logistic baseline. This modest gain is **not independent holdout confirmation**. |

The two tasks have different targets, model families, and evaluation procedures. Their metrics must not be compared directly.

## Representations

- **A:** 2048-bit radius-2 Morgan fingerprints derived from molecular SMILES.
- **B:** 12 continuous pretrained Tox21 bioactivity prediction scores.
- **D:** concatenation of A and B.

The Tox21 values are **predicted features**, not observed assay measurements for each study chemical. Their provenance and potential dependencies must be considered when interpreting results.

## Installation

Requires Python 3.10 or newer. From a fresh checkout:

```bash
git clone https://github.com/costab06/BioToxBridge.git
cd BioToxBridge
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,torch]'
python -m pytest -m 'not integration'
```

Read the [published article](https://medium.com/@costab06/biotoxbridge-building-a-scientific-experimentation-framework-on-openagentfabric-7247bb6289ea) for the scientific motivation and results.

## Run the experiments

From the repository root, in an environment with the project's dependencies installed:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m biotoxbridge.experiments.runner config/experiments/liver_loael_unified.yaml --dry-run
python -m biotoxbridge.experiments.runner config/experiments/liver_loael_unified.yaml
python scripts/audit_loael_release.py
```

The LOAEL configuration reads the **existing frozen** master dataset at `runs/equivalence-direct-004/processed/liver_master.csv` and frozen split at `runs/equivalence-direct-004/splits/split_manifest.json`. It does **not** regenerate Tox21 predictions or require the historical inference server. The original split is reused rather than creating a new independent test set. The runner writes to `runs/liver_loael_v1/unified_reproduction`.

AEID252 uses the unified runner with `config/experiments/aeid252_unified.yaml` and frozen cross-validation fold files. Check the paths in that configuration before execution. Its development CV results are not an untouched holdout test.

## Reproducibility and data availability

The frozen experimental inputs are not currently distributed in this repository. A clone alone cannot reproduce the published metrics. See [reproducibility requirements](docs/REPRODUCIBILITY.md) for the required paths and outstanding data distribution work.

The LOAEL reproduction produces `metrics.json`, `predictions.csv`, `provenance.json`, and `experiment_execution.json`. The provenance includes SHA-256 hashes of the frozen master and split manifest. The audit script checks those hashes, record counts, split integrity, and the reported regression metrics.

## Architecture: scientific framework and execution infrastructure

BioToxBridge is the **scientific framework**. It owns chemical identity and structure processing, frozen datasets, feature representations, split and fold definitions, modeling, metrics, and experiment reporting. These components are independently useful and should remain usable without OpenAgentFabric.

[OpenAgentFabric](https://github.com/costab06/OpenAgentFabric) (OAF) is an **optional execution infrastructure** for orchestrating scientific work. Its responsibilities include stage execution, durable state, checkpointing, error handling, and observability. The BioToxBridge OAF adapters are an integration boundary; scientific modeling is not part of OAF.

The current unified command-line runner provides a common experiment configuration and provenance contract, dispatching to the LOAEL regression and AEID252 classification implementations. It does **not** imply that both experiments are executed as OAF-native stage graphs, and **OAF is not required to run the unified experiments**.

## Historical components

Earlier development used a separate neural-network inference service for Tox21 feature generation. The frozen LOAEL reproduction does not use that service. Historical adapters remain in the source tree for provenance and are not part of the supported frozen-data reproduction path.

## Research limitations

The LOAEL test set contains only 35 chemicals. The AEID252 reported improvement is from development cross-validation. Neither experiment establishes prospective clinical, regulatory, or deployment validity. These results support conclusions about these specific experiments, not broad claims about Tox21 features or chemical toxicity prediction.
