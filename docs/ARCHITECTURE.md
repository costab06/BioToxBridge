# Architecture and responsibilities

| Responsibility | BioToxBridge scientific framework | OpenAgentFabric (optional) |
|---|---|---|
| Chemical identification, SMILES, structure groups | Owns | Does not own |
| Frozen datasets, feature projections, A/B/D representations | Owns | Does not own |
| Model training, evaluation, metrics | Owns | Does not own |
| Experiment YAML and scientific provenance | Owns | May record execution provenance |
| Durable orchestration, retries, checkpoints | Does not require | Provides when integrated |
| Error handling, operational observability | Application-level validation | Runtime-level facilities |

## Execution paths

1. **Standalone scientific reproduction:** the unified runner dispatches the appropriate LOAEL or AEID252 implementation using frozen data/folds and writes an execution manifest. OAF is not required.
2. **OAF-assisted execution:** existing adapters may wrap scientific operations with OAF runtime behavior without moving scientific logic into OAF.

The runner's common interface is an experiment contract, **not** a common neural-network implementation or a claim of one OAF-native graph. LOAEL uses regression and a frozen train/test split; AEID252 uses classification and development cross-validation. Their results are not directly comparable.
