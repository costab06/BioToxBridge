# BioToxBridge v0.1.0 release checklist

## Completed validation

- [x] Public unified runner and both experiment configurations.
- [x] Clean public checkout: 181 standalone tests passed.
- [x] All 24 published metrics reproduced to four decimal places, including standalone logistic baselines.
- [x] 18 LOAEL release audit checks passed.
- [x] 50 frozen input integrity checks passed in the original data workspace.
- [x] Frozen input archive and SHA-256 checksum prepared.
- [x] Validated dependency inventory prepared.
- [x] Live OAF v0.2.2 / Redis integration: 13 submission tests and one opt-in SIGKILL recovery test passed.
- [x] Git history secret scan passed after documenting four individually reviewed false positives.
- [x] README and reproduction documentation updated.
- [x] Historical nnsd adapters documented as outside the supported frozen-data reproduction path.
- [x] AEID252 described as development CV, not independent holdout validation.

## Publication

- [ ] Final PR CI and merge.
- [ ] Tag and publish v0.1.0.
- [ ] Publish permitted release assets and update availability instructions.

Dataset redistribution permission review was deferred at the maintainer's direction; permissions have not been verified. The code license does not automatically cover third-party data. Frozen input assets remain unpublished pending a distribution decision. The standalone reproduction uses frozen predicted Tox21 features rather than regenerating original model weights. OAF integration validation covers the ESOL lifecycle, not OAF-native toxicity experiment execution.

Nonfatal RDKit and pytest-asyncio warnings remain; modernization is outside the initial release scope.
