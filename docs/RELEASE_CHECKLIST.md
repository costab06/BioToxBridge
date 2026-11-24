# BioToxBridge public release checklist

The current local reproduction has succeeded; do not interpret this checklist as evidence that the remote repository is already clean.

## Required before publishing

- [ ] Run `python scripts/audit_loael_release.py` and retain its output.
- [x] Unified runner, LOAEL implementation, AEID252 implementation and both experiment YAML files are present in the public source tree.
- [ ] Confirm that the frozen LOAEL master and split manifest, and the AEID252 fold files, are either distributed with permission or available at documented permanent locations with SHA-256 checksums.
- [ ] Check licenses and redistribution permissions for Tox21, ToxCast, ToxRefDB and any derived data.
- [ ] Review `git status`, `.gitignore`, and staged files; exclude local `.venv`, logs, server outputs, API keys, large intermediate data, and private artifacts.
- [ ] Isolate the historical nnsd-dependent `tox21_adapter.py` and related pipeline paths as legacy components, or remove them only after confirming no supported imports depend on them. Do not delete them blindly: existing projectors import `TOX21_SCORE_FIELDS` from that module.
- [x] Root README describes BioToxBridge and both experiments.
- [ ] Run both unified runner commands and tests from a clean environment before tagging.
- [ ] Confirm no tracked secrets using a secret scanner and a review of `git diff --cached`.
- [ ] Describe AEID252 as a development cross-validation result, not an independent holdout success.

## Suggested staging commands

```bash
git status --short
git diff --check
python scripts/audit_loael_release.py
# Review before staging; do not use git add -A without checking artifacts.
git diff --stat
```

The deprecated RDKit Morgan fingerprint warnings are non-fatal. Modernizing the API can be handled after release, provided fingerprint equivalence is verified.
