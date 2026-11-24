# PyTorch inner-validation learning-curve diagnostics

This **does not** select or train a final model, and does not access the frozen holdout.

Both scripts are included in `experiments/toxcast_aeid252/torch/`. Install the PyTorch extra and obtain the frozen inputs before running.

Run from BioToxBridge root after setting `DATA_ROOT` to the absolute path of your local data mount:

```bash
PYTHONPATH=src python -u experiments/toxcast_aeid252/torch/diagnose_torch.py --mount "$DATA_ROOT" --device auto | tee torch_diagnostics.log
```

It repeats the exact original inner group-aware splits, seeds, network structures, optimizer, and training loop for D_small, D_wide, and B_wide, recording per-epoch training BCE, inner-validation BCE/AP/AUC, output ranges, and the stopping decisions. It deliberately **does not** use the outer validation folds for training or epoch selection.

Outputs under `data/processed/toxcast_aeid252_v1/torch_diagnostics_v1/`:

- `learning_curves.csv`: each epoch's metrics and whether it improved the early-stopping criterion
- `early_stopping_summary.csv`: selected epoch, actual stopping epoch, inner AP, stop reason
- `manifest.json`: diagnostic parameters and holdout-access flag

Note: Existing PyTorch `fit.json` files store best epoch and AP but not per-epoch learning curves. This is a reproducible rerun, not a reconstruction of historical loss values. No checkpoint or OOF predictions are overwritten.
