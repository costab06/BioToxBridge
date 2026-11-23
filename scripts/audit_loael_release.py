#!/usr/bin/env python3
"""Read-only, standard-library audit of frozen LOAEL reproduction artifacts."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/liver_loael_v1/unified_reproduction'
EXPECTED_MASTER_SHA = '9105c38086fe67d9793a909bfa880c943da389996d705e3b14449018fb2aaf2b'
EXPECTED_SPLIT_SHA = '1bef571ed9485afd82fc52feb0049cf8de23b0756e3ac107147d9760bdd80222'
EXPECTED = {
    'A': {'mae': 0.5944975018501282, 'rmse': 0.7810559868812561, 'r2': -0.0200897159123683, 'spearman_rho': 0.466139086513239},
    'B': {'mae': 0.5919240713119507, 'rmse': 0.7832022309303284, 'r2': -0.025703726444378905, 'spearman_rho': 0.04678198764886474},
    'D': {'mae': 0.5854685306549072, 'rmse': 0.7752366662025452, 'r2': -0.004945884229895503, 'spearman_rho': 0.2277470416678265},
}

def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('PASS', message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    provenance = json.loads((OUT / 'provenance.json').read_text())
    execution = json.loads((OUT / 'experiment_execution.json').read_text())
    master = ROOT / provenance['master_path']
    split = ROOT / provenance['split_path']
    check(digest(master) == EXPECTED_MASTER_SHA == provenance['master_sha256'], 'frozen master hash')
    check(digest(split) == EXPECTED_SPLIT_SHA == provenance['split_sha256'], 'frozen split hash')
    check(execution['exit_code'] == 0, 'experiment execution succeeded')
    with master.open(newline='') as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(split.read_text())
    ids = {r['dtxsid'] for r in rows}
    train, test = set(manifest['train_dtxsids']), set(manifest['test_dtxsids'])
    check(len(rows) == len(ids) == 177, '177 unique chemicals')
    check(len(train) == 142 and len(test) == 35 and not train & test and train | test == ids, 'frozen 142/35 partition')
    groups_train = {r['structure_group'] for r in rows if r['dtxsid'] in train}
    groups_test = {r['structure_group'] for r in rows if r['dtxsid'] in test}
    check(not groups_train & groups_test, 'no structure-group leakage')
    tox = [k for k in rows[0] if k.startswith('tox21_')]
    check(len(tox) == 12 and all(math.isfinite(float(r[k])) for r in rows for k in tox), '12 complete Tox21 features')
    predictions = list(csv.DictReader((OUT / 'predictions.csv').open(newline='')))
    check(len(predictions) == 105 and {p['model'] for p in predictions} == {'A','B','D'}, '35 predictions per model')
    check(all({p['dtxsid'] for p in predictions if p['model'] == model} == test for model in EXPECTED), 'predictions cover exact frozen test partition')
    for model, metrics in EXPECTED.items():
        preds = [p for p in predictions if p['model'] == model]
        true = [float(p['y_true']) for p in preds]
        predicted = [float(p['y_pred']) for p in preds]
        n = len(preds)
        mae = sum(abs(a-b) for a,b in zip(true,predicted))/n
        rmse = math.sqrt(sum((a-b)**2 for a,b in zip(true,predicted))/n)
        mean = sum(true)/n
        r2 = 1 - sum((a-b)**2 for a,b in zip(true,predicted))/sum((a-mean)**2 for a in true)
        for metric, observed in [('mae',mae),('rmse',rmse),('r2',r2)]:
            check(math.isclose(observed,metrics[metric],rel_tol=0,abs_tol=1e-5),f'{model} {metric} matches frozen result')
    print('LOAEL release audit completed successfully.')

if __name__ == '__main__':
    main()
