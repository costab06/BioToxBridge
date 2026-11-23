"""Small experiment registry. Scientific work is delegated to existing adapters.

The same manifest/run contract is used for LOAEL and AEID252.
OpenAgentFabric can schedule these deterministic commands as stage handlers.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import yaml

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(config, *, dry_run=False):
    path=Path(config)
    spec=yaml.safe_load(path.read_text())
    if spec.get('schema_version')!=1:raise ValueError('Expected schema_version: 1')
    kind=spec['kind']; name=spec['experiment_id']
    if kind=='loael_regression':
        cmd=[sys.executable,'-m','biotoxbridge.experiments.loael', '--master',spec['master_path'],'--split',spec['split_path'],'--out',spec['output_dir']]
    elif kind=='aeid252_binary':
        cmd=[sys.executable,'experiments/toxcast_aeid252/torch/run_torch.py',
             '--root',spec['development_root'],'--fold-root',spec['fold_root'],
             '--out',spec['output_dir'],'--device',spec.get('device','auto')]
    else:raise ValueError(f'Unknown experiment kind: {kind}')
    manifest={'experiment_id':name,'kind':kind,'schema_version':1,
              'config_sha256':sha256(path),'command':cmd,
              'holdout_policy':spec['holdout_policy'],
              'scientific_question':spec['scientific_question']}
    if dry_run:
        print(json.dumps(manifest,indent=2));return manifest
    output=Path(spec['output_dir']);output.mkdir(parents=True,exist_ok=True)
    manifest['start_utc_epoch']=time.time()
    # Preserve failure status and audit trail; no shell and no dynamic imports.
    result=subprocess.run(cmd,check=False)
    manifest['exit_code']=result.returncode
    manifest['finish_utc_epoch']=time.time()
    (output/'experiment_execution.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();run(a.config,dry_run=a.dry_run)
if __name__=='__main__':main()
