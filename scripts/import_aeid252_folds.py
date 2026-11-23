#!/usr/bin/env python3
"""One-time import of prepared CV folds into engine-neutral directory."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',default='/opt/neural/data/biotoxbridge_aeid252_nnsd_v1/facts')
p.add_argument('--dest',default='data/processed/toxcast_aeid252_v1/cv_folds')
a=p.parse_args()
src=Path(a.source);dst=Path(a.dest)
if not src.is_dir():p.error(f'Prepared folds missing: {src}')
manifest={}
for feature in ('A','B','D'):
 for k in range(5):
  for name in ('train.csv','validation_inputs.csv','manifest.json'):
   rel=Path(feature)/f'fold{k}'/name
   data=(src/rel).read_bytes()
   target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.read_bytes()!=data:
    raise ValueError(f'Refusing to overwrite changed fold: {target}')
   if not target.exists():target.write_bytes(data)
   manifest[str(rel)]=hashlib.sha256(data).hexdigest()
(dst/'import_manifest.json').write_text(json.dumps({'sha256':manifest},indent=2)+'\n')
print(f'Imported {len(manifest)} immutable fold files into {dst}')
