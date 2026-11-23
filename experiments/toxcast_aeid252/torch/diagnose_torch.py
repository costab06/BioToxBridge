#!/usr/bin/env python3
"""Diagnostic-only rerun of inner-split learning curves; no outer holdout access."""
import argparse,csv,json,sys,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from run_torch import ARCH,read_matrix,read_json,model_for,seed_all,get_device,predict,write_csv

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='data/processed/toxcast_aeid252_v1')
    p.add_argument('--mount',default='/opt/neural/data')
    p.add_argument('--out',default='data/processed/toxcast_aeid252_v1/torch_diagnostics_v1')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--models',nargs='+',default=['D_small','D_wide','B_wide'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--min-delta',type=float,default=0.0001)
    p.add_argument('--inner-folds',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--weight-decay',type=float,default=.0001)
    a=p.parse_args()
    for m in a.models:
        if m not in [f'{n}_{v}' for n in ARCH for v in ARCH[n]]:p.error('unknown model '+m)
    with (Path(a.root)/'train.csv').open(newline='') as f:dev=list(csv.DictReader(f))
    if len(dev)!=1282:raise ValueError('Unexpected development set')
    group=np.array([r['structure_group'] for r in dev])
    labels=np.array([int(r['label']) for r in dev])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    device=get_device(a.device)
    rows=[];summary=[]
    for model in a.models:
        name,arch=model.split('_')
        for fold in range(5):
            base=Path(a.mount)/'biotoxbridge_aeid252_nnsd_v1'/'facts'/name/f'fold{fold}'
            meta=read_json(base/'manifest.json')
            ti=np.array(meta['train_indices'],dtype=int)
            vi=np.array(meta['validation_indices'],dtype=int)
            tr=read_matrix(base/'train.csv');x=tr[:,1:];y=tr[:,0].astype(int)
            if not np.array_equal(y,labels[ti]) or set(group[ti])&set(group[vi]):
                raise ValueError('Fold integrity failure')
            inner=StratifiedGroupKFold(n_splits=a.inner_folds,shuffle=True,random_state=a.seed+fold)
            it,iv=next(inner.split(x,y,group[ti]))
            if set(group[ti][it])&set(group[ti][iv]):raise ValueError('Inner group leakage')
            seed_all(a.seed+fold)
            net=model_for(x.shape[1],ARCH[name][arch],a.dropout).to(device)
            opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
            lossfn=nn.BCEWithLogitsLoss()
            xt=torch.from_numpy(x[it]);yt=torch.from_numpy(y[it].astype(np.float32))
            rng=np.random.default_rng(a.seed+fold)
            best=-float('inf');best_epoch=0;bad=0
            for epoch in range(1,a.epochs+1):
                net.train();weighted_loss=0.;n_seen=0
                for ids in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/a.batch_size)))):
                    xx=xt[ids].to(device);yy=yt[ids].to(device)
                    opt.zero_grad(set_to_none=True)
                    logits=net(xx).squeeze(-1);loss=lossfn(logits,yy)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();opt.step()
                    weighted_loss+=loss.item()*len(ids);n_seen+=len(ids)
                val=predict(net,x[iv],device,a.batch_size)
                val_ap=float(average_precision_score(y[iv],val))
                val_auc=float(roc_auc_score(y[iv],val))
                val_bce=float(log_loss(y[iv],np.clip(val,1e-7,1-1e-7),labels=[0,1]))
                improved=val_ap>best+a.min_delta
                if improved:best=val_ap;best_epoch=epoch;bad=0
                else:bad+=1
                rows.append(dict(model=model,fold=fold,epoch=epoch,train_bce=weighted_loss/n_seen,
                    val_bce=val_bce,val_ap=val_ap,val_auc=val_auc,
                    val_min=float(val.min()),val_max=float(val.max()),
                    val_near_zero=float(np.mean(val<=1e-6)),val_near_one=float(np.mean(val>=1-1e-6)),
                    improved=int(improved),patience_counter=bad))
                if bad>=a.patience:break
            summary.append(dict(model=model,fold=fold,best_epoch=best_epoch,
                stopped_epoch=epoch,best_inner_ap=best,inner_train_size=len(it),
                inner_validation_size=len(iv),stop_reason='patience' if bad>=a.patience else 'max_epochs'))
            print(f'{model} fold={fold} best={best_epoch} stop={epoch} best_inner_AP={best:.4f}',flush=True)
    write_csv(out/'learning_curves.csv',rows)
    write_csv(out/'early_stopping_summary.csv',summary)
    (out/'manifest.json').write_text(json.dumps(dict(holdout_access=False,
        purpose='Diagnostic rerun of inner validation only; no model selection or final fit',
        arguments=vars(a)),indent=2)+'\n')
    print('Artifacts:',out)
if __name__=='__main__':main()
