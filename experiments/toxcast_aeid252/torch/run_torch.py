#!/usr/bin/env python3
"""AEID252 PyTorch group-aware CV, no holdout access.

Consumes prepared, frozen fold facts and manifests.
Train-only inner group split chooses epoch; outer folds estimate performance.
"""
import argparse, csv, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ARCH = {'A':{'small':(128,32),'wide':(256,64)},
        'B':{'small':(24,8),'wide':(48,16)},
        'D':{'small':(128,32),'wide':(256,64)}}

def read_matrix(p):
    with open(p,newline='') as f:
        a=np.asarray([[float(v) for v in row] for row in csv.reader(f)],dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError(f'Bad facts: {p}')
    return a

def read_json(p):return json.loads(p.read_text())
def write_json(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def model_for(dim,hidden,dropout):
    parts=[];last=dim
    for h in hidden:
        parts.extend([nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)])
        last=h
    parts.append(nn.Linear(last,1)) # logits; NO output sigmoid in model
    return nn.Sequential(*parts)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def get_device(choice):
    if choice=='auto':choice='cuda' if torch.cuda.is_available() else 'cpu'
    if choice=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return torch.device(choice)

def predict(m,x,device,batch):
    m.eval();parts=[]
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            t=torch.from_numpy(x[i:i+batch]).to(device)
            parts.append(torch.sigmoid(m(t).squeeze(-1)).cpu().numpy())
    p=np.concatenate(parts).astype(float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    return p

def fit(x,y,hidden,device,a,seed,inner_train=None,inner_val=None,epochs=None):
    seed_all(seed)
    net=model_for(x.shape[1],hidden,a.dropout).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    lossfn=nn.BCEWithLogitsLoss()
    if inner_train is None:inner_train=np.arange(len(y))
    xt=torch.from_numpy(x[inner_train]);yt=torch.from_numpy(y[inner_train].astype(np.float32))
    rng=np.random.default_rng(seed)
    best=-np.inf;best_state=None;best_epoch=0;bad=0
    limit=epochs or a.epochs
    for epoch in range(1,limit+1):
        net.train()
        for idx in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/a.batch_size)))):
            xx=xt[idx].to(device);yy=yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss=lossfn(net(xx).squeeze(-1),yy)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();opt.step()
        if inner_val is not None:
            p=predict(net,x[inner_val],device,a.batch_size)
            score=float(average_precision_score(y[inner_val],p))
            if score>best+a.min_delta:
                best=score;best_epoch=epoch;bad=0
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:bad+=1
            if bad>=a.patience:break
    if best_state is not None:net.load_state_dict(best_state)
    return net, (best_epoch if inner_val is not None else limit),best

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/processed/toxcast_aeid252_v1')
    p.add_argument('--fold-root',default='data/processed/toxcast_aeid252_v1/cv_folds')
    p.add_argument('--mount',default='/opt/neural/data',help='Legacy prepared-fold location for one-time migration')
    p.add_argument('--out',default='data/processed/toxcast_aeid252_v1/torch_nonlinear_v1')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--min-delta',type=float,default=0.0001)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--dropout',type=float,default=0.2)
    p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--inner-folds',type=int,default=4)
    p.add_argument('--only',choices=['all','A_small','A_wide','B_small','B_wide','D_small','D_wide'],default='all')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if not (0<=a.dropout<1) or a.epochs<1 or a.inner_folds<2:raise ValueError('Invalid hyperparameters')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    jobs=[{'id':f'{name}_{arch}_fold{fold}','name':name,'architecture':arch,'fold':fold}
          for name in ('A','B','D') for arch in ('small','wide') for fold in range(5)]
    # Read development labels/groups only; never open test.csv or final/holdout_inputs.csv.
    with (Path(a.root)/'train.csv').open(newline='') as f:dev=list(csv.DictReader(f))
    assert len(dev)==1282
    group=np.asarray([r['structure_group'] for r in dev]);labels=np.asarray([int(r['label']) for r in dev])
    if set(labels)!={0,1}:raise ValueError('Nonbinary development labels')
    device=get_device(a.device)
    print('Device:',device,'Torch:',torch.__version__,flush=True)
    print('Holdout access: disabled; outer folds from frozen manifests',flush=True)
    oof=[];fold_metrics=[];diagnostics=[]
    for job in jobs:
        name=job['name'];arch=job['architecture'];fold=int(job['fold']);key=f'{name}_{arch}'
        if a.only!='all' and key!=a.only:continue
        base=Path(a.fold_root)/name/f'fold{fold}'
        meta=read_json(base/'manifest.json')
        ti=np.asarray(meta['train_indices'],dtype=int);vi=np.asarray(meta['validation_indices'],dtype=int)
        tr=read_matrix(base/'train.csv');x=tr[:,1:];y=tr[:,0].astype(int)
        xv=read_matrix(base/'validation_inputs.csv');yv=np.asarray(meta['validation_labels'],dtype=int)
        assert np.array_equal(y,labels[ti]) and np.array_equal(yv,labels[vi])
        assert len(x)==len(ti) and len(xv)==len(vi) and x.shape[1]==xv.shape[1]
        assert not set(group[ti])&set(group[vi])
        # Prepared B/D features already use fold-training-only standardization.
        # A fingerprint bits stay binary; no additional min/max scaling.
        inner=StratifiedGroupKFold(n_splits=a.inner_folds,shuffle=True,random_state=a.seed+fold)
        inner_tr,inner_va=next(inner.split(x,y,group[ti]))
        assert set(y[inner_tr])=={0,1} and set(y[inner_va])=={0,1}
        assert not set(group[ti][inner_tr])&set(group[ti][inner_va])
        dest=out/'models'/job['id'];dest.mkdir(parents=True,exist_ok=True)
        predfile=dest/'validation_predictions.csv';infofile=dest/'fit.json'
        if predfile.exists() and not a.overwrite:
            with predfile.open(newline='') as f:rr=list(csv.DictReader(f))
            if len(rr)!=len(vi) or any(int(r['train_index'])!=int(i) or int(r['label'])!=int(t) for r,i,t in zip(rr,vi,yv)):
                raise ValueError(f'Stale cached predictions: {predfile}; use --overwrite')
            prob=np.asarray([float(r['score']) for r in rr]);info=read_json(infofile)
        else:
            net,epoch,inner_ap=fit(x,y,ARCH[name][arch],device,a,a.seed+fold,inner_tr,inner_va)
            # Refit on entire OUTER training fold for selected number of epochs;
            # inner validation is never used to train the early-stopped model.
            net,_,_=fit(x,y,ARCH[name][arch],device,a,a.seed+fold+1000,epochs=epoch)
            prob=predict(net,xv,device,a.batch_size)
            torch.save({'state_dict':{k:v.cpu() for k,v in net.state_dict().items()},
                        'input_dim':x.shape[1],'hidden':ARCH[name][arch],
                        'dropout':a.dropout,'epoch':epoch},dest/'model.pt')
            write_csv(predfile,[{'train_index':int(i),'label':int(t),'score':float(s)} for i,t,s in zip(vi,yv,prob)])
            info={'job':job['id'],'best_epoch':epoch,'inner_ap':inner_ap,'inner_train_size':len(inner_tr),
                  'inner_val_size':len(inner_va),'seed':a.seed+fold,'device':str(device)}
            write_json(infofile,info)
        if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise ValueError('Bad prediction')
        met={'model':key,'fold':fold,'n':len(vi),'auc':float(roc_auc_score(yv,prob)),
             'ap':float(average_precision_score(yv,prob)),'best_epoch':info['best_epoch']}
        fold_metrics.append(met)
        oof.extend({'model':key,'fold':fold,'train_index':int(i),'label':int(t),'score':float(s)}
                   for i,t,s in zip(vi,yv,prob))
        diagnostics.append({'model':key,'fold':fold,'unique_scores':len(np.unique(prob)),
                            'min':float(prob.min()),'max':float(prob.max()),
                            'near_zero_fraction':float(np.mean(prob<=1e-6)),
                            'near_one_fraction':float(np.mean(prob>=1-1e-6))})
        print(f'{job["id"]}: AP={met["ap"]:.4f} AUC={met["auc"]:.4f} epoch={met["best_epoch"]}',flush=True)
    summary=[]
    for key in sorted({r['model'] for r in oof}):
        rr=[r for r in oof if r['model']==key]
        if len(rr)!=len(dev) or sorted(r['train_index'] for r in rr)!=list(range(len(dev))):
            raise ValueError(f'{key}: CV partition incomplete')
        y=np.asarray([r['label'] for r in rr]);s=np.asarray([r['score'] for r in rr]);ff=[r for r in fold_metrics if r['model']==key]
        summary.append({'model':key,'pooled_AUC':float(roc_auc_score(y,s)),
                        'pooled_AP':float(average_precision_score(y,s)),
                        'mean_fold_AUC':float(np.mean([r['auc'] for r in ff])),
                        'sd_fold_AUC':float(np.std([r['auc'] for r in ff],ddof=1)),
                        'mean_fold_AP':float(np.mean([r['ap'] for r in ff])),
                        'sd_fold_AP':float(np.std([r['ap'] for r in ff],ddof=1)),
                        'unique_scores':len(np.unique(s))})
    summary.sort(key=lambda r:r['mean_fold_AP'],reverse=True)
    write_csv(out/'model_comparison.csv',summary);write_csv(out/'fold_metrics.csv',fold_metrics)
    write_csv(out/'oof_predictions.csv',oof);write_csv(out/'score_diagnostics.csv',diagnostics)
    write_json(out/'run_manifest.json',{'engine':'pytorch','holdout_evaluated':False,
      'selection_metric':'mean_fold_AP','parameters':vars(a),
      'notes':'Early stopping uses inner group-disjoint split; refit on full outer train fold. CV comparison only.'})
    print('CV comparison (holdout NOT evaluated):',flush=True)
    for r in summary:print(f'{r["model"]:<9} AP={r["mean_fold_AP"]:.4f} pooled_AUC={r["pooled_AUC"]:.4f}',flush=True)
    print('Artifacts:',out,flush=True)
if __name__=='__main__':main()
