#!/usr/bin/env python3
"""Reproduce logistic baselines using frozen AEID252 CV folds.
Reads frozen development-fold inputs only. Never opens holdout.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CS=[.001,.01,.1,1.,10.,100.]
def load_csv(path):
    with open(path,newline="") as f: return list(csv.DictReader(f))
def write_csv(path, rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def metrics(y,p):
    return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold-root",default="data/processed/toxcast_aeid252_v1/cv_folds")
    ap.add_argument("--out",default="data/processed/toxcast_aeid252_v1/logistic_v1")
    a=ap.parse_args()
    root=Path(a.fold_root)
    dest=Path(a.out)/"cv_comparison"
    dest.mkdir(parents=True,exist_ok=True)
    folds={}
    for name in ("A","B","D"):
        folds[name]=[]
        for k in range(5):
            base=root/name/f"fold{k}"
            meta=json.loads((base/"manifest.json").read_text())
            with (base/"train.csv").open(newline="") as f:
                tr=np.asarray([[float(x) for x in row] for row in csv.reader(f)])
            with (base/"validation_inputs.csv").open(newline="") as f:
                xv=np.asarray([[float(x) for x in row] for row in csv.reader(f)])
            yt=tr[:,0].astype(int);xt=tr[:,1:]
            yv=np.asarray(meta["validation_labels"],dtype=int)
            ix=np.asarray(meta["validation_indices"],dtype=int)
            assert xt.shape[1]==xv.shape[1] and len(xv)==len(yv)==len(ix)
            assert len(set(ix))==len(ix) and set(yt)==set(yv)=={0,1}
            folds[name].append((k,xt,yt,xv,yv,ix))
        assert sorted(np.concatenate([f[5] for f in folds[name]]).tolist())==list(range(1282))
    # Verify the three feature representations use identical validation partitions and labels.
    for k in range(5):
        assert all(np.array_equal(folds[n][k][5],folds["A"][k][5]) for n in ("B","D"))
        assert all(np.array_equal(folds[n][k][4],folds["A"][k][4]) for n in ("B","D"))
    cv_rows=[];oof=[];fold_rows=[]
    for name in ("A","B","D"):
        scores_by_c={}
        for c in CS:
            preds=[];ys=[];ixs=[];foldaps=[]
            for k,xt,yt,xv,yv,ix in folds[name]:
                # Matches run_models.py: B centered; A and D scaled without centering.
                pipe=make_pipeline(StandardScaler() if name=="B" else StandardScaler(with_mean=False),
                    LogisticRegression(C=c,max_iter=3000,solver="liblinear",random_state=42))
                pipe.fit(xt,yt)
                p=pipe.predict_proba(xv)[:,1]
                preds.append(p);ys.append(yv);ixs.append(ix)
                foldaps.append(average_precision_score(yv,p))
            scores_by_c[c]=(preds,ys,ixs,foldaps)
            cv_rows.append(dict(model=f"{name}_logistic",C=c,mean_fold_AP=float(np.mean(foldaps)),
                                sd_fold_AP=float(np.std(foldaps,ddof=1))))
        best=max(CS,key=lambda c:(np.mean(scores_by_c[c][3]),-c))
        preds,ys,ixs,_=scores_by_c[best]
        for k,(p,y,ix) in enumerate(zip(preds,ys,ixs)):
            fold_rows.append(dict(model=f"{name}_logistic",fold=k,**metrics(y,p),C=best))
            oof.extend(dict(model=f"{name}_logistic",fold=k,train_index=int(i),
                            label=int(t),score=float(s)) for i,t,s in zip(ix,y,p))
        print(f"{name}_logistic selected C={best} by same-fold mean AP")
    results=[]
    for model in sorted({r["model"] for r in oof}):
        rr=[r for r in oof if r["model"]==model]
        assert len(rr)==1282 and len({int(r["train_index"]) for r in rr})==1282
        y=np.asarray([r["label"] for r in rr]);p=np.asarray([r["score"] for r in rr])
        fr=[r for r in fold_rows if r["model"]==model]
        results.append(dict(model=model,pooled_AUC=roc_auc_score(y,p),
            pooled_AP=average_precision_score(y,p),mean_fold_AUC=np.mean([r["auc"] for r in fr]),
            sd_fold_AUC=np.std([r["auc"] for r in fr],ddof=1),
            mean_fold_AP=np.mean([r["ap"] for r in fr]),
            sd_fold_AP=np.std([r["ap"] for r in fr],ddof=1),
            score_min=float(p.min()),score_max=float(p.max())))
    results.sort(key=lambda r:r["mean_fold_AP"],reverse=True)
    write_csv(dest/"model_comparison.csv",results)
    write_csv(dest/"fold_metrics.csv",fold_rows)
    write_csv(dest/"logistic_c_grid.csv",cv_rows)
    write_csv(dest/"oof_predictions.csv",oof)
    (dest/"notes.json").write_text(json.dumps({
        "holdout_evaluated":False,"selection":"mean five-fold AP",
        "caveat":"Logistic C selection and reporting reuse the same CV folds; scores are model-selection estimates, not unbiased nested-CV estimates.",
        "scaling":"B StandardScaler; A/D StandardScaler(with_mean=False), matching original logistic baseline.",
    },indent=2)+"\n")
    print("CV-only comparison:")
    for r in results: print(f'{r["model"]:12s} mean AP={r["mean_fold_AP"]:.4f} pooled AUC={r["pooled_AUC"]:.4f}')
    print("Artifacts:",dest)
if __name__=="__main__": main()
