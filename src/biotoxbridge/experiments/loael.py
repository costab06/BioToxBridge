"""Run frozen LOAEL A/B/D comparison without external Tox21 scoring."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def run(master_path,split_path,out_path):
    from biotoxbridge.splitting.splitter import load_split
    from biotoxbridge.datasets.projectors import project_A,project_B,project_D
    from biotoxbridge.training.trainer import train_model
    from biotoxbridge.evaluation.evaluator import evaluate,bootstrap_ci,compare_models
    master_path=Path(master_path);split_path=Path(split_path);out_path=Path(out_path)
    with master_path.open(newline='') as f: records=list(csv.DictReader(f))
    expected=['tox21_'+ep for ep in ('NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53')]
    if len(records)!=177 or len({r['dtxsid'] for r in records})!=len(records):
        raise ValueError('Expected 177 distinct frozen LOAEL chemicals')
    for r in records:
        if not r.get('canonical_smiles') or not r.get('structure_group') or not r.get('log10_loael_median'):
            raise ValueError('Missing LOAEL target/structure/group')
        for field in expected:
            if field not in r or not math.isfinite(float(r[field])):
                raise ValueError(f'Missing or non-finite {field}')
    train,test=load_split(records,split_path)
    metrics={};predictions=[]
    for name,project in [('A',project_A),('B',project_B),('D',project_D)]:
        Xtr,Xte,ytr,yte,ids_tr,ids_te,meta=project(train,test)
        fitted,info=train_model(Xtr,ytr,model_id=name)
        yhat=fitted.predict(Xte)
        scaler=meta['y_scaler']
        result=evaluate(yte,yhat,label='Model '+name,y_scaler=scaler)
        result['ci']=bootstrap_ci(yte,yhat,n_iter=1000,seed=42,y_scaler=scaler)
        result['training']=info
        metrics[name]=result
        ytrue=scaler.inverse_transform(yte.reshape(-1,1)).ravel()
        ypred=scaler.inverse_transform(yhat.reshape(-1,1)).ravel()
        for sid,yt,yp in zip(ids_te,ytrue,ypred):
            predictions.append({'dtxsid':sid,'model':name,'y_true':float(yt),'y_pred':float(yp),'residual':float(yt-yp)})
    out_path.mkdir(parents=True,exist_ok=True)
    (out_path/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    with (out_path/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dtxsid','model','y_true','y_pred','residual']);writer.writeheader();writer.writerows(predictions)
    manifest={'experiment_id':'liver_loael_v1','master_path':str(master_path),'master_sha256':hashlib.sha256(master_path.read_bytes()).hexdigest(),'split_path':str(split_path),'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'n_train':len(train),'n_test':len(test),'models':['A','B','D'],'tox21_feature_count':12}
    (out_path/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'provenance':manifest,'comparison':compare_models(metrics)},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.master,a.split,a.out)
if __name__=='__main__':main()
