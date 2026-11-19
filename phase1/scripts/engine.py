"""Execute frozen nested fits with immutable-input provenance and resumable outer cases."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT.parent/'phase0_3'
sys.path.insert(0,str(FROZEN/'scripts'))
from pipeline_components import estimator, permute_training_labels, destroy_training_genes, control_seed
DATA={}


def initialize_worker():
    for task,prefix,name in [('T2','primary','FROZEN_PRIMARY_DATASET.tsv'),('T1','T1_sensitivity','FROZEN_T1_SENSITIVITY_DATASET.tsv')]:
        frame=pd.read_csv(FROZEN/'data'/name,sep='\t')
        frame['y']=(frame.label=='tumor').astype(int)
        array=np.load(FROZEN/f'data/{prefix}_inputs.npz',allow_pickle=False)
        DATA[task]={'frame':frame,'rank':array['rank'],'provided':array['provided'],
                    'lookup':dict(zip(frame.record,range(len(frame)))),'genes':array['gene_ids']}


def case_signature(case):
    return hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()


def run_case(case):
    path=ROOT/'cases'/case['id']
    signature=case_signature(case)
    if path.with_suffix('.json').exists():
        old=json.loads(path.with_suffix('.json').read_text())
        assert old['case_signature']==signature
        if old['status']=='COMPLETE':
            assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==old['prediction_sha256']
            assert hashlib.sha256((ROOT/old['model_path']).read_bytes()).hexdigest()==old['model_sha256']
        return old
    if not DATA:
        initialize_worker()
    bundle=DATA[case['task']]
    frame=bundle['frame']
    X=bundle[case['representation']]
    y=LabelEncoder().fit_transform(frame.study) if case['multiclass'] else frame.y.to_numpy()
    train=np.asarray(case['train_indices'],dtype=int)
    test=np.asarray(case['test_indices'],dtype=int)
    inner=np.asarray(case['inner_folds'],dtype=int)
    assert not(set(train)&set(test))
    if case['boundary']!='V0':
        assert not(set(frame.iloc[train].patient)&set(frame.iloc[test].patient))
    if case['boundary']=='V2':
        assert not(set(frame.iloc[train].study)&set(frame.iloc[test].study))
    audit=[]
    timer=time.monotonic()
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def fit(values,labels,C,ratio,scope,corruption_seed):
        model=estimator(C,case['seed'],scaled=case.get('scaled',False),l1_ratio=ratio)
        if case['multiclass']:
            model.set_params(model__solver='lbfgs')
        start=time.monotonic()
        row=dict(case_id=case['id'],analysis=case['analysis'],study=case.get('target','ALL'),boundary=case['boundary'],
                 outer_fold=case['outer_fold'],inner_fold=scope,seed=case['seed'],control=case.get('control',0),
                 control_replicate=case.get('control_replicate',-1),corruption_seed=corruption_seed,
                 hyperparameter=json.dumps({'C':C,'l1_ratio':ratio}),convergence='FAIL',warning='',error='')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model.fit(values,labels)
                row['warning']=' | '.join(str(w.message) for w in caught)
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                    raise RuntimeError('FROZEN_CONVERGENCE_FAILURE')
                row['convergence']='PASS'
                row['iterations']=int(np.max(model.named_steps['model'].n_iter_))
                row['retained_features']=int(model.named_steps['variance'].get_support().sum())
            except Exception as exc:
                row['error']=repr(exc)
                raise
            finally:
                row['elapsed_seconds']=time.monotonic()-start
                audit.append(row)
        return model

    def scope_arrays(indices,inner_id):
        values=X[indices]
        labels=y[indices]
        if case['control']:
            seed=control_seed(case['control'],case['control_replicate'],int(case['boundary'][-1]),case['repeat'],case['outer_fold'],inner_id)
            scope_frame=frame.iloc[indices].reset_index(drop=True)
            if case['control']==1: labels=permute_training_labels(scope_frame,labels,seed)
            else: values=destroy_training_genes(scope_frame,values,seed)
            return values,labels,seed
        return values,labels,None
    try:
        grid=[(C,ratio) for C in case['C_grid'] for ratio in case['ratio_grid']]
        scores=[]
        # Corruption is training-only and fixed per scope, across candidate parameters.
        inner_data={}
        for k in sorted(set(inner)):
            a,b=train[inner!=k],train[inner==k]
            if case['boundary']!='V0':
                assert not(set(frame.iloc[a].patient)&set(frame.iloc[b].patient))
            inner_data[int(k)]=(a,b,scope_arrays(a,int(k)))
        for C,ratio in grid:
            fold_scores=[]
            for k,(a,b,(values,labels,corruption_seed)) in inner_data.items():
                model=fit(values,labels,C,ratio,k,corruption_seed)
                if case['multiclass']:
                    fold_scores.append(balanced_accuracy_score(y[b],model.predict(X[b])))
                else:
                    probability=model.predict_proba(X[b])[:,1]
                    for study in sorted(frame.iloc[b].study.unique()):
                        mask=frame.iloc[b].study.to_numpy()==study
                        fold_scores.append(roc_auc_score(y[b][mask],probability[mask]))
            scores.append(float(np.mean(fold_scores)))
        best=next(i for i,s in enumerate(scores) if s>=max(scores)-1e-12)
        C,ratio=grid[best]
        values,labels,corruption_seed=scope_arrays(train,3)
        model=fit(values,labels,C,ratio,3,corruption_seed)
        probability=model.predict_proba(X[test])
        if not case['multiclass']:
            probability=probability[:,1]
        assert np.isfinite(probability).all()
        support=model.named_steps['variance'].get_support()
        scale=model.named_steps.get('scale')
        model_path=ROOT/'models'/f"{case['id']}.npz"
        np.savez_compressed(model_path,coefficient=model.named_steps['model'].coef_,intercept=model.named_steps['model'].intercept_,
                            stable_gene_ids=bundle['genes'][support],feature_mask=support,
                            training_record_ids=frame.iloc[train].record.to_numpy(dtype=str),
                            scaler_mean=np.array([]) if scale is None else scale.mean_,
                            scaler_scale=np.array([]) if scale is None else scale.scale_)
        np.savez_compressed(path.with_suffix('.npz'),indices=test,probability=probability)
        result=dict(case_id=case['id'],case_signature=signature,status='COMPLETE',C=C,l1_ratio=ratio,
                    inner_scores=scores,code_sha256=code_sha,elapsed_seconds=time.monotonic()-timer,case=case,
                    prediction_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),model_path=str(model_path.relative_to(ROOT)),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    except Exception as exc:
        result=dict(case_id=case['id'],case_signature=signature,status='FAIL',error=repr(exc),code_sha256=code_sha,
                    elapsed_seconds=time.monotonic()-timer,case=case)
    pd.DataFrame(audit).to_csv(path.with_suffix('.fits.tsv'),sep='\t',index=False)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(path.with_suffix('.json'))
    return result
