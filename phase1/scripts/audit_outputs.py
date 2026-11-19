"""Recompute scope, prediction, model and numerical evidence after all fits."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.special import expit,softmax
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT.parent/'phase0_3'


def main():
    inputs={}
    for task,prefix in [('T2','primary'),('T1','T1_sensitivity')]:
        arrays=np.load(FROZEN/f'data/{prefix}_inputs.npz',allow_pickle=False)
        inputs[task]={name:arrays[name] for name in ['rank','provided']}
    replay_errors=[]
    fit_tables=[]
    records=[]
    provenance=[]
    feature_sha=hashlib.sha256((FROZEN/'platform/COMMON_GENE_UNIVERSE.tsv').read_bytes()).hexdigest()
    engine_sha=hashlib.sha256((ROOT/'protocol/EXECUTED_CODE_SNAPSHOT/engine.py').read_bytes()).hexdigest()
    for path in sorted((ROOT/'cases').glob('*.json')):
        if path.name.startswith('._'):continue
        info=json.loads(path.read_text())
        case=info['case']
        assert info['code_sha256']==engine_sha,'MIXED_ENGINE_VERSION'
        fits=path.with_suffix('.fits.tsv')
        if fits.exists():fit_tables.append(pd.read_csv(fits,sep='\t'))
        train=np.asarray(case['train_indices']);test=np.asarray(case['test_indices'])
        frame=pd.read_csv(FROZEN/'data'/('FROZEN_PRIMARY_DATASET.tsv' if case['task']=='T2' else 'FROZEN_T1_SENSITIVITY_DATASET.tsv'),sep='\t')
        a,b=frame.iloc[train],frame.iloc[test]
        patient_overlap=len(set(a.patient)&set(b.patient))
        study_overlap=len(set(a.study)&set(b.study))
        valid=not(set(train)&set(test)) and (case['boundary']=='V0' or patient_overlap==0)
        if case['boundary']=='V2':valid=valid and study_overlap==0
        inner=np.array(case['inner_folds'])
        for k in sorted(set(inner)):
            if case['boundary']!='V0':assert not(set(a.iloc[np.flatnonzero(inner!=k)].patient)&set(a.iloc[np.flatnonzero(inner==k)].patient))
            if case['boundary']=='V2':assert case['target'] not in set(a.study)
        records.append(dict(case_id=case['id'],analysis=case['analysis'],boundary=case['boundary'],
            patient_overlap=patient_overlap,study_overlap=study_overlap,inner_training_only=True,status='PASS' if valid else 'FAIL',fit_status=info['status']))
        if info['status']=='COMPLETE':
            model=ROOT/info['model_path']
            assert hashlib.sha256(model.read_bytes()).hexdigest()==info['model_sha256']
            array=np.load(model,allow_pickle=False)
            assert set(array['training_record_ids'])==set(a.record)
            values=inputs[case['task']][case['representation']][test][:,array['feature_mask']]
            if case['scaled']:values=(values-array['scaler_mean'])/array['scaler_scale']
            linear=values@array['coefficient'].T+array['intercept']
            replay=softmax(linear,axis=1) if case['multiclass'] else expit(linear[:,0])
            saved=np.load(path.with_suffix('.npz'),allow_pickle=False)['probability']
            replay_errors.append(float(np.max(np.abs(replay-saved))))
            assert np.allclose(replay,saved,atol=1e-10,rtol=0),'SAVED_MODEL_PREDICTION_REPLAY_FAILED'
            prediction_path=path.with_suffix('.npz')
            assert hashlib.sha256(prediction_path.read_bytes()).hexdigest()==info['prediction_sha256']
            split_relative='splits/WITHIN_STUDY_POSITIVE_CONTROL_ASSIGNMENTS.tsv' if case['analysis'].startswith('positive') else 'splits/T2_MATCHED_TRAIN_SIZE_ASSIGNMENTS.tsv' if case['analysis']=='matched' else f"splits/{case['task']}_NESTED_ASSIGNMENTS.tsv"
            split_sha=hashlib.sha256((FROZEN/split_relative).read_bytes()).hexdigest()
            for row in b.itertuples():
                provenance.append(dict(analysis=case['analysis'],boundary=case['boundary'],record_id=row.record,patient_id=row.patient,study=row.study,seed=case['seed'],control_replicate=case['control_replicate'],case_id=case['id'],training_ids_file=info['model_path'],prediction_file=str(prediction_path.relative_to(ROOT)),model_sha256=info['model_sha256'],prediction_sha256=info['prediction_sha256'],feature_universe_sha256=feature_sha,split_path='phase0_3/'+split_relative,split_sha256=split_sha,frozen_root_sha256=case['freeze_root_sha256'],engine_code_sha256=info['code_sha256'],C=info['C'],l1_ratio=info['l1_ratio']))
    pd.DataFrame(provenance).to_csv(ROOT/'predictions/ALL_PREDICTION_PROVENANCE.tsv',sep='\t',index=False)
    pd.concat(fit_tables,ignore_index=True).to_csv(ROOT/'logs/MODEL_FIT_AUDIT.tsv',sep='\t',index=False)
    audit=pd.DataFrame(records)
    audit.to_csv(ROOT/'logs/LEAKAGE_AUDIT.tsv',sep='\t',index=False)
    pred=pd.read_csv(ROOT/'predictions/PRIMARY_OOF_PREDICTIONS.tsv',sep='\t')
    frame=pd.read_csv(FROZEN/'data/FROZEN_PRIMARY_DATASET.tsv',sep='\t')
    assert np.array_equal(pred.record_id,frame.record)
    assert np.array_equal(pred.label,frame.label)
    y=(frame.label=='tumor').to_numpy()
    metrics=pd.read_csv(ROOT/'results/PRIMARY_METRICS.tsv',sep='\t')
    for b in ['V0','V1','V2']:
        assert pred.filter(regex=f'^prob_{b}').notna().all().all()
        assert np.allclose(pred[f'prob_{b}'],pred[[f'prob_{b}_seed{s}' for s in [17,29,43]]].mean(axis=1),atol=1e-15)
        for name,score in [('AUROC',roc_auc_score(y,pred[f'prob_{b}'])),('AUPRC',average_precision_score(y,pred[f'prob_{b}']))]:
            reported=metrics[(metrics.scope=='POOLED')&(metrics.boundary==b)&(metrics.metric==name)].estimate.iloc[0]
            assert np.isclose(score,reported,atol=1e-14)
    assert audit.status.eq('PASS').all()
    result=dict(leakage_audit='PASS',cases_audited=len(audit),model_fit_rows=sum(len(t) for t in fit_tables),
                primary_complete_records=len(pred),primary_metric_recomputation='PASS',saved_model_prediction_replay='PASS',max_probability_replay_error=max(replay_errors))
    (ROOT/'logs/OUTPUT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
