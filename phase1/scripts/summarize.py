"""Calculate preregistered real metrics from complete, ID-aligned outer predictions."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[variable]='1'
import argparse
from functools import lru_cache
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from statistical_analysis import bootstrap_weights,weighted_metrics,interval,validate_bootstrap,calibration
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT.parent/'phase0_3'
SEEDS=[17,29,43]


@lru_cache(maxsize=None)
def read_case_metadata(path):
    return json.loads(path.read_text())


def manifest(task='T2'):
    frame=pd.read_csv(FROZEN/'data'/('FROZEN_PRIMARY_DATASET.tsv' if task=='T2' else 'FROZEN_T1_SENSITIVITY_DATASET.tsv'),sep='\t')
    frame['y']=(frame.label=='tumor').astype(int)
    return frame


def load_predictions(analysis,task='T2',control_replicate=None):
    frame=manifest(task)
    records=[]
    split_relative='splits/WITHIN_STUDY_POSITIVE_CONTROL_ASSIGNMENTS.tsv' if analysis.startswith('positive') else 'splits/T2_MATCHED_TRAIN_SIZE_ASSIGNMENTS.tsv' if analysis=='matched' else f'splits/{task}_NESTED_ASSIGNMENTS.tsv'
    feature_sha=hashlib.sha256((FROZEN/'platform/COMMON_GENE_UNIVERSE.tsv').read_bytes()).hexdigest()
    split_sha=hashlib.sha256((FROZEN/split_relative).read_bytes()).hexdigest()
    boundaries=['V1','V2'] if analysis=='matched' else ['V1'] if analysis.startswith('positive') else ['V0','V1','V2']
    result=frame.rename(columns={'record':'record_id','patient':'patient_id'})[['record_id','patient_id','study','label']].copy()
    for boundary in boundaries:
        for seed in SEEDS: result[f'prob_{boundary}_seed{seed}']=np.nan
    for path in sorted((ROOT/'cases').glob(f'{analysis}_*.json')):
        if path.name.startswith('._'): continue
        info=read_case_metadata(path)
        case=info['case']
        if case['task']!=task or (control_replicate is not None and case['control_replicate']!=control_replicate): continue
        if info['status']!='COMPLETE': continue
        array=np.load(path.with_suffix('.npz'),allow_pickle=False)
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==info['prediction_sha256']
        indices=array['indices']
        column=f"prob_{case['boundary']}_seed{case['seed']}"
        assert result.loc[indices,column].isna().all(),'Duplicate outer prediction'
        result.loc[indices,column]=array['probability']
        for index in indices:
            records.append(dict(analysis=analysis,record_id=frame.loc[index,'record'],patient_id=frame.loc[index,'patient'],
                study=frame.loc[index,'study'],boundary=case['boundary'],seed=case['seed'],case_id=case['id'],model='ElasticNet' if analysis=='elastic_net' else 'L2',
                C=info['C'],l1_ratio=info['l1_ratio'],case_signature=info['case_signature'],
                input_path=f'phase0_3/data/{"primary" if task=="T2" else "T1_sensitivity"}_inputs.npz',
                frozen_input_root_sha256=case['freeze_root_sha256'],training_ids_file=info['model_path'],
                model_sha256=info['model_sha256'],test_record_id=frame.loc[index,'record'],
                feature_universe_sha256=feature_sha,split_path='phase0_3/'+split_relative,split_sha256=split_sha,engine_code_sha256=info['code_sha256']))
    for boundary in boundaries:
        columns=[f'prob_{boundary}_seed{s}' for s in SEEDS]
        result[f'prob_{boundary}']=result[columns].mean(axis=1,skipna=False)
    complete=result[[f'prob_{b}' for b in boundaries]].notna().all().all()
    return frame,result,pd.DataFrame(records),bool(complete)


def evaluate(frame,pred,boundaries,analysis,save_bootstrap=False):
    weights=bootstrap_weights(frame)
    metrics_rows=[]
    contrasts=[]
    all_distributions={}
    subsets=[('POOLED',frame.index.to_numpy())]+[(study,part.index.to_numpy()) for study,part in frame.groupby('study',sort=True)]
    for scope,indices in subsets:
        local=frame.loc[indices]
        distributions={}
        for boundary in boundaries:
            p=pred[f'prob_{boundary}'].to_numpy()[indices]
            point=weighted_metrics(local.y.to_numpy(),p,np.ones((1,len(indices))))
            dist=weighted_metrics(local.y.to_numpy(),p,weights[:,indices])
            distributions[boundary]=dist
            for metric,estimate in point.items():
                low,high,valid=interval(dist[metric])
                metrics_rows.append(dict(analysis=analysis,scope=scope,boundary=boundary,metric=metric,estimate=float(estimate[0]),
                    CI_lower=low,CI_upper=high,bootstrap_valid=valid,N_records=len(local),N_patients=local.patient.nunique(),positive_prevalence=local.y.mean()))
        for name,a,b in [('study_boundary','V1','V2'),('patient_boundary','V0','V1')]:
            if a not in boundaries or b not in boundaries: continue
            for metric in ['AUROC','AUPRC','Brier','log_loss','balanced_accuracy']:
                point_a=next(r['estimate'] for r in metrics_rows if r['scope']==scope and r['boundary']==a and r['metric']==metric)
                point_b=next(r['estimate'] for r in metrics_rows if r['scope']==scope and r['boundary']==b and r['metric']==metric)
                low,high,valid=interval(distributions[a][metric]-distributions[b][metric])
                contrasts.append(dict(analysis=analysis,scope=scope,contrast=name,definition=f'{a}_minus_{b}',metric=metric,
                    estimate=point_a-point_b,CI_lower=low,CI_upper=high,bootstrap_valid=valid))
        if scope=='POOLED': all_distributions=distributions
    for boundary in boundaries:
        for metric in ['AUROC','AUPRC','Brier','log_loss','balanced_accuracy']:
            rows=[r for r in metrics_rows if r['scope']!='POOLED' and r['boundary']==boundary and r['metric']==metric]
            metrics_rows.append(dict(analysis=analysis,scope='MACRO_DESCRIPTIVE',boundary=boundary,metric=metric,
                estimate=float(np.mean([r['estimate'] for r in rows])),CI_lower=np.nan,CI_upper=np.nan,
                bootstrap_valid=0,N_records=len(frame),N_patients=frame.patient.nunique(),positive_prevalence=np.nan))
    if save_bootstrap:
        boot=pd.DataFrame({'replicate':np.arange(2000),'seed':170329})
        for b,d in all_distributions.items():
            for metric,values in d.items(): boot[f'{metric}_{b}']=values
        for metric in ['AUROC','AUPRC']:
            boot[f'{metric}_delta_patient']=boot[f'{metric}_V0']-boot[f'{metric}_V1']
            boot[f'{metric}_delta_study']=boot[f'{metric}_V1']-boot[f'{metric}_V2']
        boot.to_csv(ROOT/'bootstrap/PRIMARY_BOOTSTRAP_2000.tsv',sep='\t',index=False)
    return pd.DataFrame(metrics_rows),pd.DataFrame(contrasts)


def seed_stability(frame,pred):
    rows=[]
    for seed in SEEDS:
        row={'seed':seed}
        for b in ['V0','V1','V2']:
            scores=weighted_metrics(frame.y,pred[f'prob_{b}_seed{seed}'],np.ones((1,len(frame))))
            row[f'AUROC_{b}']=scores['AUROC'][0]
        row['delta_patient']=row['AUROC_V0']-row['AUROC_V1']
        row['delta_study']=row['AUROC_V1']-row['AUROC_V2']
        rows.append(row)
    return pd.DataFrame(rows)


def primary():
    validate_bootstrap()
    frame,pred,provenance,complete=load_predictions('primary')
    pred.to_csv(ROOT/'predictions/PRIMARY_OOF_PREDICTIONS.tsv',sep='\t',index=False)
    provenance.to_csv(ROOT/'predictions/PREDICTION_PROVENANCE.tsv',sep='\t',index=False)
    if not complete:
        pred[pred.filter(regex='^prob_').isna().any(axis=1)].to_csv(ROOT/'logs/MISSING_PRIMARY_PREDICTIONS.tsv',sep='\t',index=False)
        raise RuntimeError('PRIMARY_ANALYSIS_INCOMPLETE')
    metrics,contrasts=evaluate(frame,pred,['V0','V1','V2'],'primary',True)
    metrics.to_csv(ROOT/'results/PRIMARY_METRICS.tsv',sep='\t',index=False)
    contrasts.to_csv(ROOT/'results/PRIMARY_CONTRASTS.tsv',sep='\t',index=False)
    pd.concat([metrics[metrics.scope=='POOLED'].assign(kind='metric'),contrasts[contrasts.scope=='POOLED'].assign(kind='contrast')]).to_csv(ROOT/'bootstrap/BOOTSTRAP_SUMMARY.tsv',sep='\t',index=False)
    seed_stability(frame,pred).to_csv(ROOT/'results/SEED_STABILITY.tsv',sep='\t',index=False)
    loso=[]
    comparison=[]
    for study,part in frame.groupby('study',sort=True):
        subset=metrics[(metrics.scope==study)&(metrics.boundary=='V2')]
        row=dict(held_out_study=study,N_patients=part.patient.nunique(),N_records=len(part),N_tumor=part.y.sum(),N_normal=(1-part.y).sum(),positive_prevalence=part.y.mean())
        row.update(dict(zip(subset.metric,subset.estimate)))
        for metric in ['AUROC','AUPRC']:
            item=subset[subset.metric==metric].iloc[0]
            row[f'{metric}_CI_lower']=item.CI_lower; row[f'{metric}_CI_upper']=item.CI_upper
        loso.append(row)
        row=dict(study=study,N_patients=part.patient.nunique())
        for metric in ['AUROC','AUPRC']:
            for b in ['V1','V2']: row[f'{metric}_{b}']=metrics[(metrics.scope==study)&(metrics.boundary==b)&(metrics.metric==metric)].estimate.iloc[0]
            row[f'delta_{metric}']=row[f'{metric}_V1']-row[f'{metric}_V2']
            item=contrasts[(contrasts.scope==study)&(contrasts.contrast=='study_boundary')&(contrasts.metric==metric)].iloc[0]
            row[f'delta_{metric}_CI_lower']=item.CI_lower;row[f'delta_{metric}_CI_upper']=item.CI_upper
        comparison.append(row)
    pd.DataFrame(loso).to_csv(ROOT/'results/LOSO_PER_STUDY_METRICS.tsv',sep='\t',index=False)
    pd.DataFrame(comparison).to_csv(ROOT/'results/PER_STUDY_BOUNDARY_COMPARISON.tsv',sep='\t',index=False)
    # Calibration is descriptive and never fed back into disease probabilities.
    rows=[]
    weights=bootstrap_weights(frame)
    for scope,indices in [('POOLED',frame.index.to_numpy())]+[(s,p.index.to_numpy()) for s,p in frame.groupby('study')]:
        part=frame.loc[indices].reset_index(drop=True)
        for b in ['V0','V1','V2']:
            probability=pred[f'prob_{b}'].to_numpy()[indices]
            point=calibration(part,probability)
            row=dict(scope=scope,boundary=b,**point)
            if point['status']=='ESTIMATED':
                boot=[calibration(part,probability,w[indices]) for w in weights]
                for key in ['intercept','slope']:
                    low,high,valid=interval([r[key] for r in boot])
                    row[f'{key}_CI_lower']=low;row[f'{key}_CI_upper']=high;row[f'{key}_bootstrap_valid']=valid
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/CALIBRATION_DESCRIPTIVE.tsv',sep='\t',index=False)
    print('PRIMARY_METRICS_COMPLETE')


def sensitivity():
    summary=[]
    seed_rows=[]
    for analysis,task in [('elastic_net','T2'),('source_scaled','T2'),('matched','T2'),('T1','T1')]:
        frame,pred,provenance,complete=load_predictions(analysis,task)
        pred.to_csv(ROOT/'predictions'/f'{analysis.upper()}_OOF_PREDICTIONS.tsv',sep='\t',index=False)
        provenance.to_csv(ROOT/'predictions'/f'{analysis.upper()}_PROVENANCE.tsv',sep='\t',index=False)
        summary.append(dict(analysis=analysis,complete=complete))
        if not complete:
            output={'elastic_net':'ELASTIC_NET_RESULTS','source_scaled':'SOURCE_SCALED_RESULTS','matched':'TRAINING_SIZE_MATCHED_RESULTS','T1':'T1_SENSITIVITY_RESULTS'}[analysis]
            pd.DataFrame([dict(analysis=analysis,kind='failure',scope='NOT_ESTIMATED',contrast='NOT_ESTIMATED',metric='NA',status='INCOMPLETE',estimate=np.nan,CI_lower=np.nan,CI_upper=np.nan)]).to_csv(ROOT/'sensitivity'/f'{output}.tsv',sep='\t',index=False)
            continue
        for seed in SEEDS:
            row={'analysis':analysis,'seed':seed}
            for boundary in ['V0','V1','V2']:
                column=f'prob_{boundary}_seed{seed}'
                row[f'AUROC_{boundary}']=weighted_metrics(frame.y,pred[column],np.ones((1,len(frame))))['AUROC'][0] if column in pred else np.nan
            row['delta_patient']=row['AUROC_V0']-row['AUROC_V1']
            row['delta_study']=row['AUROC_V1']-row['AUROC_V2']
            seed_rows.append(row)
        boundaries=['V1','V2'] if analysis=='matched' else ['V0','V1','V2']
        metrics,contrasts=evaluate(frame,pred,boundaries,analysis)
        name={'elastic_net':'ELASTIC_NET_RESULTS','source_scaled':'SOURCE_SCALED_RESULTS','matched':'TRAINING_SIZE_MATCHED_RESULTS','T1':'T1_SENSITIVITY_RESULTS'}[analysis]
        combined=pd.concat([metrics.assign(kind='metric'),contrasts.assign(kind='contrast')],ignore_index=True)
        combined.to_csv(ROOT/'sensitivity'/f'{name}.tsv',sep='\t',index=False)
        if analysis=='T1':
            rows=[]
            for subtype,keep in [('cancer_patient_normal',frame.tissue!='healthy normal'),('healthy_normal',(frame.tissue=='tumor')|(frame.tissue=='healthy normal'))]:
                indices=frame.index[keep]
                for scope,subset in [('POOLED',indices)]+[(s,np.intersect1d(indices,p.index)) for s,p in frame.groupby('study')]:
                    local=frame.loc[subset]
                    for b in boundaries:
                        if local.y.nunique()!=2:
                            rows.append(dict(normal_subtype=subtype,scope=scope,boundary=b,status='NA_NO_BOTH_CLASSES'))
                        else:
                            values=weighted_metrics(local.y,pred.loc[subset,f'prob_{b}'],np.ones((1,len(subset))))
                            rows.append(dict(normal_subtype=subtype,scope=scope,boundary=b,status='ESTIMATED',N_records=len(local),N_patients=local.patient.nunique(),positive_prevalence=local.y.mean(),**{k:v[0] for k,v in values.items()}))
            pd.DataFrame(rows).to_csv(ROOT/'sensitivity/T1_NORMAL_SUBTYPE_RESULTS.tsv',sep='\t',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'sensitivity/COMPLETION_STATUS.tsv',sep='\t',index=False)
    pd.DataFrame(seed_rows).to_csv(ROOT/'sensitivity/SEED_STABILITY.tsv',sep='\t',index=False)


def controls():
    rows=[]
    for task in ['T2','T1']:
        frame,pred,_,complete=load_predictions('positive_'+task,task)
        if complete:
            metrics,_=evaluate(frame,pred,['V1'],'positive_'+task)
            rows.append(metrics.assign(task=task))
            pred.to_csv(ROOT/'predictions'/f'POSITIVE_{task}_OOF.tsv',sep='\t',index=False)
    if rows: pd.concat(rows).to_csv(ROOT/'controls/POSITIVE_CONTROL.tsv',sep='\t',index=False)
    decisions=[]
    for control in [1,2]:
        allrows=[]
        predictions=[]
        frame=manifest()
        weights=bootstrap_weights(frame)
        for replicate in range(20):
            _,pred,_,complete=load_predictions(f'NC{control}',control_replicate=replicate)
            if not complete:
                decisions.append(dict(control=control,status='INCOMPLETE',replicate=replicate))
                continue
            pred.to_csv(ROOT/'predictions'/f'NC{control}_REPLICATE_{replicate:02d}_OOF.tsv',sep='\t',index=False)
            distributions={}
            for b in ['V0','V1','V2']:
                for scope,indices in [('POOLED',frame.index.to_numpy())]+[(s,p.index.to_numpy()) for s,p in frame.groupby('study')]:
                    point=weighted_metrics(frame.loc[indices].y,pred.loc[indices,f'prob_{b}'],np.ones((1,len(indices))))
                    allrows.append(dict(control=control,replicate=replicate,boundary=b,scope=scope,**{k:v[0] for k,v in point.items()}))
                distributions[b]=weighted_metrics(frame.y,pred[f'prob_{b}'],weights)['AUROC']
            predictions.append(distributions)
        pd.DataFrame(allrows).to_csv(ROOT/'controls'/f'NEGATIVE_CONTROL_{control}.tsv',sep='\t',index=False)
        if len(predictions)==20:
            for b in ['V0','V1','V2']:
                values=np.mean([r[b] for r in predictions],axis=0)
                low,high,valid=interval(values)
                subset=[r['AUROC'] for r in allrows if r['scope']=='POOLED' and r['boundary']==b]
                count=int((np.array(subset)>.5).sum())
                decisions.append(dict(control=control,boundary=b,replicates=20,above_chance=count,mean_AUROC=float(np.mean(subset)),
                    CI_lower=low,CI_upper=high,bootstrap_valid=valid,replicate_SD=float(np.std(subset,ddof=1)),replicate_min=float(np.min(subset)),replicate_max=float(np.max(subset)),status='FAIL' if count>=16 and low>.5 else 'PASS'))
    pd.DataFrame(decisions).to_csv(ROOT/'controls/CONTROL_DECISIONS.tsv',sep='\t',index=False)


def study_identity():
    frame=manifest()
    rows=[]
    for repeat,seed in enumerate(SEEDS):
        p=np.full((len(frame),3),np.nan)
        for path in sorted((ROOT/'cases').glob('study_identity_*.json')):
            if path.name.startswith('._'):continue
            info=read_case_metadata(path)
            if info['status']!='COMPLETE' or info['case']['repeat']!=repeat:continue
            assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==info['prediction_sha256']
            data=np.load(path.with_suffix('.npz'),allow_pickle=False)
            assert np.isnan(p[data['indices']]).all(),'Duplicate study diagnostic OOF prediction'
            p[data['indices']]=data['probability']
        if np.isfinite(p).all():
            truth=pd.Categorical(frame.study,categories=sorted(frame.study.unique())).codes
            rows.append(dict(seed=seed,balanced_accuracy=balanced_accuracy_score(truth,np.argmax(p,axis=1)),chance=1/3))
    pd.DataFrame(rows).to_csv(ROOT/'results/STUDY_IDENTITY_DIAGNOSTIC.tsv',sep='\t',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['primary','all'])
    args=parser.parse_args()
    validate_bootstrap()
    if args.part=='primary': primary()
    else:
        sensitivity();controls();study_identity()
