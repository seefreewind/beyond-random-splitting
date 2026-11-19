#!/usr/bin/env python3
"""Plan and run checkpointed independent outer folds under the original freeze."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import yaml
from engine import run_case, initialize_worker
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT.parent/'phase0_3'


def plan(analyses):
    config=yaml.safe_load((FROZEN/'protocol/HYPERPARAMETER_SPACE.yaml').read_text())
    root_sha=(FROZEN/'manifests/asset_hashes.sha256').read_text().split()[0]
    cases=[]
    for analysis in analyses:
        task='T1' if analysis in ['T1','positive_T1'] else 'T2'
        frame=pd.read_csv(FROZEN/'data'/('FROZEN_PRIMARY_DATASET.tsv' if task=='T2' else 'FROZEN_T1_SENSITIVITY_DATASET.tsv'),sep='\t')
        lookup=dict(zip(frame.record,range(len(frame))))
        if analysis=='matched':
            splits=pd.read_csv(FROZEN/'splits/T2_MATCHED_TRAIN_SIZE_ASSIGNMENTS.tsv',sep='\t')
            groups=splits.groupby(['repeat','outer_fold','target_study','arm'],sort=True)
        elif analysis.startswith('positive'):
            splits=pd.read_csv(FROZEN/'splits/WITHIN_STUDY_POSITIVE_CONTROL_ASSIGNMENTS.tsv',sep='\t')
            groups=splits[splits.task==task].groupby(['repeat','outer_fold','study'],sort=True)
        else:
            splits=pd.read_csv(FROZEN/f'splits/{task}_NESTED_ASSIGNMENTS.tsv',sep='\t')
            if analysis=='study_identity': splits=splits[splits.boundary=='V1']
            groups=splits.groupby(['boundary','repeat','outer_fold'],sort=True)
        for keys,part in groups:
            part=part.copy()
            if analysis=='matched':
                repeat,fold,target,arm=keys
                boundary='V1' if arm=='known_studies' else 'V2'
                part['outer_role']=part.role
            elif analysis.startswith('positive'):
                repeat,fold,target=keys
                arm='within_study'
                boundary='V1'
            else:
                boundary,repeat,fold=keys
                target='ALL' if boundary!='V2' else part[part.outer_role=='test'].study.iloc[0]
                arm='native'
            train=part[part.outer_role=='train']
            test=part[part.outer_role=='test']
            control=int(analysis[-1]) if analysis in ['NC1','NC2'] else 0
            for replicate in range(20 if control else 1):
                case_id=f'{analysis}_{boundary}_r{repeat}_f{fold}_{target}_{arm}_c{replicate}'
                cases.append(dict(id=case_id,analysis=analysis,task=task,boundary=boundary,repeat=int(repeat),
                    outer_fold=int(fold),target=target,arm=arm,seed=config['validation']['repeat_seeds'][int(repeat)],
                    train_indices=[lookup[r] for r in train.record_id],test_indices=[lookup[r] for r in test.record_id],
                    inner_folds=train.inner_fold.astype(int).tolist(),
                    C_grid=config['models']['sensitivity' if analysis=='elastic_net' else 'primary']['C'],
                    ratio_grid=config['models']['sensitivity']['l1_ratio'] if analysis=='elastic_net' else [0.0],
                    representation='provided' if analysis=='source_scaled' else 'rank',
                    scaled=analysis=='source_scaled',control=control,control_replicate=replicate if control else -1,
                    multiclass=analysis=='study_identity',freeze_root_sha256=root_sha))
    return cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['primary','controls','robustness','T1'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    check=subprocess.run([sys.executable,str(FROZEN/'scripts/verify_freeze.py')],capture_output=True,text=True)
    if check.returncode:
        raise SystemExit('ABORTED_FREEZE_VIOLATION\n'+check.stderr)
    analyses={'primary':['primary'],'controls':['positive_T2','NC1','NC2'],
              'robustness':['elastic_net','source_scaled','matched','study_identity'],
              'T1':['T1','positive_T1']}[args.stage]
    if args.stage=='T1':
        assert (ROOT/'results/PRIMARY_CONTRASTS.tsv').exists(),'Complete T2 primary analysis before T1'
    cases=plan(analyses)
    engine_sha=hashlib.sha256((ROOT/'scripts/engine.py').read_bytes()).hexdigest()
    for case in cases:
        existing=ROOT/'cases'/f"{case['id']}.json"
        if existing.exists():
            assert json.loads(existing.read_text())['code_sha256']==engine_sha,'CHECKPOINT_CODE_VERSION_MISMATCH'
    (ROOT/'protocol'/f'{args.stage}_CASE_PLAN.json').write_text(json.dumps(cases,indent=2)+'\n')
    started=time.time()
    progress=ROOT/'logs'/f'{args.stage}_PROGRESS.json'
    pending=list(cases)
    finished=[]
    failures=set()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
        active={}
        while pending or active:
            while pending and len(active)<args.workers:
                case=pending.pop(0)
                if case['analysis'] not in failures:
                    active[pool.submit(run_case,case)]=case
            if not active: break
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                case=active.pop(future)
                result=future.result()
                finished.append(dict(case_id=case['id'],analysis=case['analysis'],status=result['status'],elapsed_seconds=result.get('elapsed_seconds')))
                if result['status']=='FAIL': failures.add(case['analysis'])
            progress.write_text(json.dumps(dict(stage=args.stage,planned=len(cases),completed=len(finished),
                failures=sorted(failures),active=[c['id'] for c in active.values()],remaining=len(pending),elapsed_seconds=time.time()-started),indent=2)+'\n')
    pd.DataFrame(finished).to_csv(ROOT/'logs'/f'{args.stage}_COMPLETION.tsv',sep='\t',index=False)
    print(json.dumps(dict(stage=args.stage,planned=len(cases),completed=len(finished),failures=sorted(failures),elapsed_seconds=time.time()-started)))

if __name__=='__main__': main()
