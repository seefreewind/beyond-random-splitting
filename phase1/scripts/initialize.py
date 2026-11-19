"""Validate the original freeze and bind its actual asset names before fitting."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT.parent / 'phase0_3'
result = subprocess.run([sys.executable, str(FROZEN / 'scripts/verify_freeze.py')], capture_output=True, text=True)
(ROOT / 'logs/FREEZE_VERIFICATION.log').write_text(result.stdout + result.stderr)
if result.returncode:
    (ROOT / 'reports/PHASE1_DECISION.md').write_text('PHASE1_STATUS = ABORTED_FREEZE_VIOLATION\n')
    raise SystemExit('ABORTED_FREEZE_VIOLATION')
references = {
 'FROZEN_SAMPLE_MANIFEST.tsv':'data/FROZEN_PRIMARY_DATASET.tsv',
 'FROZEN_PATIENT_MANIFEST.tsv':'data/FROZEN_PRIMARY_DATASET.tsv',
 'FROZEN_GENE_UNIVERSE.tsv':'platform/COMMON_GENE_UNIVERSE.tsv',
 'FROZEN_SPLITS.tsv':'splits/T2_NESTED_ASSIGNMENTS.tsv',
 'FROZEN_PROTOCOL.yaml':'protocol/PROTOCOL_FREEZE.json',
 'FROZEN_HYPERPARAMETER_SPACE.yaml':'protocol/HYPERPARAMETER_SPACE.yaml',
 'FROZEN_ANALYSIS_PLAN.md':'reports/STATISTICAL_ANALYSIS_PLAN.md',
 'freeze/FREEZE_MANIFEST.tsv':'manifests/asset_hashes.tsv'}
rows=[]
for requested, actual in references.items():
    path=FROZEN / actual
    rows.append(dict(requested_name=requested, actual_frozen_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     interpretation='patient units derived from existing sample manifest' if requested=='FROZEN_PATIENT_MANIFEST.tsv' else 'existing frozen asset'))
pd.DataFrame(rows).to_csv(ROOT/'protocol/FROZEN_ASSET_BINDINGS.tsv',sep='\t',index=False)
for task,prefix,name in [('T2','primary','FROZEN_PRIMARY_DATASET.tsv'),('T1','T1_sensitivity','FROZEN_T1_SENSITIVITY_DATASET.tsv')]:
    frame=pd.read_csv(FROZEN/'data'/name,sep='\t')
    matrix=np.load(FROZEN/f'data/{prefix}_inputs.npz',allow_pickle=False)
    sys.path.insert(0,str(FROZEN/'scripts'))
    from prepare_inputs import rank_samples
    assert np.array_equal(matrix['record_ids'],frame.record.to_numpy())
    assert np.array_equal(matrix['rank'],rank_samples(matrix['provided']))
    assert all(np.array_equal(matrix['rank'][i],rank_samples(matrix['provided'][i:i+1])[0]) for i in range(len(frame)))
    if task=='T2':
        census=[]
        for study,part in frame.groupby('study',sort=True):
            pairs=part.groupby('patient').label.agg(list)
            census.append(dict(study=study,N_records=len(part),N_patients=part.patient.nunique(),
                N_tumor_records=(part.label=='tumor').sum(),N_normal_records=(part.label!='tumor').sum(),
                N_complete_patient_pairs=pairs.map(lambda x:len(x)==2 and len(set(x))==2).sum()))
        pd.DataFrame(census).to_csv(ROOT/'data/PRIMARY_ANALYSIS_CENSUS.tsv',sep='\t',index=False)
(ROOT/'protocol/PROTOCOL_GAP_NOTE.md').write_text('''# Pre-metric execution record

Recorded before the first real disease-model fit or metric. Phase 0.3 uses different filenames from the generic names listed in the Phase 1 request. FROZEN_ASSET_BINDINGS.tsv binds every requested role to its existing, verified, immutable asset. The patient units derive from the frozen sample manifest. The original freeze remains unchanged.

Seed aggregation, NC1, NC2 and training-size matching are already fully specified. Three seed probabilities are averaged arithmetically per record. Balanced accuracy uses the previously frozen constant threshold 0.5. No result-driven gap resolution is required.

Implementation logs one row per fit and checkpoints each outer model. Independent outer cases may execute concurrently, with each numerical library limited to one thread. NC1 and NC2 training corruption is freshly restricted to each inner-training or outer-training scope. A fixed scope/control seed is shared across all candidate hyperparameters, matching the frozen seed definition.

All authorized T2 analyses run before T1 sensitivity. Both T2 and T1 within-study positive-control split sets are executed. The frozen source-normalized plus training-scaling sensitivity is retained. The optional frozen study-identity diagnostic is executed.

Algorithm instability will be flagged when the seed-specific primary or secondary AUROC gap range exceeds 0.05. This is a descriptive execution flag set before real metrics, with all seed values reported. The interpretation patterns use the raw differences and their intervals, without a post-hoc equivalence margin.

A convergence failure is logged and stops the affected analysis under the frozen policy. The frozen grid and iteration limit remain fixed. Completed independent analyses are retained. Technical execution repairs must preserve seeds, objective, parameters and evaluation scopes.
''')
print('FREEZE_PASS; SAMPLE_LOCAL_PASS; PRE_METRIC_EXECUTION_RECORD_WRITTEN')
