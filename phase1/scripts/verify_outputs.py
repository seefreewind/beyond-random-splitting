"""Read-only verification of the completed Phase 1 package and original freeze."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    assets=ROOT/'manifests/asset_hashes.tsv'
    assert sha(assets)==(ROOT/'manifests/asset_hashes.sha256').read_text().split()[0]
    table=pd.read_csv(assets,sep='\t')
    for row in table.to_dict('records'):
        path=ROOT/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    result=subprocess.run([sys.executable,str(ROOT.parent/'phase0_3/scripts/verify_freeze.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    required=['predictions/PRIMARY_OOF_PREDICTIONS.tsv','predictions/PREDICTION_PROVENANCE.tsv',
        'results/PRIMARY_METRICS.tsv','results/PRIMARY_CONTRASTS.tsv','results/LOSO_PER_STUDY_METRICS.tsv',
        'results/PER_STUDY_BOUNDARY_COMPARISON.tsv','results/SEED_STABILITY.tsv','bootstrap/PRIMARY_BOOTSTRAP_2000.tsv',
        'bootstrap/BOOTSTRAP_SUMMARY.tsv','controls/POSITIVE_CONTROL.tsv','controls/NEGATIVE_CONTROL_1.tsv','controls/NEGATIVE_CONTROL_2.tsv',
        'sensitivity/ELASTIC_NET_RESULTS.tsv','sensitivity/TRAINING_SIZE_MATCHED_RESULTS.tsv','sensitivity/T1_SENSITIVITY_RESULTS.tsv',
        'logs/MODEL_FIT_AUDIT.tsv','logs/LEAKAGE_AUDIT.tsv']
    for relative in required:assert (ROOT/relative).is_file(),relative
    for report in ['EXECUTIVE_REPORT','PRIMARY_ANALYSIS','LOSO_RESULTS','CONTROLS','SENSITIVITY','LIMITATIONS','MANUSCRIPT_QUALIFICATION','DECISION']:
        assert (ROOT/f'reports/PHASE1_{report}.md').is_file(),report
    original=pd.read_csv(ROOT.parent/'phase0_3/data/FROZEN_PRIMARY_DATASET.tsv',sep='\t')
    pred=pd.read_csv(ROOT/'predictions/PRIMARY_OOF_PREDICTIONS.tsv',sep='\t')
    assert len(pred)==len(original) and pred.filter(regex='^prob_').notna().all().all()
    assert len(pd.read_csv(ROOT/'bootstrap/PRIMARY_BOOTSTRAP_2000.tsv',sep='\t'))==2000
    print(json.dumps(dict(status='PASS',verified_assets=len(table),primary_records=len(pred),primary_patients=int(original.patient.nunique()),bootstrap_replicates=2000,original_freeze='PASS'),indent=2))

if __name__=='__main__':main()
