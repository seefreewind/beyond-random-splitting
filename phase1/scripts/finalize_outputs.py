"""Freeze completed Phase 1 analyses after independent review."""
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def annotate_metadata():
    definitions=pd.DataFrame([
        dict(metric='AUROC',display_label='AUROC',definition='area_under_ROC'),
        dict(metric='AUPRC',display_label='AUPRC(AP)',definition='sklearn_average_precision_noninterpolated'),
        dict(metric='Brier',display_label='Brier',definition='mean_squared_probability_error'),
        dict(metric='log_loss',display_label='log loss',definition='binary_cross_entropy_probability_clip_1e-15'),
        dict(metric='balanced_accuracy',display_label='balanced accuracy',definition='mean_sensitivity_specificity_fixed_threshold_0.5')])
    definitions.to_csv(ROOT/'protocol/METRIC_DEFINITIONS.tsv',sep='\t',index=False)
    for folder in ['results','sensitivity','controls']:
        for path in (ROOT/folder).glob('*.tsv'):
            if path.name.startswith('._'):continue
            frame=pd.read_csv(path,sep='\t')
            if 'metric' in frame:
                frame['metric_label']=frame.metric.map(dict(zip(definitions.metric,definitions.display_label))).fillna(frame.metric)
            if 'AUPRC' in frame:frame['AUPRC_definition']='average_precision'
            frame.to_csv(path,sep='\t',index=False)


def main():
    assert (ROOT/'protocol/METRIC_DEFINITIONS.tsv').exists()
    scripts=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in sorted((ROOT/'scripts').glob('*.py')) if not path.name.startswith('._')]
    (ROOT/'protocol/FINAL_CODE_MANIFEST.json').write_text(json.dumps(scripts,indent=2)+'\n')
    state=json.loads((ROOT/'results/FINAL_STATUS.json').read_text())
    (ROOT/'logs/PIPELINE_FINISHED.json').write_text(json.dumps(dict(status=state['PHASE1_STATUS'],independent_final_audit='COMPLETE',manuscript_drafting_executed=False),indent=2)+'\n')
    excluded={'manifests/asset_hashes.tsv','manifests/asset_hashes.sha256','logs/FINAL_VERIFICATION.log'}
    rows=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name.startswith('._') or '__pycache__' in path.parts:continue
        if str(path.relative_to(ROOT)) in excluded:continue
        rows.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    (ROOT/'manifests').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT/'manifests/asset_hashes.tsv',sep='\t',index=False)
    checksum=sha(ROOT/'manifests/asset_hashes.tsv')
    (ROOT/'manifests/asset_hashes.sha256').write_text(checksum+'  asset_hashes.tsv\n')
    print(json.dumps(dict(assets=len(rows),sha256=checksum)))

if __name__=='__main__':
    if '--annotate-only' in sys.argv:annotate_metadata()
    else:main()
