#!/usr/bin/env python3
"""Wait for authorized batches, then complete T1, summaries and deterministic audits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT.parent/'phase0_3'


def execute(script,args,log):
    with (ROOT/'logs'/log).open('w') as handle:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed with exit {result.returncode}; see {log}')


def main():
    while not all((ROOT/'logs'/f'{stage}_COMPLETION.tsv').exists() for stage in ['controls','robustness']):
        time.sleep(30)
    execute('run_stage.py',['T1','--workers','4'],'T1_run.log')
    execute('summarize.py',['primary'],'primary_statistics_final.log')
    execute('summarize.py',['all'],'all_statistics.log')
    execute('finalize_outputs.py',['--annotate-only'],'metric_annotations.log')
    execute('audit_outputs.py',[],'output_audit.log')
    with (ROOT/'logs/POST_RUN_FREEZE_VERIFICATION.log').open('w') as handle:
        checked=subprocess.run([sys.executable,str(FROZEN/'scripts/verify_freeze.py')],stdout=handle,stderr=subprocess.STDOUT)
    if checked.returncode:
        (ROOT/'reports/PHASE1_DECISION.md').write_text('PHASE1_STATUS = ABORTED_FREEZE_VIOLATION\n')
        raise RuntimeError('ABORTED_FREEZE_VIOLATION')
    execute('write_reports.py',[],'report_generation.log')
    (ROOT/'logs/PIPELINE_FINISHED.json').write_text(json.dumps({'status':'ANALYSES_AND_REPORTS_GENERATED','independent_final_audit':'PENDING'})+'\n')
    print('ANALYSES_AND_REPORTS_GENERATED; INDEPENDENT_FINAL_AUDIT_PENDING')

if __name__=='__main__':main()
