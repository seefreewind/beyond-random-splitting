from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ['cases','models','logs','protocol','results','sensitivity','controls','predictions','reports']:
 (ROOT/'phase1'/folder).mkdir(parents=True,exist_ok=True)
for folder in ['figures','reports','auc_decomposition','score_comparability','representation']:
 (ROOT/'phase1_1'/folder).mkdir(parents=True,exist_ok=True)
print('Output directories ready.')
