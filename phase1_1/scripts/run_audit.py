"""Post-primary explanation from immutable saved OOF probabilities. No fitting."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
P1=ROOT.parent/'phase1'
sys.path.insert(0,str(P1/'scripts'))
from statistical_analysis import bootstrap_weights,weighted_metrics,interval,calibration
sys.path.insert(0,str(ROOT.parent/'phase0_3/scripts'))
from pipeline_components import patient_bootstrap_indices
STUDIES=['GSE44076','GSE37182','GSE50760']
def save(df,path):df.to_csv(ROOT/path,sep='\t',index=False)
def load(name):
    f=pd.read_csv(P1/f'predictions/{name}_OOF_PREDICTIONS.tsv',sep='\t')
    f['y']=(f.label=='tumor').astype(int)
    f['patient']=f.patient_id
    return f
frames={name:load(file) for name,file in [('L2_rank','PRIMARY'),('Elastic_Net','ELASTIC_NET'),('source_scaled','SOURCE_SCALED'),('matched','MATCHED'),('T1','T1')]}
assert frames['L2_rank'].y.sum()==130
# Recompute every formal primary/sensitivity metric and CI from original draws.
checks=[]
for name,f in frames.items():
    if name=='L2_rank':
        tables=[pd.read_csv(P1/'results/PRIMARY_METRICS.tsv',sep='\t').assign(kind='metric'),pd.read_csv(P1/'results/PRIMARY_CONTRASTS.tsv',sep='\t').assign(kind='contrast')]
        official=pd.concat(tables,ignore_index=True)
    else:
        file={'Elastic_Net':'ELASTIC_NET','source_scaled':'SOURCE_SCALED','matched':'TRAINING_SIZE_MATCHED','T1':'T1_SENSITIVITY'}[name]
        official=pd.read_csv(P1/f'sensitivity/{file}_RESULTS.tsv',sep='\t')
    w=bootstrap_weights(f)
    for scope in ['POOLED',*sorted(f.study.unique())]:
        idx=np.arange(len(f)) if scope=='POOLED' else np.flatnonzero(f.study==scope)
        y=f.y.to_numpy()[idx]
        ds={};ps={}
        for b in ['V0','V1','V2']:
            if f'prob_{b}' not in f:continue
            p=f[f'prob_{b}'].to_numpy()[idx]
            ps[b]=weighted_metrics(y,p,np.ones(len(idx)))
            ds[b]=weighted_metrics(y,p,w[:,idx])
            assert np.isclose(ps[b]['AUROC'][0],roc_auc_score(y,p),atol=1e-14)
            assert np.isclose(ps[b]['AUPRC'][0],average_precision_score(y,p),atol=1e-14)
        for row in official[official.scope==scope].to_dict('records'):
            metric=row['metric']
            if row['kind']=='metric':point=ps[row['boundary']][metric][0];dist=ds[row['boundary']][metric]
            else:
                a,b=('V1','V2') if row['contrast']=='study_boundary' else ('V0','V1')
                point=ps[a][metric][0]-ps[b][metric][0];dist=ds[a][metric]-ds[b][metric]
            lo,hi,_=interval(dist)
            for field,value in [('estimate',point),('CI_lower',lo),('CI_upper',hi)]:
                error=abs(value-row[field]);assert error<1e-12,(name,scope,row,field,error)
                checks.append(dict(analysis=name,scope=scope,kind=row['kind'],metric=metric,field=field,max_error=error,status='PASS'))
save(pd.DataFrame(checks),'reports/NUMERIC_CHECKS.tsv')
print('NUMERIC_RECOMPUTATION_PASS',len(checks),flush=True)
# Exact ordered positive-negative comparison cells.
allrows=[];summaries=[];seedrows=[];distributions=[];offsets=[];quality=[];rankrows=[]
for name,f in frames.items():
    if name in ['matched','T1']:continue
    y=f.y.to_numpy();total=int(y.sum()*(1-y).sum())
    for seed in ['aggregate',17,29,43]:
        comps={}
        for b in ['V1','V2']:
            p=f[f'prob_{b}'+('' if seed=='aggregate' else f'_seed{seed}')].to_numpy()
            rows=[]
            for a in STUDIES:
                for c in STUDIES:
                    pos=p[(f.study==a)&(y==1)];neg=p[(f.study==c)&(y==0)]
                    diff=pos[:,None]-neg[None,:]
                    con=int((diff>0).sum());ties=int((diff==0).sum());n=diff.size
                    rows.append(dict(analysis=name,seed=seed,boundary=b,pair_type='within' if a==c else 'between',positive_study=a,negative_study=c,N_positive=len(pos),N_negative=len(neg),N_pairs=n,concordant=con,discordant=int((diff<0).sum()),ties=ties,pairwise_concordance=(con+.5*ties)/n,weighted_contribution_to_pooled_auc=(con+.5*ties)/total))
            cells=pd.DataFrame(rows);allrows.extend(rows)
            within=cells[cells.pair_type=='within'];between=cells[cells.pair_type=='between']
            auc=roc_auc_score(y,p);wc=within.weighted_contribution_to_pooled_auc.sum();bc=between.weighted_contribution_to_pooled_auc.sum()
            assert abs(auc-wc-bc)<1e-14
            comps[b]=(auc,wc,bc)
            summaries.append(dict(analysis=name,seed=seed,boundary=b,pooled_AUROC=auc,within_component=wc,between_component=bc,within_pair_count=within.N_pairs.sum(),between_pair_count=between.N_pairs.sum(),weighted_within_AUROC=wc*total/within.N_pairs.sum(),macro_AUROC=within.pairwise_concordance.mean()))
            if seed!='aggregate':continue
            rp=np.empty(len(f))
            for s in STUDIES:
                idx=np.flatnonzero(f.study==s);local=p[idx];rp[idx]=(rankdata(local,method='average')-.5)/len(idx)
                for label in [0,1]:
                    v=local[y[idx]==label]
                    distributions.append(dict(analysis=name,boundary=b,study=s,label=label,N=len(v),mean_probability=v.mean(),median_probability=np.median(v),SD=np.std(v,ddof=1),IQR=np.quantile(v,.75)-np.quantile(v,.25),min=v.min(),max=v.max()))
                mt=local[y[idx]==1].mean();mn=local[y[idx]==0].mean()
                offsets.append(dict(analysis=name,boundary=b,study=s,mean_score_tumor=mt,mean_score_normal=mn,within_study_separation=mt-mn,overall_score_center=local.mean(),score_range=np.ptp(local)))
                cal=calibration(f.iloc[idx],local)
                quality.append(dict(analysis=name,boundary=b,study=s,Brier=np.mean((local-y[idx])**2),log_loss=np.mean(-y[idx]*np.log(np.clip(local,1e-15,1-1e-15))-(1-y[idx])*np.log(1-np.clip(local,1e-15,1-1e-15))),extreme_probability_fraction=np.mean((local<.01)|(local>.99)),**cal))
            rankrows.append(dict(analysis=name,boundary=b,AUROC=roc_auc_score(y,rp),status='POST_HOC_SCORE_COMPARABILITY_DIAGNOSTIC'))
        seedrows.append(dict(analysis=name,seed=seed,AUROC_V1=comps['V1'][0],AUROC_V2=comps['V2'][0],delta_total=comps['V1'][0]-comps['V2'][0],delta_within=comps['V1'][1]-comps['V2'][1],delta_between=comps['V1'][2]-comps['V2'][2]))
allcells=pd.DataFrame(allrows);summary=pd.DataFrame(summaries);seeds=pd.DataFrame(seedrows)
save(allcells,'auc_decomposition/ALL_PAIR_DECOMPOSITIONS.tsv')
primary=allcells[(allcells.analysis=='L2_rank')&(allcells.seed=='aggregate')]
save(primary,'auc_decomposition/AUC_PAIR_DECOMPOSITION.tsv')
save(summary,'auc_decomposition/WITHIN_STUDY_SUMMARY.tsv')
save(seeds,'score_comparability/SEED_AGGREGATION_AUDIT.tsv')
save(pd.DataFrame(distributions),'score_comparability/SCORE_DISTRIBUTIONS.tsv')
save(pd.DataFrame(offsets),'score_comparability/SCORE_OFFSETS.tsv')
save(pd.DataFrame(quality),'score_comparability/PROBABILITY_QUALITY.tsv')
save(pd.DataFrame(rankrows),'score_comparability/WITHIN_STUDY_RANK_DIAGNOSTIC.tsv')
mat={b:primary[primary.boundary==b].pivot(index='positive_study',columns='negative_study',values='pairwise_concordance').reindex(index=STUDIES,columns=STUDIES) for b in ['V1','V2']}
mat['DELTA']=mat['V1']-mat['V2']
for b,m in mat.items():m.to_csv(ROOT/f'auc_decomposition/{b}_CONCORDANCE_MATRIX.tsv',sep='\t')
# Original indices were not saved. New fixed explanatory seed as requested.
f=frames['L2_rank'];rng=np.random.default_rng(170330);weights=np.zeros((2000,len(f)))
for b in range(2000):weights[b]=np.bincount(patient_bootstrap_indices(f,rng),minlength=len(f))
np.savez_compressed(ROOT/'auc_decomposition/EXPLANATORY_BOOTSTRAP_WEIGHTS.npz',weights=weights,seed=170330,record_ids=f.record_id.to_numpy(dtype=str))
pos=np.flatnonzero(f.y==1);neg=np.flatnonzero(f.y==0);same=f.study.to_numpy()[pos,None]==f.study.to_numpy()[neg][None,:]
den=weights[:,pos].sum(1)*weights[:,neg].sum(1);boot={}
for boundary in ['V1','V2']:
    p=f[f'prob_{boundary}'].to_numpy();d=p[pos,None]-p[neg][None,:];con=(d>0)+.5*(d==0)
    for label,mask in [('within',same),('between',~same)]:boot[f'{boundary}_{label}']=np.sum((weights[:,pos]@(con*mask))*weights[:,neg],axis=1)/den
boot=pd.DataFrame(boot)
for label in ['within','between']:boot[f'delta_{label}']=boot[f'V1_{label}']-boot[f'V2_{label}']
boot['delta_total']=boot.delta_within+boot.delta_between
save(boot,'auc_decomposition/EXPLANATORY_BOOTSTRAP_2000.tsv')
ci=[]
for label in ['total','within','between']:
    lo,hi,n=interval(boot[f'delta_{label}']);r=seeds[(seeds.analysis=='L2_rank')&(seeds.seed=='aggregate')].iloc[0]
    ci.append(dict(component=label,estimate=r[f'delta_{label}'],CI_lower=lo,CI_upper=hi,seed=170330,status='POST_HOC_EXPLANATORY_BOOTSTRAP',B=n))
save(pd.DataFrame(ci),'auc_decomposition/DECOMPOSITION_SUMMARY.tsv')
# Diagnostic figures.
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,b in zip(axes,['V1','V2']):
    vals=[];labels=[]
    for s in STUDIES:
        for label in [0,1]:vals.append(f.loc[(f.study==s)&(f.y==label),f'prob_{b}']);labels.append(s+'\n'+('normal' if label==0 else 'tumor'))
    ax.boxplot(vals,tick_labels=labels,showfliers=True);ax.set_title(b);ax.tick_params(axis='x',labelsize=7);ax.set_ylabel('OOF probability')
fig.tight_layout();fig.savefig(ROOT/'figures/diagnostic_A.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(mat['DELTA'],cmap='coolwarm',vmin=-.2,vmax=.2)
ax.set_xticks(range(3),STUDIES);ax.set_yticks(range(3),STUDIES);ax.set_xlabel('Negative study');ax.set_ylabel('Positive study')
for i in range(3):
    for j in range(3):ax.text(j,i,f'{mat["DELTA"].iloc[i,j]:.4f}',ha='center')
fig.colorbar(im,ax=ax,label='V1 minus V2 concordance');fig.tight_layout();fig.savefig(ROOT/'figures/diagnostic_B.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots();cs=pd.DataFrame(ci);ax.bar(cs.component,cs.estimate);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('AUROC contribution difference V1 - V2');fig.tight_layout();fig.savefig(ROOT/'figures/diagnostic_C.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots();ss=seeds[(seeds.seed=='aggregate')&(seeds.analysis.isin(['L2_rank','source_scaled']))].set_index('analysis');ss[['delta_total','delta_within','delta_between']].plot.bar(ax=ax);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('AUROC contribution difference');fig.tight_layout();fig.savefig(ROOT/'figures/diagnostic_D.png',dpi=160);plt.close(fig)
print(seeds[seeds.seed=='aggregate'].to_string(index=False));print(pd.DataFrame(ci).to_string(index=False))
