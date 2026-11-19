"""Render frozen aggregate summaries with revised display and vector exports."""
from pathlib import Path
import csv
import json
import os

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'qa/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'axes.linewidth':.7,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE,TEAL,RED,GRAY,PURPLE='#0F4D92','#287D83','#B64342','#767676','#6B5A8E'
ORDER=['GSE44076','GSE37182','GSE50760']
PLOTTED=[]

def read(name):
 with (ROOT/'source_data'/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))

def one(rows,**criteria):
 selected=[r for r in rows if all(r.get(k)==v for k,v in criteria.items())]
 assert len(selected)==1,criteria
 return selected[0]

def fmt(value,signed=False,digits=4):
 x=float(value)
 if x==0:return '0'
 s=f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'
 return s.replace('-', '−')

def interval(ax,row,y,color=BLUE,marker='o',fields=('estimate','CI_lower','CI_upper'),identity='original paired patient-bootstrap CI'):
 x,lo,hi=[float(row[k]) for k in fields]
 assert lo<=x<=hi
 ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,ms=4.5,capsize=3,lw=1.15,mew=.9)
 PLOTTED.append(dict(figure=plt.gcf().get_label(),estimate=x,lower=lo,upper=hi,interval_identity=identity,source_row=row))
 return x

def zero(ax):ax.axvline(0,color='#666666',lw=.8,ls='--',zorder=0)

def heading(fig,x,y,text):fig.text(x,y,text,ha='left',va='top',fontsize=10,fontweight='bold')

def save(fig,n):
 for extension in ['pdf','png','eps','svg']:
  fig.savefig(OUT/f'figure_{n}.{extension}',dpi=600 if extension=='png' else None)
 plt.close(fig)

metrics=read('PRIMARY_METRICS.tsv');contrasts=read('PRIMARY_CONTRASTS.tsv')
studies=read('PER_STUDY_BOUNDARY_COMPARISON.tsv');decomp=read('DECOMPOSITION_SUMMARY.tsv')
primary=one(contrasts,scope='POOLED',metric='AUROC',contrast='study_boundary')

# Illustrative assignments use one distinct symbol per tissue record.
fig=plt.figure(figsize=(6.67,3.8),num='Figure 1')
fig.text(.5,.985,'Same evaluation records across validation schemes',ha='center',va='top',fontweight='bold',fontsize=10)
fig.text(.5,.93,'261 tissue records · 131 patients · 17,819 genes',ha='center',va='top',fontsize=9)
titles=['V0  Record-level','V1  Patient-grouped','V2  Study-held-out']
foot=['V0: Patient overlap permitted','V1: Patients separated','V2: Studies separated']
for boundary in range(3):
 ax=fig.add_axes([.035+boundary*.325,.28,.29,.51]);ax.set(xlim=(-.4,3.35),ylim=(-.4,2.9));ax.axis('off')
 ax.set_title(titles[boundary],fontweight='bold',pad=11,fontsize=9.5)
 for s,study in enumerate(ORDER):
  y=2.15-s*1.02
  ax.text(-.35,y+.32,study,fontsize=8.5)
  if boundary==2:ax.add_patch(Rectangle((-.13,y-.23),3.26,.48,fill=False,ec=RED if s==2 else BLUE,lw=.8))
  for patient in range(4):
   x=.15+patient*.88
   states=([patient==0,patient==1] if boundary==0 else [patient==0]*2 if boundary==1 else [s==2]*2)
   ax.plot([x,x],[y-.12,y+.12],color='#999999',lw=.8,zorder=1)
   for held,dy,marker in zip(states,[-.12,.12],['o','^']):
    ax.scatter(x,y+dy,s=24,marker=marker,facecolors='white' if held else BLUE,edgecolors=RED if held else BLUE,linewidths=1,zorder=2)
 fig.text(.18+boundary*.325,.23,foot[boundary],ha='center',fontsize=8.3)
fig.text(.5,.15,'Blue filled = training     Red open = held out',ha='center',fontsize=9)
fig.text(.5,.10,'○ normal     △ tumor     Line = same patient',ha='center',fontsize=9)
fig.text(.5,.035,'One outer split shown schematically. Symbol counts do not represent sample sizes.',ha='center',fontsize=8)
save(fig,1)

fig=plt.figure(figsize=(6.67,3.0),num='Figure 2')
a=fig.add_axes([.075,.32,.30,.47]);b=fig.add_axes([.63,.32,.245,.47])
heading(fig,.035,.95,'A  Pooled AUROC');heading(fig,.54,.95,'B  Paired AUROC contrasts')
for y,boundary,color in [(2,'V0',GRAY),(1,'V1',BLUE),(0,'V2',TEAL)]:
 row=one(metrics,scope='POOLED',metric='AUROC',boundary=boundary)
 x=interval(a,row,y,color)
 a.text(1.06,y,fmt(x),transform=a.get_yaxis_transform(),va='center',fontsize=8.8)
a.set(yticks=[2,1,0],yticklabels=['V0','V1','V2'],ylim=(-.5,2.5),xlim=(.95,1.001),xticks=[.95,.975,1],xlabel='Pooled AUROC')
for y,name,color in [(1,'patient_boundary',GRAY),(0,'study_boundary',BLUE)]:
 row=one(contrasts,scope='POOLED',metric='AUROC',contrast=name)
 x=interval(b,row,y,color);b.text(1.08,y,fmt(x,True),transform=b.get_yaxis_transform(),va='center',fontsize=8.8)
zero(b);b.set(yticks=[1,0],yticklabels=['V0−V1\nsecondary','V1−V2\nprimary'],ylim=(-.5,1.5),xlim=(-.005,.025),xticks=[0,.01,.02],xlabel='Paired ΔAUROC')
fig.text(.5,.13,'Estimates use probabilities averaged across three repeats.',ha='center',fontsize=8.5)
fig.text(.5,.055,'Bars show paired patient-bootstrap 95% intervals.',ha='center',fontsize=8.5)
save(fig,2)

fig=plt.figure(figsize=(6.67,3.2),num='Figure 3')
a=fig.add_axes([.18,.29,.24,.48]);b=fig.add_axes([.69,.29,.19,.48])
heading(fig,.18,.95,'A  Study-specific AUROC');heading(fig,.65,.95,'B  Paired V1−V2')
fig.text(.49,.79,'V1',ha='center',fontsize=8.5,fontweight='bold');fig.text(.58,.79,'V2',ha='center',fontsize=8.5,fontweight='bold')
labels=[]
for y,study in zip([2,1,0],ORDER):
 r=one(studies,study=study);x1,x2=float(r['AUROC_V1']),float(r['AUROC_V2'])
 labels.append(f"{study}\n{r['N_patients']} patients")
 a.plot([x1,x2],[y+.09,y-.09],color='#B0B0B0',lw=.8)
 a.scatter(x1,y+.09,s=22,color=BLUE,marker='o',zorder=3)
 a.scatter(x2,y-.09,s=22,facecolors='white',edgecolors=TEAL,marker='D',linewidths=1,zorder=3)
 for xpos,val in [(.49,x1),(.58,x2)]:fig.text(xpos,.29+.48*(y+.5)/3,fmt(val),ha='center',va='center',fontsize=8.5)
 x=interval(b,r,y,fields=('delta_AUROC','delta_AUROC_CI_lower','delta_AUROC_CI_upper'))
 b.text(1.10,y,fmt(x,True),transform=b.get_yaxis_transform(),va='center',fontsize=8.5)
a.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.5,2.5),xlim=(.87,1.007),xticks=[.90,.95,1],xlabel='Study-specific AUROC')
zero(b);b.set(yticks=[],ylim=(-.5,2.5),xlim=(-.18,.045),xticks=[-.15,0],xlabel='ΔAUROC')
fig.text(.5,.125,'● V1 patient-grouped     ◇ V2 study-held-out',ha='center',fontsize=8.5)
summary=read('WITHIN_STUDY_SUMMARY.tsv')
v1=one(summary,analysis='L2_rank',seed='aggregate',boundary='V1');v2=one(summary,analysis='L2_rank',seed='aggregate',boundary='V2')
macro=float(v1['macro_AUROC'])-float(v2['macro_AUROC'])
fig.text(.5,.055,f'Descriptive macro difference {fmt(macro,True)}. Each study receives equal weight.',ha='center',fontsize=8.5)
save(fig,3)

fig=plt.figure(figsize=(6.67,5.55),num='Figure 4')
a=fig.add_axes([.18,.55,.23,.32]);b=fig.add_axes([.64,.55,.28,.32])
heading(fig,.035,.95,'A  Weighted contributions');heading(fig,.56,.95,'B  Study-pair concordance change')
for y,component,color in [(2,'total',BLUE),(1,'within',TEAL),(0,'between',RED)]:
 r=primary if component=='total' else one(decomp,component=component)
 identity='original prespecified primary CI' if component=='total' else 'post-primary explanatory CI'
 x=interval(a,r,y,color,marker='s' if component=='total' else 'o',identity=identity)
 a.text(1.05,y,fmt(x,True),transform=a.get_yaxis_transform(),va='center',fontsize=8.5)
zero(a);a.set(yticks=[2,1,0],yticklabels=['Total\nprimary CI','Within\nexplanatory CI','Between\nexplanatory CI'],ylim=(-.5,2.5),xlim=(-.005,.027),xticks=[0,.01,.02],xlabel='Weighted contribution to ΔAUROC,\nV1−V2')
rows=read('DELTA_CONCORDANCE_MATRIX.tsv')
matrix=np.array([[float(one(rows,positive_study=s)[c]) for c in ORDER] for s in ORDER])
norm=matplotlib.colors.Normalize(vmin=-.14,vmax=.14)
cmap=plt.get_cmap('RdBu_r')
im=matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
for row_index in range(3):
 for column_index in range(3):
  b.add_patch(Rectangle((column_index-.5,row_index-.5),1,1,facecolor=cmap(norm(matrix[row_index,column_index])),edgecolor='none'))
b.set(xlim=(-.5,2.5),ylim=(2.5,-.5),aspect='equal')
b.set_xticks(range(3),ORDER,rotation=25,ha='right',fontsize=8);b.set_yticks(range(3),ORDER,fontsize=8)
b.set_xlabel('Normal-record study',fontsize=8);b.set_ylabel('Tumor-record study',fontsize=8,labelpad=1)
for r in range(3):
 for c in range(3):
  b.text(c,r,fmt(matrix[r,c],True),ha='center',va='center',fontsize=8.2,color='white' if abs(matrix[r,c])>.095 else '#222222')
  if r==c:b.add_patch(Rectangle((c-.5,r-.5),1,1,fill=False,ec='#222222',lw=1.2))
cb=fig.colorbar(im,cax=fig.add_axes([.665,.425,.23,.015]),orientation='horizontal',ticks=[-.1,0,.1]);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.ax.tick_params(labelsize=7,pad=1);cb.set_label('Δ concordance, V1−V2',fontsize=8,labelpad=1)
pairs=read('AUC_PAIR_DECOMPOSITION.tsv')
x1=one(pairs,analysis='L2_rank',seed='aggregate',boundary='V1',positive_study='GSE50760',negative_study='GSE44076')
x2=one(pairs,analysis='L2_rank',seed='aggregate',boundary='V2',positive_study='GSE50760',negative_study='GSE44076')
conditional=float(x1['pairwise_concordance'])-float(x2['pairwise_concordance'])
weighted=float(x1['weighted_contribution_to_pooled_auc'])-float(x2['weighted_contribution_to_pooled_auc'])
assert abs(conditional*int(x1['N_pairs'])/17030-weighted)<1e-14
fig.text(.5,.345,'GSE50760 tumor × GSE44076 normal',ha='center',fontsize=8.5,fontweight='bold')
fig.text(.5,.317,f'Δ concordance: {fmt(conditional,True,6)}    Weighted contribution: {fmt(weighted,True,6)}',ha='center',fontsize=8.5)
heading(fig,.035,.275,'C  Score-shift illustration')
for column,offset,title in [(0,0,'Before study B score shift'),(1,.40,'After study B score shift')]:
 ax=fig.add_axes([.14+column*.48,.105,.31,.115]);ax.set(xlim=(0,1),ylim=(-.3,1.3),yticks=[1,0],yticklabels=['Study A','Study B'],xticks=[])
 ax.set_title(title,loc='left',fontsize=8.5,pad=5)
 for y,n,t,color in [(1,.35,.60,BLUE),(0,.25+offset,.50+offset,RED)]:
  ax.plot([n,t],[y,y],color=color,lw=1);ax.scatter(n,y,marker='o',s=22,color=color,zorder=3);ax.scatter(t,y,marker='^',s=24,color=color,zorder=3)
 ax.plot([.60,.25+offset],[1,0],ls='--',color='#888888',lw=.8)
 ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=8)
 ax.set_xlabel('Schematic score →',fontsize=8,labelpad=2)
fig.text(.5,.045,'Within-study ordering is preserved; cross-study ordering changes.',ha='center',fontsize=8.5)
fig.text(.5,.015,'Conceptual illustration; no empirical estimates.    ○ normal    △ tumor',ha='center',fontsize=8)
save(fig,4)

fig=plt.figure(figsize=(6.67,4.4),num='Figure 5')
a=fig.add_axes([.36,.31,.22,.47]);b=fig.add_axes([.67,.31,.20,.47])
heading(fig,.36,.95,'A  Full scale');heading(fig,.66,.95,'B  Zoom −0.02 to +0.055')
specs=[('Primary','T2 · rank · L2',primary,BLUE)]
for title,subtitle,file,color in [('Elastic Net','model sensitivity','ELASTIC_NET_RESULTS.tsv',TEAL),('Training matched','matched training contrast','TRAINING_SIZE_MATCHED_RESULTS.tsv',GRAY),('Source scale','representation sensitivity','SOURCE_SCALED_RESULTS.tsv',RED),('T1','case-mix sensitivity','T1_SENSITIVITY_RESULTS.tsv',PURPLE)]:
 specs.append((title,subtitle,one(read(file),scope='POOLED',metric='AUROC',kind='contrast',contrast='study_boundary'),color))
for y,(title,subtitle,row,color) in zip(range(4,-1,-1),specs):
 x=interval(a,row,y,color)
 a.text(-.10,y,title+'\n'+subtitle,transform=a.get_yaxis_transform(),ha='right',va='center',fontsize=8.5)
 b.text(1.13,y,fmt(x,True),transform=b.get_yaxis_transform(),ha='left',va='center',fontsize=8.5)
 if y==1:
  # The arrow denotes an off-scale estimate, never an empirical endpoint.
  b.annotate('',xy=(.052,y),xytext=(.040,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.3})
  b.text(-.017,y+.20,'Off-scale →',color=color,fontsize=8)
 else:interval(b,row,y,color)
for ax in [a,b]:
 zero(ax);ax.axhline(.5,color='#C4C4C4',lw=.6);ax.set(yticks=[],ylim=(-.55,4.55),xlabel='V1−V2 ΔAUROC')
a.set(xlim=(-.02,.22),xticks=[0,.1,.2]);b.set(xlim=(-.02,.055),xticks=[-.02,0,.02,.04])
fig.text(.5,.17,'T1: 311 records / 181 patients · 50 healthy normals from GSE44076',ha='center',fontsize=8.5)
fig.text(.5,.125,'T2 rows, including matched: 261 records / 131 patients per arm.',ha='center',fontsize=8.5)
fig.text(.5,.065,'Estimates use probabilities averaged across three repeats.',ha='center',fontsize=8.5)
fig.text(.5,.02,'Bars show original paired patient-bootstrap 95% intervals.',ha='center',fontsize=8.5)
save(fig,5)
(ROOT/'source_data/PLOTTED_VALUES.json').write_text(json.dumps(PLOTTED,indent=2))
(ROOT/'source_data/FIGURE4_MAPPING.json').write_text(json.dumps({'positive_study':'GSE50760','negative_study':'GSE44076','delta_concordance':conditional,'weighted_contribution':weighted,'N_pairs':int(x1['N_pairs']),'total_pairs':17030,'formula':'delta_concordance * N_pairs / total_pairs','primary_total_CI':[primary['CI_lower'],primary['CI_upper']],'explanatory_total_CI':one(decomp,component='total'),'supplement_location':'supplementary_methods_v6.md, Table S8, DECOMPOSITION_SUMMARY.tsv block','macro_difference':macro},indent=2))
print('Five figures exported as vector PDF/EPS/SVG and 600-dpi PNG.')
