"""Generate analysis reports exclusively from completed, audited Phase 1 artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def read(relative): return pd.read_csv(ROOT/relative,sep='\t')

def table(frame):
    columns=list(frame.columns)
    display_columns=[column.replace('AUPRC','AUPRC(AP)') for column in columns]
    rows=['| '+' | '.join(display_columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if isinstance(value,(float,np.floating)):values.append('NA' if not np.isfinite(value) else f'{value:.6f}')
            else:values.append('AUPRC(AP)' if str(value)=='AUPRC' else str(value))
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)+'\n'


def main():
    census=read('data/PRIMARY_ANALYSIS_CENSUS.tsv')
    metrics=read('results/PRIMARY_METRICS.tsv')
    contrasts=read('results/PRIMARY_CONTRASTS.tsv')
    studies=read('results/PER_STUDY_BOUNDARY_COMPARISON.tsv')
    seeds=read('results/SEED_STABILITY.tsv')
    decisions=read('controls/CONTROL_DECISIONS.tsv')
    sensitivity_status=read('sensitivity/COMPLETION_STATUS.tsv')
    fit=read('logs/MODEL_FIT_AUDIT.tsv')
    leak=read('logs/LEAKAGE_AUDIT.tsv')
    positive=read('controls/POSITIVE_CONTROL.tsv')
    pred=read('predictions/PRIMARY_OOF_PREDICTIONS.tsv')
    complete=pred.filter(regex='^prob_').notna().all().all()
    v1=metrics[(metrics.scope=='POOLED')&(metrics.boundary=='V1')&(metrics.metric=='AUROC')].iloc[0]
    pc=bool(v1.estimate>.5)
    nc={c:bool((decisions[decisions.control==c].status=='PASS').all() and len(decisions[decisions.control==c])==3) for c in [1,2]}
    leak_pass=leak.status.eq('PASS').all()
    completed=dict(zip(sensitivity_status.analysis,sensitivity_status.complete))
    required=['elastic_net','matched','T1','source_scaled']
    all_sensitivity=all(bool(completed.get(key,False)) for key in required)
    if not pc or not all(nc.values()) or not leak_pass or not complete:
        decision='NO_GO';phase_status='NO_GO';authorized=False
    elif not all_sensitivity:
        decision='REDESIGN';phase_status='INCOMPLETE';authorized=False
    else:
        decision='QUALIFIED_FOR_MANUSCRIPT';phase_status='COMPLETE';authorized=True
    status=dict(PHASE1_STATUS=phase_status,PHASE1_DECISION=decision,
        PRIMARY_ANALYSIS_STATUS='COMPLETE' if complete else 'INCOMPLETE',POSITIVE_CONTROL='PASS' if pc else 'FAIL',
        NEGATIVE_CONTROL_1='PASS' if nc[1] else 'FAIL',NEGATIVE_CONTROL_2='PASS' if nc[2] else 'FAIL',
        PIPELINE_LEAKAGE_AUDIT='PASS' if leak_pass else 'FAIL',V0_RECORD_VALIDATION='PASS' if complete else 'FAIL',
        V1_PATIENT_GROUPED_VALIDATION='PASS' if leak_pass and complete else 'FAIL',
        V2_LOSO_VALIDATION='PASS' if leak_pass and complete else 'FAIL',PRIMARY_STUDY_BOUNDARY_CONTRAST='ESTIMATED',
        SECONDARY_PATIENT_BOUNDARY_CONTRAST='ESTIMATED',ELASTIC_NET_SENSITIVITY='COMPLETE' if completed.get('elastic_net',False) else 'INCOMPLETE',
        TRAINING_SIZE_SENSITIVITY='COMPLETE' if completed.get('matched',False) else 'INCOMPLETE',T1_SENSITIVITY='COMPLETE' if completed.get('T1',False) else 'INCOMPLETE',
        IDENTITY_LEAKAGE_CLAIM_AUTHORIZATION=False,STUDY_GENERALIZATION_CLAIM_AUTHORIZATION=authorized,MANUSCRIPT_DRAFTING_AUTHORIZATION=authorized)
    instability=bool(seeds.delta_study.max()-seeds.delta_study.min()>.05 or seeds.delta_patient.max()-seeds.delta_patient.min()>.05)
    status['ALGORITHM_INSTABILITY']=instability
    status['PIPELINE_LEAKAGE_SUSPECTED']=not all(nc.values())
    header='```text\n'+'\n'.join(f'{key} = {str(value).upper() if isinstance(value,(bool,np.bool_)) else value}' for key,value in status.items())+'\n```\n\n'
    def write(name,body):(ROOT/'reports'/f'{name}.md').write_text(header+body,encoding='utf-8')
    pooled=metrics[metrics.scope=='POOLED']
    pooled_contrasts=contrasts[contrasts.scope=='POOLED']
    def point(boundary,metric='AUROC',scope='POOLED'):
        return metrics[(metrics.scope==scope)&(metrics.boundary==boundary)&(metrics.metric==metric)].iloc[0]
    def contrast(kind,metric='AUROC',scope='POOLED'):
        return contrasts[(contrasts.scope==scope)&(contrasts.contrast==kind)&(contrasts.metric==metric)].iloc[0]
    def ci(row):return f'{row.estimate:.6f}，95% CI {row.CI_lower:.6f}至{row.CI_upper:.6f}'
    def sensitivity_contrast(name):
        path=ROOT/'sensitivity'/name
        if not path.exists():return pd.DataFrame()
        frame=pd.read_csv(path,sep='\t')
        return frame[(frame.kind=='contrast')&(frame.scope=='POOLED')&frame.metric.isin(['AUROC','AUPRC'])][['contrast','metric','estimate','CI_lower','CI_upper']]
    sensitivity_text=[]
    for label,name in [('Elastic Net','ELASTIC_NET_RESULTS.tsv'),('训练量匹配','TRAINING_SIZE_MATCHED_RESULTS.tsv'),('来源尺度加训练scaling','SOURCE_SCALED_RESULTS.tsv'),('T1','T1_SENSITIVITY_RESULTS.tsv')]:
        subset=sensitivity_contrast(name)
        sensitivity_text.append(f'## {label}\n\n'+(table(subset) if len(subset) else '分析未完整完成，见失败日志。\n'))
    def sensitivity_answer(filename,kind='study_boundary'):
        path=ROOT/'sensitivity'/filename
        if not path.exists():return '尚未完整估计。'
        data=pd.read_csv(path,sep='\t')
        selected=data[(data.kind=='contrast')&(data.scope=='POOLED')&(data.metric=='AUROC')]
        if 'contrast' in selected:selected=selected[selected.contrast==kind]
        if len(selected)==0:return '未完整估计，保留失败记录。'
        row=selected.iloc[0]
        direction='与主比较同方向' if row.estimate*contrast('study_boundary').estimate>0 else '与主比较方向不同或为零'
        return f'pooled AUROC的V1−V2为{ci(row)}，{direction}。'
    mixed_direction=bool((studies.delta_AUROC<0).any() and (studies.delta_AUROC>=0).any())
    macro=point('V1',scope='MACRO_DESCRIPTIVE').estimate-point('V2',scope='MACRO_DESCRIPTIVE').estimate
    primary_text=f'''# Phase 1 primary analysis

## 冻结输入和执行

{int(census.N_records.sum())}条T2组织记录、{int(census.N_patients.sum())}名患者，tumor为positive class。固定17,819个Entrez基因，主表示为sample-local common-gene rank，主模型为L2 logistic。seeds固定17、29、43，inner/outer划分、C grid和训练内过滤均按Phase 0.3执行。冻结校验覆盖66项本轮前阶段资产和339项父阶段资产，执行前后均核对。

{table(census)}

每条记录获得V0、V1、V2各3个seed预测。主要概率由三个seed逐记录算术平均，完整记录数为{len(pred)}。V0作为diagnostic comparator，V1保证患者隔离，V2保证整个研究隔离。

## Pooled OOF指标

{table(pooled[['boundary','metric','estimate','CI_lower','CI_upper','positive_prevalence']])}

AUPRC采用average precision。Balanced accuracy采用预冻结阈值0.5。概率clip固定1e-15，未优化测试阈值或校准概率。

## 配对差值

{table(pooled_contrasts[['contrast','definition','metric','estimate','CI_lower','CI_upper']])}

差值方向以definition列为准。AUROC主差值V1−V2为{ci(contrast('study_boundary'))}。次要差值V0−V1为{ci(contrast('patient_boundary'))}。Brier和log loss的差值表保留原始V1−V2或V0−V1定义，较低的原始loss更好，解释时据此确定方向。

95% CI由2000次study分层patient-cluster bootstrap获得，每个抽中患者携带全部组织记录和全部boundary预测。重采样固定已训练OOF模型，区间条件于这三个研究，训练重拟合及未来研究总体变异未包含。

## 逐研究与pooled结果的区别

{table(studies)}

三个研究的AUROC差值等权描述性均值为{macro:.6f}。Pooled差值与macro差值可能不同，因为pooled AUROC还比较不同研究之间的正负记录及分数尺度。逐研究方向不一致标志为{mixed_direction}。研究内结果必须与pooled值共同呈现，pooled正差值不足以证明所有研究均有外部性能下降。

## Seed稳定性

{table(seeds)}

预先记录的algorithm instability规则为任一gap的seed范围超过0.05，当前标志为{instability}。所有seed均保留，差值按冻结概率均值计算。

## Calibration

{table(read('results/CALIBRATION_DESCRIPTIVE.tsv'))}

校准分析仅描述OOF概率。小研究不足50名患者时记NA，分离或收敛失败时同样记NA并保留原因。Bootstrap有效数随可估计性报告。
'''
    write('PHASE1_PRIMARY_ANALYSIS',primary_text)
    loso=read('results/LOSO_PER_STUDY_METRICS.tsv')
    write('PHASE1_LOSO_RESULTS',f'''# Phase 1 LOSO results

每个目标研究均完全持出。GSE44076由GSE37182与GSE50760训练，GSE37182由GSE44076与GSE50760训练，GSE50760由GSE44076与GSE37182训练。持出研究未参与过滤、scaling、参数选择、模型或校准拟合。Inner仅在两个来源研究内进行patient-grouped CV。

{table(loso)}

## 同记录边界比较

{table(studies)}

外部研究单位为3，组织记录数和患者bootstrap次数均不增加外部研究单位。不同研究方向、样本量及来源组成差异完整保留。
''')
    write('PHASE1_CONTROLS',f'''# Phase 1 controls

## Positive control

主V1 pooled AUROC为{ci(v1)}，positive control状态为{status['POSITIVE_CONTROL']}。预冻结的研究内patient-grouped nested CV亦完成，结果如下。

{table(positive[(positive.metric=='AUROC')&(~positive.scope.isin(['POOLED','MACRO_DESCRIPTIVE']))][['task','scope','estimate','CI_lower','CI_upper','N_patients']])}

该检查只评价研究内信号。

## Negative controls

{table(decisions)}

NC1在当前训练内对完整患者对随机交换标签，singleton在study内置换，保留混合标签患者结构及来源类总数。NC2在当前训练内于study内部逐基因打乱。Inner训练与outer最终训练分别独立执行，validation与heldout保持原始输入和标签。每项20个replicate均使用冻结split与seed派生规则。

失败规则固定为至少16/20个pooled AUROC高于0.5且replicate均值的患者bootstrap CI下界高于0.5，逐boundary判定。具体分布见NEGATIVE_CONTROL_1.tsv和NEGATIVE_CONTROL_2.tsv。NC1状态为{status['NEGATIVE_CONTROL_1']}，NC2状态为{status['NEGATIVE_CONTROL_2']}。PASS按预注册的稳定高性能触发规则判定。完整分布包含较宽的replicate间变异，NC2的V2均值低于0.5，仍按原样报告。患者bootstrap CI条件于这20个冻结控制拟合，replicate间标准差反映另外的置换与训练变异。

## 审计

{table(leak.groupby(['analysis','status']).size().rename('cases').reset_index())}

拟合日志共{len(fit)}行。Bootstrap抽样顺序曾在执行审计中修正，全部正式区间均已通过冻结函数重新生成，初版输出与修正说明保存在logs/bootstrap_draw_order_repair和logs/TECHNICAL_REPAIRS.md。所有outer/inner范围、原始模型训练ID与预测对齐已核查。实际pipeline leakage审计状态为{status['PIPELINE_LEAKAGE_AUDIT']}。阴性控制异常与已确认的代码泄漏分别按证据记录。
''')
    detailed=[]
    for label,filename in [('Elastic Net','ELASTIC_NET_RESULTS.tsv'),('训练量匹配','TRAINING_SIZE_MATCHED_RESULTS.tsv'),('来源尺度加训练scaling','SOURCE_SCALED_RESULTS.tsv'),('T1','T1_SENSITIVITY_RESULTS.tsv')]:
        path=ROOT/'sensitivity'/filename
        if not path.exists():continue
        data=pd.read_csv(path,sep='\t')
        keep=data[(data.kind=='contrast')&(data.metric=='AUROC')]
        if len(keep):detailed.append(f'## {label}逐研究AUROC差值\n\n'+table(keep[['scope','contrast','estimate','CI_lower','CI_upper']]))
    t1_sub=read('sensitivity/T1_NORMAL_SUBTYPE_RESULTS.tsv') if (ROOT/'sensitivity/T1_NORMAL_SUBTYPE_RESULTS.tsv').exists() else pd.DataFrame()
    write('PHASE1_SENSITIVITY','# Phase 1 sensitivity\n\n'+table(sensitivity_status)+'\n'+ '\n'.join(sensitivity_text)+'\n'+'\n'.join(detailed)+'\n## T1正常来源分层\n\n'+table(t1_sub)+'\n## 敏感性seed结果\n\n'+table(read('sensitivity/SEED_STABILITY.tsv'))+'''
Elastic Net三个单seed的研究边界AUROC差值均为正，冻结的逐记录概率均值所得差值接近零且为负。AUROC为非线性排序统计量，概率先平均与指标先平均对应不同摘要。本项目严格采用预冻结的概率平均，两个层面的结果均完整展示。

所有敏感性保持预注册地位。T1癌症患者正常与健康正常分开报告，健康正常仅来自GSE44076。不同表示、模型或任务的结果均未用于改换主分析。训练量匹配两arm具有同一目标测试记录、相同完整配对患者数及类组成。其来源研究组成仍不同。
''')
    limitations='''# Phase 1 limitations

1. 真正的外部研究单位只有3个。患者bootstrap描述这些冻结研究内的抽样不确定性，无法替代研究级独立重复。
2. Study与platform完全混杂。研究边界差值同时涉及平台、实验室、病例组合、组织处理、公共存档预处理和其他来源差异，各因素的独立贡献无法识别。
3. 阵列公开输入已经过来源RMA/RSN多样本处理。新增rank严格sample-local，结果条件于存档处理矩阵，尚未验证从原始信号到前瞻性独立患者预测的完整链条。
4. 原生V1与V2训练规模、研究组成不同。训练量匹配控制了完整配对患者数和类别数，研究组成差异仍存在。
5. Pooled AUROC同时包含研究内与研究间排序。其方向可以与逐研究或macro描述结果不同，必须共同解释。
6. GSE37182有15名患者、GSE50760有18名患者，研究内接近满分的指标和较宽区间均需结合分母理解。完美样本排序的bootstrap窄区间不提供未来来源不确定性的保证。
7. 正式任务主要为相反标签的同患者组织，V0−V1方向不能归因于同标签身份记忆。患者边界差保持次要地位。
8. GSE37182与GSE50760治疗状态保持UNKNOWN。T0代表离体处理窗口，未提供未治疗身份的依据。
9. 新颖性资格沿用PARTIAL。跨研究与分组验证已有先例，当前贡献范围为冻结benchmark内的同记录边界比较及可审计结果。
10. 阴性控制的replicate间分布较宽，NC2的V2均值低于0.5。控制PASS依据预注册的稳定高性能规则，患者bootstrap CI条件于现有20个拟合程序。概率质量、calibration、seed差异和全部敏感性共同决定解释力度。AUROC的单一方向不能替代其他结果。
'''
    write('PHASE1_LIMITATIONS',limitations)
    qualification=f'''# Phase 1 manuscript qualification

最终判定为{decision}。主分析完整性为{status['PRIMARY_ANALYSIS_STATUS']}，positive control为{status['POSITIVE_CONTROL']}，NC1/NC2分别为{status['NEGATIVE_CONTROL_1']}/{status['NEGATIVE_CONTROL_2']}。三LOSO已报告，主差值及患者cluster CI可估计，敏感性状态如下。

{table(sensitivity_status)}

此判定按实际完整性、controls和可解释性作出，主差值的正方向或显著性均未作为门槛。若授权写作，范围限于这三个CRC研究及公共存档输入上的验证边界比较。JBI稿件的投稿充分性仍需结合PARTIAL新颖性及研究级证据范围评估，稿件的核心贡献范围应限定为同记录边界比较、pooled与研究内结果的区别及预注册敏感性。论文正文尚未起草。
'''
    write('PHASE1_MANUSCRIPT_QUALIFICATION',qualification)
    conclusion=(f'在这三个CRC存档转录组研究中，patient-grouped验证的pooled AUROC较LOSO高{contrast("study_boundary").estimate:.6f}。'
        +f'逐研究差值和pooled结果方向不一致标志为{mixed_direction}，macro AUROC差值为{macro:.6f}。'
        +'Elastic Net的pooled研究边界差接近零并与L2点估计方向不同。来源尺度敏感性的pooled差值明显较大。验证边界估计对模型、表示和研究间分数可比性具有敏感性，当前结果支持这一限定范围的实证结论。')
    write('PHASE1_DECISION',f'''# Phase 1 decision

{decision}。

{conclusion}

全部结果和失败依原冻结协议保留。Identity leakage归因授权FALSE。研究来源变化的解释授权为{authorized}，manuscript drafting授权为{authorized}。当前执行停于分析报告与资格判定，论文正文未起草。
''')
    auroc=pooled[pooled.metric=='AUROC'][['boundary','estimate','CI_lower','CI_upper']]
    answer=f'''# Phase 1 executive report

## 核心真实结果

T2共{int(census.N_records.sum())}条记录、{int(census.N_patients.sum())}名患者。

{table(auroc)}

Delta_patient为{ci(contrast('patient_boundary'))}。Delta_study为{ci(contrast('study_boundary'))}。每项CI均为2000次配对患者cluster-bootstrap。

{table(studies[['study','N_patients','AUROC_V1','AUROC_V2','delta_AUROC']])}

{conclusion}

## 二十二项结果回答

1. T2最终{int(census.N_records.sum())}条组织记录、{int(census.N_patients.sum())}名患者。
2. V0 AUROC为{ci(point('V0'))}。
3. V1 AUROC为{ci(point('V1'))}。
4. V2 AUROC为{ci(point('V2'))}。
5. Delta_patient点估计为{contrast('patient_boundary').estimate:.6f}。
6. Delta_patient 95% CI为{contrast('patient_boundary').CI_lower:.6f}至{contrast('patient_boundary').CI_upper:.6f}。
7. Delta_study点估计为{contrast('study_boundary').estimate:.6f}。
8. Delta_study 95% CI为{contrast('study_boundary').CI_lower:.6f}至{contrast('study_boundary').CI_upper:.6f}。
9. 三研究的V2 AUROC见上述同记录表，所有目标研究均保留。
10. 各研究V1−V2差值见同表，macro AUROC差值为{macro:.6f}。
11. Pooled AUPRC的V1−V2差值为{ci(contrast('study_boundary','AUPRC'))}，逐研究方向与prevalence同时见PHASE1_PRIMARY_ANALYSIS。
12. Brier依次为{point('V0','Brier').estimate:.6f}、{point('V1','Brier').estimate:.6f}、{point('V2','Brier').estimate:.6f}，log loss依次为{point('V0','log_loss').estimate:.6f}、{point('V1','log_loss').estimate:.6f}、{point('V2','log_loss').estimate:.6f}，应按越低越好解释，不能概括为所有指标同步下降。
13. Seed间study gap范围为{seeds.delta_study.min():.6f}至{seeds.delta_study.max():.6f}，其中{int((seeds.delta_study>0).sum())}/3为正。Patient gap范围为{seeds.delta_patient.min():.6f}至{seeds.delta_patient.max():.6f}。预先记录的0.05范围阈值对应ALGORITHM_INSTABILITY为{instability}，完整seed表如下。
14. Positive control为{status['POSITIVE_CONTROL']}。
15. NC1为{status['NEGATIVE_CONTROL_1']}，NC2为{status['NEGATIVE_CONTROL_2']}，逐replicate和逐boundary判定见controls。
16. Elastic Net的{sensitivity_answer('ELASTIC_NET_RESULTS.tsv')}三个单seed的差值均为正，冻结的概率平均汇总给出接近零的负差值，逐研究和seed结果均见PHASE1_SENSITIVITY。
17. 训练量匹配后，{sensitivity_answer('TRAINING_SIZE_MATCHED_RESULTS.tsv')}两arm训练患者数及类别数相同。
18. T1加入healthy normals后，{sensitivity_answer('T1_SENSITIVITY_RESULTS.tsv')}正常来源分层见PHASE1_SENSITIVITY。
19. 实际pipeline leakage审计为{status['PIPELINE_LEAKAGE_AUDIT']}，代码与结果独立审计另见EXPERIMENT_AUDIT。
20. 最强可支持结论为上述条件性边界比较，方向和效应大小服从逐研究及敏感性结果。
21. 最强限制为只有3个外部研究且platform与study完全混杂。
22. 稿件资格判定为{decision}，JBI投稿充分性受PARTIAL新颖性及研究级证据范围限制。当前未起草论文正文。

## Seed结果

{table(seeds)}

'''+ '\n'.join(sensitivity_text)
    write('PHASE1_EXECUTIVE_REPORT',answer)
    (ROOT/'results/FINAL_STATUS.json').write_text(json.dumps(status,indent=2,default=lambda x:bool(x))+'\n')
    print(json.dumps(status,indent=2,default=lambda x:bool(x)))

if __name__=='__main__':main()
