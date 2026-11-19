```text
PHASE0_3_STATUS = COMPLETE
PHASE0_3_DECISION = GO
PHASE1_AUTHORIZATION = TRUE
FORMAL_MODEL_TRAINING_AUTHORIZATION = TRUE
PRIMARY_TASK_T2_QUALIFICATION = PASS
CROSS_PLATFORM_QUALIFICATION = PASS
PATIENT_GROUPED_VALIDATION_QUALIFICATION = PASS
LOSO_VALIDATION_QUALIFICATION = PASS
NOVELTY_QUALIFICATION = PARTIAL
PRIMARY_CONTRAST_AUTHORIZATION = TRUE
SECONDARY_PATIENT_BOUNDARY_ANALYSIS_AUTHORIZATION = TRUE
IDENTITY_LEAKAGE_CLAIM_AUTHORIZATION = FALSE
STUDY_GENERALIZATION_CLAIM_AUTHORIZATION = TRUE
MANUSCRIPT_DRAFTING_AUTHORIZATION = FALSE
```

# Statistical analysis plan

## 估计量与记录对齐

以三个repeat的同记录OOF概率平均值作为每boundary的最终预测。主要估计量为pooled OOF AUROC_V1−AUROC_V2。次要估计量为AUROC_V0−AUROC_V1，方向开放。正负差值均原样报告。每条记录三种预测必须具有完全相同的record、patient、study和observed label。T2缺失任何boundary预测时先修复运行，禁止通过删除失败记录形成有利交集。

AUROC按组织预测计算。重采样按patient，混合标签的两条组织作为同一依赖块。患者ID跨研究具有唯一namespace，跨研究独立性依据前阶段provenance审计。外部研究统计单位只有3个。

## 固定bootstrap实现

采用study分层的paired patient-cluster bootstrap，B=2000，seed=170329。每个study分别有放回抽取原患者数的患者，抽中的患者携带该任务下全部组织记录，重复抽中时保留全部块的相应倍数。各boundary使用完全相同的抽样记录索引。患者数保持原分层规模，因normal-only患者存在，记录数允许随抽样变化。

每次重新计算两种AUROC及其差值，以2.5和97.5百分位数构成95% CI。若单类抽样使AUROC不可计算，记录为invalid并报告有效replicate数，禁止填入0.5或零差值。当前T2大量完整配对患者使该问题预计很少，仍由代码处理。次要V0−V1采用相同程序。每个研究单独进行患者bootstrap时只使用该研究患者。

置信区间条件于冻结研究、训练流程与现有OOF预测，bootstrap不重新拟合模型，未涵盖重新训练、重复来源抽样或未来研究总体的全部不确定性。三个seed的指标和差值范围单独描述，不将repeat当作独立研究。三外部研究不进行主证据性的paired t-test或强总体meta-analysis推断。

## 指标定义

- 主指标AUROC，tumor为positive class。
- AUPRC固定为average precision，采用sklearn average_precision_score的非插值AP定义。表头注明AUPRC(AP)，不切换为梯形积分结果。
- Log loss为二项交叉熵，概率clip到[1e-15,1−1e-15]，越低越好。
- Brier score为概率与二元标签平方误差的记录均值，越低越好。
- Balanced accuracy为sensitivity与specificity的平均，固定概率阈值0.5。阈值不根据outer test优化。
- 对每个study、pooled OOF和三个study的macro descriptive mean同时报告AUROC和AP，其他指标亦保留。Macro为研究等权的描述性平均，pooled指标会受到大研究权重及跨研究分数尺度影响，两者共同解释。

AUROC和AP的边界差采用V1−V2或V0−V1。Loss与Brier的表格同时给出原始值，若提供改进方向差值则固定为loss_V2−loss_V1或loss_V1−loss_V0，表头写明定义。全部次要指标与敏感性属于描述性/探索性，报告效应与CI，不用显著性排序选择主结论。

## Calibration

只有分析集合至少50名患者、positive和negative各至少20条记录、预测logit有变异且拟合无分离/收敛问题时，才报告calibration intercept与slope。Intercept由标签对固定logit预测的offset logistic拟合，slope由标签对logit预测的自由截距logistic拟合。仅评价冻结OOF概率，不用这些拟合重新校准测试概率。患者bootstrap用于CI。条件未满足记NA并报告原因。GSE37182与GSE50760患者数较小，对其单独slope不作数值解读。

## 队列差异、controls与敏感性

每项结果显示3个研究各自点估计、CI与n_records/n_patients。主差值混合了研究熟悉度、来源组成及训练规模变化。训练量匹配比较给出补充估计，其两arm测试集、配对患者数和类数相同。平台与study完全混杂，任何差值均不能分解为独立平台效应。

T1保留healthy与cancer-patient normal的分层，显示各研究可用性。训练仅用预冻结T1任务，不能根据敏感性显著性更换主任务。Positive control、NC1、NC2按FINAL_ANALYSIS_PROTOCOL执行，真实控制结果尚未观察。信号不足或控制异常时按协议收敛结论或暂停解释，原始预测和失败仍保留。

## 软件验证与实施边界

pipeline_components.py中已实现AUROC及次要指标、paired患者bootstrap、NC1、NC2和train-only estimator组件。synthetic_validation.py使用纯合成数据完成完整主模型嵌套调参、OOF、计分、2000次bootstrap。真实输入仅执行有限性、ID、sample-local和LOSO可用性检查。本轮没有正式疾病性能结果。
