"""Paired study-stratified patient bootstrap with exact weighted rank metrics."""
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap_weights(frame, B=2000, seed=170329):
    rng=np.random.default_rng(seed)
    weights=np.zeros((B,len(frame)),dtype=np.float64)
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'phase0_3/scripts'))
    from pipeline_components import patient_bootstrap_indices
    for b in range(B):
        indices=patient_bootstrap_indices(frame,rng)
        weights[b]=np.bincount(indices,minlength=len(frame))
    return weights


def weighted_metrics(y,p,weights):
    y=np.asarray(y,dtype=float)
    p=np.asarray(p,dtype=float)
    weights=np.atleast_2d(weights)
    positive=weights@y
    negative=weights@(1-y)
    valid=(positive>0)&(negative>0)
    order=np.argsort(p,kind='stable')
    boundaries=np.flatnonzero(np.r_[p[order][1:]!=p[order][:-1],True])
    cp=np.cumsum(weights[:,order]*y[order],axis=1)[:,boundaries]
    cn=np.cumsum(weights[:,order]*(1-y[order]),axis=1)[:,boundaries]
    gp=np.diff(np.c_[np.zeros(len(weights)),cp],axis=1)
    gn=np.diff(np.c_[np.zeros(len(weights)),cn],axis=1)
    auc=np.sum(gp*(cn-.5*gn),axis=1)/(positive*negative)
    order=np.argsort(-p,kind='stable')
    boundaries=np.flatnonzero(np.r_[p[order][1:]!=p[order][:-1],True])
    tp=np.cumsum(weights[:,order]*y[order],axis=1)[:,boundaries]
    total=np.cumsum(weights[:,order],axis=1)[:,boundaries]
    delta_tp=np.diff(np.c_[np.zeros(len(weights)),tp],axis=1)
    precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
    ap=np.sum(delta_tp*precision,axis=1)/positive
    brier=weights@((p-y)**2)/weights.sum(axis=1)
    clipped_probability=np.clip(p,1e-15,1-1e-15)
    loss=weights@(-y*np.log(clipped_probability)-(1-y)*np.log(1-clipped_probability))/weights.sum(axis=1)
    balanced=.5*((weights@(y*(p>=.5)))/positive+(weights@((1-y)*(p<.5)))/negative)
    result={'AUROC':auc,'AUPRC':ap,'Brier':brier,'log_loss':loss,'balanced_accuracy':balanced}
    for values in result.values(): values[~valid]=np.nan
    return result


def interval(values):
    valid=np.asarray(values)[np.isfinite(values)]
    return (float(np.quantile(valid,.025)),float(np.quantile(valid,.975)),len(valid)) if len(valid) else (np.nan,np.nan,0)


def validate_bootstrap():
    rng=np.random.default_rng(170329)
    y=np.r_[np.zeros(10),np.ones(11)]
    p=np.round(rng.random(len(y)),1)
    weights=rng.integers(0,4,(10,len(y))).astype(float)
    result=weighted_metrics(y,p,weights)
    for i,w in enumerate(weights):
        assert np.isclose(result['AUROC'][i],roc_auc_score(y,p,sample_weight=w),atol=1e-14)
        assert np.isclose(result['AUPRC'][i],average_precision_score(y,p,sample_weight=w),atol=1e-14)


    extreme_labels=np.array([1,0,1,0])
    extreme_probabilities=np.array([1e-17,2e-17,.5,1.0])
    extreme_result=weighted_metrics(extreme_labels,extreme_probabilities,np.ones(4))
    assert np.isclose(extreme_result['AUROC'][0],roc_auc_score(extreme_labels,extreme_probabilities),atol=1e-14)
    assert np.isclose(extreme_result['AUPRC'][0],average_precision_score(extreme_labels,extreme_probabilities),atol=1e-14)


def calibration(frame,p,weights=None):
    if frame.patient.nunique()<50 or min(np.bincount(frame.y.to_numpy(),minlength=2))<20:
        return {'status':'NA_SAMPLE_SIZE','intercept':np.nan,'slope':np.nan}
    z=logit(np.clip(p,1e-15,1-1e-15))
    y=frame.y.to_numpy()
    w=np.ones(len(y)) if weights is None else weights
    if np.ptp(z)<1e-12: return {'status':'NA_CONSTANT_LOGIT','intercept':np.nan,'slope':np.nan}
    def solve(design,offset):
        beta=np.zeros(design.shape[1])
        for _ in range(100):
            eta=design@beta+offset
            fitted=expit(eta)
            curvature=w*fitted*(1-fitted)
            hessian=design.T@(curvature[:,None]*design)
            if np.linalg.cond(hessian)>1e12: return None
            step=np.linalg.solve(hessian,design.T@(w*(y-fitted)))
            beta+=step
            if not np.isfinite(beta).all() or np.max(np.abs(beta))>1e6: return None
            if np.max(np.abs(step))<1e-8: return beta
        return None
    intercept=solve(np.ones((len(y),1)),z)
    slope=solve(np.c_[np.ones(len(y)),z],np.zeros(len(y)))
    if intercept is None or slope is None: return {'status':'NA_SEPARATION_OR_CONVERGENCE','intercept':np.nan,'slope':np.nan}
    return {'status':'ESTIMATED','intercept':float(intercept[0]),'slope':float(slope[1])}
