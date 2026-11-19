"""Reusable frozen operations, exercised exclusively on synthetic data in Phase 0.3."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss, balanced_accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def estimator(C, seed, scaled=False, l1_ratio=0.0):
    steps = [('variance', VarianceThreshold(1e-12))]
    if scaled:
        steps.append(('scale', StandardScaler()))
    # sklearn 1.9 selects the penalty through l1_ratio. Zero specifies L2.
    steps.append(('model', LogisticRegression(C=C, l1_ratio=l1_ratio,
                  solver='liblinear' if l1_ratio == 0 else 'saga',
                  tol=1e-6 if l1_ratio == 0 else 1e-4, max_iter=10000,
                  class_weight=None, random_state=seed, fit_intercept=True)))
    return Pipeline(steps)


def fit_checked(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        return model.fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    return dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), Brier=float(brier_score_loss(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)))


def fit_nested(X, frame, split, grid, scaled=False, l1_ratio=0.0):
    """One outer split. Each variance/scaling estimator is newly fit in its training scope."""
    locations = dict(zip(frame.record, range(len(frame))))
    train_rows = split[split.outer_role == 'train']
    test_rows = split[split.outer_role == 'test']
    seed = int(split.seed.iloc[0])
    scores = []
    for C in grid:
        aucs = []
        for fold in sorted(train_rows.inner_fold.unique()):
            fit = train_rows[train_rows.inner_fold != fold]
            valid = train_rows[train_rows.inner_fold == fold]
            a = np.array([locations[r] for r in fit.record_id])
            b = np.array([locations[r] for r in valid.record_id])
            model = fit_checked(estimator(C, seed, scaled, l1_ratio), X[a], frame.y.to_numpy()[a])
            p = model.predict_proba(X[b])[:, 1]
            for study in sorted(valid.study.unique()):
                mask = valid.study.to_numpy() == study
                aucs.append(roc_auc_score(frame.y.to_numpy()[b][mask], p[mask]))
        scores.append(float(np.mean(aucs)))
    best = next(i for i, score in enumerate(scores) if score >= max(scores) - 1e-12)
    a = np.array([locations[r] for r in train_rows.record_id])
    b = np.array([locations[r] for r in test_rows.record_id])
    model = fit_checked(estimator(grid[best], seed, scaled, l1_ratio), X[a], frame.y.to_numpy()[a])
    return b, model.predict_proba(X[b])[:, 1], grid[best]


def patient_bootstrap_indices(frame, rng):
    """Keep each selected patient's complete record block, with draw multiplicity."""
    blocks = []
    for _, part in frame.groupby('study', sort=True):
        patients = sorted(part.patient.unique())
        drawn = rng.choice(patients, size=len(patients), replace=True)
        blocks.extend(part.index[part.patient == patient].to_numpy() for patient in drawn)
    return np.concatenate(blocks)


def paired_bootstrap(frame, p1, p2, replicates=2000, seed=170329):
    frame = frame.reset_index(drop=True)
    y = frame.y.to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        indices = patient_bootstrap_indices(frame, rng)
        if len(np.unique(y[indices])) == 2:
            samples.append(roc_auc_score(y[indices], p1[indices]) - roc_auc_score(y[indices], p2[indices]))
    if not samples:
        raise ValueError('No bootstrap replicate contains both classes')
    return dict(estimate=float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
                lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
                requested=replicates, valid=len(samples), seed=seed)


def permute_training_labels(frame, y, seed):
    """Pair swaps preserve opposite-label patient profiles and study class totals."""
    frame = frame.reset_index(drop=True)
    result = np.asarray(y).copy()
    rng = np.random.default_rng(seed)
    for _, part in frame.groupby('study', sort=True):
        singles = []
        for _, patient in part.groupby('patient', sort=True):
            indices = patient.index.to_numpy()
            if len(indices) == 2 and len(np.unique(result[indices])) == 2:
                if rng.integers(2):
                    result[indices] = result[indices][::-1]
            elif len(indices) == 1:
                singles.extend(indices)
            else:
                raise ValueError('Unexpected patient profile for frozen tasks')
        result[singles] = rng.permutation(result[singles])
    return result


def destroy_training_genes(frame, X, seed):
    frame = frame.reset_index(drop=True)
    result = np.asarray(X).copy()
    rng = np.random.default_rng(seed)
    for _, part in frame.groupby('study', sort=True):
        indices = part.index.to_numpy()
        for gene in range(X.shape[1]):
            result[indices, gene] = X[rng.permutation(indices), gene]
    return result


def control_seed(control_id, control_replicate, boundary_id, repeat_id, outer_id, inner_id):
    sequence = np.random.SeedSequence([300000, control_id, control_replicate, boundary_id, repeat_id, outer_id, inner_id])
    return int(sequence.generate_state(1)[0])
