#!/usr/bin/env python3
"""Deterministic record/patient/study boundaries. Expression is never read."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 29, 43]


def assignments(frame, folds, seed, grouped):
    rng = np.random.default_rng(seed)
    result = pd.Series(-1, index=frame.index, dtype=int)
    for study, subset in frame.groupby('study', sort=True):
        if grouped:
            profiles = subset.groupby('patient')['label'].agg(lambda x: '|'.join(sorted(x)))
            # Keep mixed-label patients together and distribute rare profiles across folds.
            offset = 0
            for profile in sorted(profiles.unique()):
                patients = rng.permutation(sorted(profiles[profiles == profile].index))
                for index, patient in enumerate(patients):
                    result.loc[subset[subset.patient == patient].index] = (index + offset) % folds
                offset = (offset + len(patients)) % folds
        else:
            for label, part in subset.groupby('label', sort=True):
                indices = rng.permutation(part.index.to_numpy())
                result.loc[indices] = np.arange(len(indices)) % folds
    assert (result >= 0).all()
    return result


def construct(frame, task):
    rows, audits = [], []
    for repeat, seed in enumerate(SEEDS):
        for boundary in ['V0', 'V1', 'V2']:
            if boundary == 'V2':
                outer_ids = frame.study.map({s: i for i, s in enumerate(sorted(frame.study.unique()))})
            else:
                outer_ids = assignments(frame, 5, seed, boundary == 'V1')
            for outer_fold in sorted(outer_ids.unique()):
                test = frame.index[outer_ids == outer_fold]
                train = frame.index[outer_ids != outer_fold]
                inner = assignments(frame.loc[train], 3, seed + 1000 + int(outer_fold), boundary != 'V0')
                for index in frame.index:
                    rows.append(dict(task=task, boundary=boundary, repeat=repeat, seed=seed,
                                     outer_fold=outer_fold, record_id=frame.loc[index, 'record'],
                                     patient_id=frame.loc[index, 'patient'], study=frame.loc[index, 'study'],
                                     outer_role='test' if index in test else 'train',
                                     inner_fold=-1 if index in test else int(inner.loc[index])))
                scopes = [('outer', train, test)]
                scopes += [(f'inner_{k}', inner.index[inner != k], inner.index[inner == k]) for k in range(3)]
                for scope, fit, evaluate in scopes:
                    a, b = frame.loc[fit], frame.loc[evaluate]
                    patient_overlap = len(set(a.patient) & set(b.patient))
                    study_overlap = len(set(a.study) & set(b.study))
                    assert not (set(a.record) & set(b.record))
                    assert a.label.nunique() == b.label.nunique() == 2
                    if boundary != 'V0':
                        assert patient_overlap == 0
                    if boundary == 'V2' and scope == 'outer':
                        assert study_overlap == 0
                    else:
                        assert set(a.study) == set(b.study)
                    audits.append(dict(task=task, boundary=boundary, repeat=repeat, outer_fold=outer_fold,
                                       scope=scope, train_records=len(a), test_records=len(b),
                                       train_patients=a.patient.nunique(), test_patients=b.patient.nunique(),
                                       patient_overlap=patient_overlap, study_overlap=study_overlap,
                                       both_classes=True, status='PASS'))
    result = pd.DataFrame(rows)
    for (_, _, _), part in result.groupby(['boundary', 'repeat', 'task']):
        counts = part[part.outer_role == 'test'].record_id.value_counts()
        assert len(counts) == len(frame) and (counts == 1).all()
    return result, pd.DataFrame(audits)


def matched_size(frame, splits):
    rows = []
    profiles = frame.groupby('patient').label.agg(list)
    paired = set(profiles[profiles.map(lambda x: len(x) == 2 and len(set(x)) == 2)].index)
    v1 = splits[splits.boundary == 'V1']
    for (repeat, fold), part in v1.groupby(['repeat', 'outer_fold']):
        for target in sorted(frame.study.unique()):
            rng = np.random.default_rng(SEEDS[repeat] + 10000 + int(fold) * 100 + sorted(frame.study.unique()).index(target))
            known = frame[frame.record.isin(part[part.outer_role == 'train'].record_id) & frame.patient.isin(paired)]
            unseen = frame[(frame.study != target) & frame.patient.isin(paired)]
            count = min(known.patient.nunique(), unseen.patient.nunique())
            test = frame[(frame.study == target) & frame.record.isin(part[part.outer_role == 'test'].record_id)]
            for arm, pool in [('known_studies', known), ('unseen_study', unseen)]:
                # First draw one patient per represented study, then fill without replacement.
                required = [rng.choice(sorted(p.patient.unique())) for _, p in pool.groupby('study', sort=True)]
                others = sorted(set(pool.patient) - set(required))
                selected = required + list(rng.choice(others, count - len(required), replace=False))
                train = pool[pool.patient.isin(selected)]
                assert len(train) == 2 * count
                assert not (set(train.patient) & set(test.patient))
                inner = assignments(train, 3, SEEDS[repeat] + 20000 + int(fold), True)
                for role, part_frame in [('train', train), ('test', test)]:
                    for index, record in part_frame.iterrows():
                        rows.append(dict(repeat=repeat, outer_fold=fold, target_study=target, arm=arm,
                                         role=role, record_id=record.record, patient_id=record.patient,
                                         study=record.study, matched_train_patients=count,
                                         inner_fold=int(inner.loc[index]) if role == 'train' else -1))
    return pd.DataFrame(rows)


def main():
    all_audits = []
    for task, filename in [('T2', 'FROZEN_PRIMARY_DATASET.tsv'), ('T1', 'FROZEN_T1_SENSITIVITY_DATASET.tsv')]:
        frame = pd.read_csv(ROOT / 'data' / filename, sep='\t')
        splits, audits = construct(frame, task)
        splits.to_csv(ROOT / f'splits/{task}_NESTED_ASSIGNMENTS.tsv', sep='\t', index=False)
        all_audits.append(audits)
        if task == 'T2':
            matched_size(frame, splits).to_csv(ROOT / 'splits/T2_MATCHED_TRAIN_SIZE_ASSIGNMENTS.tsv', sep='\t', index=False)
    audit = pd.concat(all_audits, ignore_index=True)
    audit.to_csv(ROOT / 'splits/OVERLAP_AUDIT.tsv', sep='\t', index=False)
    summary = dict(audited_partitions=len(audit), failures=int((audit.status != 'PASS').sum()),
                   V0_patient_overlap_allowed=True, V1_patient_overlap=0, V2_outer_study_overlap=0,
                   repeats=len(SEEDS), seeds=SEEDS, inner_folds=3, V0_V1_outer_folds=5,
                   each_record_tested_once_per_boundary_per_repeat=True)
    (ROOT / 'manifests/split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
