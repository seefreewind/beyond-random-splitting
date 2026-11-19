#!/usr/bin/env python3
"""Freeze annotation-defined genes and sample-local inputs. No model fitting."""

import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PREVIOUS = PROJECT / 'phase0_1_rescue'
STUDIES = ['GSE44076', 'GSE37182', 'GSE50760']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_tsv(relative, frame):
    frame.to_csv(ROOT / relative, sep='\t', index=False, lineterminator='\n')


def rank_samples(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite sample-by-gene matrix with at least two genes')
    return (rankdata(values, method='average', axis=1) - 1) / (values.shape[1] - 1)


def gene_reference():
    path = ROOT / 'evidence/Homo_sapiens.gene_info.gz'
    reference = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    reference = reference[reference['#tax_id'] == '9606']
    active = set(reference['GeneID'])
    symbols = dict(zip(reference['GeneID'], reference['Symbol']))
    official, aliases = defaultdict(set), defaultdict(set)
    for row in reference.to_dict('records'):
        official[row['Symbol']].add(row['GeneID'])
        for synonym in row['Synonyms'].split('|'):
            if synonym != '-':
                aliases[synonym].add(row['GeneID'])
    version = 'NCBI_Homo_sapiens.gene_info_SNAPSHOT_SHA256_' + sha(path)
    return active, symbols, official, aliases, version


def platform_table(platform):
    plain = ROOT / f'evidence/{platform}.soft'
    compressed = ROOT / f'evidence/{platform}_family.soft.gz'
    path = plain if plain.exists() else compressed
    opener = gzip.open if path.suffix == '.gz' else open
    table_lines = []
    metadata = []
    in_table = False
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.lower().startswith('!platform_table_begin'):
                in_table = True
            elif line.lower().startswith('!platform_table_end'):
                break
            elif in_table:
                table_lines.append(line)
            else:
                metadata.append(line.strip())
    frame = pd.read_csv(io.StringIO(''.join(table_lines)), sep='\t', dtype=str, keep_default_na=False)
    return frame, path, metadata


def array_matrix(study):
    path = PREVIOUS / f'evidence/benchmark/{study}_series_matrix.txt.gz'
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('!series_matrix_table_begin'):
                break
        lines = []
        for line in handle:
            if line.startswith('!series_matrix_table_end'):
                break
            lines.append(line)
    frame = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', index_col=0)
    frame.index = frame.index.astype(str)
    return frame, path


def rnaseq_matrix(records):
    path = PREVIOUS / 'evidence/benchmark/GSE50760_RAW.tar'
    columns = {}
    with tarfile.open(path) as archive:
        members = {member.name.split('_', 1)[0]: member for member in archive if member.isfile()}
        for record in records:
            member = members[record]
            with gzip.GzipFile(fileobj=archive.extractfile(member)) as handle:
                frame = pd.read_csv(handle, sep='\t', index_col=0)
            columns[record] = frame.iloc[:, 0]
    frame = pd.DataFrame(columns)
    if frame.isna().any().any() or (frame.to_numpy() < 0).any():
        raise ValueError('Missing or negative deposited RNA-seq FPKM')
    return frame, path


def resolve_feature(feature, study, active, official, aliases, feature_annotations):
    if study == 'GSE50760':
        candidates = official.get(feature, set())
        method = 'NCBI_EXACT_OFFICIAL_SYMBOL'
        if not candidates:
            candidates = aliases.get(feature, set())
            method = 'NCBI_UNIQUE_HISTORICAL_SYNONYM'
    else:
        candidates = set(re.findall(r'\d+', feature_annotations.get(feature, '')))
        method = 'GEO_PLATFORM_EXPLICIT_ENTREZ'
    if len(candidates) == 1 and next(iter(candidates)) in active:
        return next(iter(candidates)), method, 'MAPPED_UNIQUE_ACTIVE'
    status = 'EXCLUDED_AMBIGUOUS' if len(candidates) > 1 else 'EXCLUDED_RETIRED_OR_UNMAPPED'
    return '', method, status


def main():
    eligibility = pd.read_csv(PREVIOUS / 'eligibility/formal_sample_eligibility.tsv', sep='\t', dtype=str, keep_default_na=False)
    formal = eligibility[eligibility['eligible'] == 'TRUE'].copy()
    formal = formal.sort_values(['dataset', 'patient_id', 'GSM']).reset_index(drop=True)
    frozen = pd.DataFrame({
        'study': formal['dataset'], 'patient': formal['patient_id'], 'record': formal['GSM'],
        'label': np.where(formal['label'] == 'tumor', 'tumor', 'non_tumor'),
        'tissue': formal['label'], 'normal_subtype': formal['normal_subtype'],
        'platform': formal['platform'],
        'assay': np.where(formal['dataset'] == 'GSE50760', 'RNA-seq', 'microarray'),
        'feature_source': np.where(formal['dataset'] == 'GSE50760', 'GSE50760_RAW.tar/FPKM', 'deposited_GEO_series_matrix'),
        'included': True, 'therapy_status': formal['therapy_status'],
        'patient_evidence': formal['patient_evidence'],
    })
    primary = frozen[frozen['tissue'] != 'healthy normal'].copy()
    primary['label'] = np.where(primary['tissue'] == 'tumor', 'tumor', 'cancer_patient_normal')
    save_tsv('data/FROZEN_PRIMARY_DATASET.tsv', primary)
    save_tsv('data/FROZEN_T1_SENSITIVITY_DATASET.tsv', frozen)
    active, symbols, official, aliases, version = gene_reference()
    inputs, gene_sets, mapping_rows, audits = {}, {}, [], []
    for study in STUDIES:
        selected = frozen[frozen['study'] == study]
        if study == 'GSE50760':
            expression, input_path = rnaseq_matrix(selected['record'].tolist())
            annotation_path = ROOT / 'evidence/Homo_sapiens.gene_info.gz'
            annotation = None
            feature_type = 'deposited_gene_symbols_historical_hg19'
        else:
            expression, input_path = array_matrix(study)
            annotation, annotation_path, metadata = platform_table(selected['platform'].iloc[0])
            column = next(column for column in annotation.columns if re.sub('[^a-z0-9]', '', column.lower()) in ['entrezgeneid', 'entrezgene'])
            feature_annotations = dict(zip(annotation['ID'], annotation[column]))
            feature_type = 'Affymetrix_probe_set' if study == 'GSE44076' else 'Illumina_probe'
        mapping = {}
        for feature in expression.index:
            gene, method, status = resolve_feature(feature, study, active, official, aliases,
                                                   feature_annotations if study != 'GSE50760' else {})
            if gene:
                mapping[feature] = gene
            mapping_rows.append(dict(study=study, original_feature=feature, stable_gene_id=gene,
                                     gene_symbol=symbols.get(gene, ''), mapping_source=str(annotation_path.relative_to(ROOT)),
                                     mapping_version=version, resolution_method=method, status=status))
        # Feature lists and annotations determine the universe. No expression-value filtering.
        gene_sets[study] = set(mapping.values())
        kept = expression.loc[list(mapping), selected['record'].tolist()].copy()
        kept.index = [mapping[feature] for feature in kept.index]
        collapsed = kept.groupby(level=0, sort=False).median()
        if not np.isfinite(collapsed.to_numpy()).all():
            raise ValueError(f'Nonfinite mapped input for {study}')
        if study == 'GSE50760':
            collapsed = np.log2(1 + collapsed)
        inputs[study] = collapsed
        audits.append(dict(study=study, platform=selected['platform'].iloc[0], assay=selected['assay'].iloc[0],
                           original_feature_type=feature_type, original_feature_n=len(expression),
                           normalization='DEPOSITED_RMA' if study == 'GSE44076' else 'DEPOSITED_RSN' if study == 'GSE37182' else 'DEPOSITED_FPKM',
                           log_status='DEPOSITED_LOG2_UNCHANGED' if study != 'GSE50760' else 'SAMPLE_LOCAL_LOG2_1_PLUS_FPKM',
                           annotation_source=str(annotation_path.relative_to(ROOT)), gene_mapping_version=version,
                           annotation_sha256=sha(annotation_path), mapped_gene_n=len(gene_sets[study]),
                           mapped_feature_n=len(mapping), duplicate_gene_policy='WITHIN_SAMPLE_MEDIAN_OF_ALL_UNIQUELY_MAPPED_FEATURES',
                           input_expression_sha256=sha(input_path), records_primary=len(primary[primary['study'] == study]),
                           original_assembly='hg19' if study == 'GSE50760' else 'GEO_PLATFORM_ANNOTATION_SNAPSHOT',
                           original_annotation_release='NOT_REPORTED_FOR_DEPOSITED_SYMBOLS' if study == 'GSE50760' else 'GEO_PLATFORM_SNAPSHOT_SEE_SHA256'))
    genes = sorted(set.intersection(*gene_sets.values()), key=int)
    if len(genes) < 2:
        raise ValueError('Insufficient common stable identifiers')
    universe = pd.DataFrame({'stable_gene_id': genes, 'gene_symbol': [symbols[gene] for gene in genes],
                             'mapping_source': 'NCBI_Gene_info_and_frozen_GEO_platform_annotations', 'mapping_version': version})
    save_tsv('platform/COMMON_GENE_UNIVERSE.tsv', universe)
    save_tsv('platform/FEATURE_MAPPING_AUDIT.tsv', pd.DataFrame(mapping_rows))
    save_tsv('platform/PLATFORM_AUDIT.tsv', pd.DataFrame(audits))
    matrix_by_record = {record: inputs[study].loc[genes, record].to_numpy(dtype=np.float64)
                        for study in STUDIES for record in inputs[study].columns}
    qc = []
    for name, manifest in [('primary', primary), ('T1_sensitivity', frozen)]:
        values = np.vstack([matrix_by_record[record] for record in manifest['record']])
        ranks = rank_samples(values)
        for index, row in manifest.reset_index(drop=True).iterrows():
            independently = rank_samples(values[index:index + 1])
            if not np.array_equal(ranks[index], independently[0]):
                raise AssertionError('Sample-local rank invariance failed')
            qc.append(dict(task=name, record=row['record'], study=row['study'], n_common_genes=len(genes),
                           nonfinite_n=int((~np.isfinite(values[index])).sum()), independent_rank_equals_batch=True))
        np.savez_compressed(ROOT / f'data/{name}_inputs.npz', provided=values, rank=ranks,
                            record_ids=manifest['record'].to_numpy(dtype=str), gene_ids=np.asarray(genes),
                            patient_ids=manifest['patient'].to_numpy(dtype=str), study=manifest['study'].to_numpy(dtype=str),
                            y=(manifest['label'] == 'tumor').to_numpy(dtype=np.int64))
    save_tsv('data/INPUT_AUDIT.tsv', pd.DataFrame(qc))
    template = primary.rename(columns={'record': 'record_id', 'patient': 'patient_id'})[['record_id', 'patient_id', 'study', 'label']].copy()
    for boundary in ['V0', 'V1', 'V2']:
        template[f'prob_{boundary}'] = ''
    save_tsv('protocol/PREDICTION_TABLE_TEMPLATE.tsv', template)
    summary = dict(primary_records=len(primary), primary_patients=primary['patient'].nunique(),
                   primary_labels=primary['label'].value_counts().to_dict(),
                   sensitivity_records=len(frozen), sensitivity_patients=frozen['patient'].nunique(),
                   genes_by_study={study: len(genes_set) for study, genes_set in gene_sets.items()},
                   common_genes=len(genes), gene_universe_rule='ANNOTATION_INTERSECTION_ONLY',
                   rank_rule='COMMON_GENES_FIRST;AVERAGE_TIES;(RANK_MINUS_1)/(P_MINUS_1);SINGLE_SAMPLE',
                   gene_mapping_version=version, formal_real_models_fitted=0, real_disease_AUROC_computed=0)
    (ROOT / 'manifests/input_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    primary_path = ROOT / 'data/FROZEN_PRIMARY_DATASET.tsv'
    (ROOT / 'data/FROZEN_PRIMARY_DATASET.tsv.sha256').write_text(sha(primary_path) + '  FROZEN_PRIMARY_DATASET.tsv\n')
    hashes = [dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for path in sorted((ROOT / 'data').glob('*'))
              if path.is_file() and not path.name.startswith('._') and path.name != 'DATA_SHA256.tsv']
    save_tsv('data/DATA_SHA256.tsv', pd.DataFrame(hashes))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
