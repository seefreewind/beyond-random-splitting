# Data provenance

Processed expression matrices derive from public GEO studies GSE44076, GSE37182 and GSE50760. The frozen tissue and patient identifiers link records within these archived sources. Study/platform confounding and inherited archive preprocessing apply.

- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44076
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37182
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50760

The NPZ files retain provided expression, within-sample rank representation, record IDs and gene IDs. Manifests, annotation mapping audits and split assignments identify the processed inputs. The T2 task has 261 tissue records, 131 patients and 17,819 genes. T1 adds 50 healthy normal records from GSE44076, for 311 records and 181 patients.

SOURCE_PROVENANCE.json records SHA-256 values of included original project files. The public verify_freeze.py checks the distributed frozen inputs using FROZEN_INPUTS_SHA256.tsv. The original historical freeze-root digest is retained as analysis provenance. The public verification scope is the distributed package.

Original GEO records and third-party source data remain subject to their original rights and terms. The software license applies to original project code. Reuse of source data should acknowledge its original creators and accessions.
