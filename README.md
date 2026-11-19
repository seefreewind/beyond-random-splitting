# Beyond Random Splitting

Validation Boundaries and Score Comparability in Cross-Study Transcriptomic Machine Learning

This package provides the computational materials for a three-study colorectal transcriptomic benchmark. The primary analysis evaluates 261 tissue records from 131 patients using 17,819 genes under record-level, patient-grouped and study-held-out validation. The main estimand is the pooled patient-grouped minus study-held-out AUROC difference. Study-specific discrimination has heterogeneous directions. The post-primary additive decomposition identifies within-study and between-study pair contributions.

## Package contents

- `phase0_3/data/` contains frozen processed expression arrays and sample manifests for T2 and T1.
- `phase0_3/platform/` contains the common gene universe, platform and feature mapping audit.
- `phase0_3/splits/` and `phase0_3/protocol/` contain nested assignments and settings.
- `phase1/scripts/` contains fitting, statistical evaluation and audit code.
- `phase1/predictions/` contains saved out-of-fold predictions, including negative-control replicates.
- `phase1/results/`, `phase1/sensitivity/` and `phase1/controls/` contain frozen full-precision estimates.
- `phase1_1/` contains explanatory decomposition and score-comparability analyses.
- `source_data/`, `scripts/make_figures.py` and `figures/` contain frozen main figure inputs, plotting code and exported figures.

## Verify and inspect

Use Python 3.12 and create a virtual environment. Exact historical numerical versions appear in `phase1/protocol/EXECUTION_RUNTIME.json`. The plotting dependency is pinned separately in `requirements.txt`.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_package.py
python phase0_3/scripts/verify_freeze.py
python scripts/make_figures.py
```

All reported results and confidence intervals are supplied at full precision. These commands validate the release and render figures from the supplied summaries.

## Refit the frozen analyses

Run in a separate working copy to retain the distributed results. Each fitting stage can be expensive. These commands invoke the original frozen analysis algorithms.

```sh
python scripts/prepare_run.py
python phase1/scripts/run_stage.py primary --workers 4
python phase1/scripts/summarize.py primary
python phase1/scripts/run_stage.py controls --workers 4
python phase1/scripts/run_stage.py robustness --workers 4
python phase1/scripts/run_stage.py T1 --workers 4
python phase1/scripts/summarize.py all
python phase1_1/scripts/run_audit.py
```

The original historical freeze verifier referenced the full development archive. The release supplies a scoped verifier for the frozen inputs distributed in this repository. Its hashes are in `FROZEN_INPUTS_SHA256.tsv`. `SOURCE_PROVENANCE.json` identifies copied source files and their original checksums. `phase0_3/manifests/asset_hashes.sha256` retains the historical freeze identity used by case recipes. Original development administration scripts are included for provenance. The entry points above initialize the public package directly.

## Interpretation

Study and platform are fully confounded. The decomposition provides signed additive statistical attribution. Within-study positive-negative record comparisons can involve different patients. Patient-bootstrap intervals condition on the observed studies and saved predictions. Study-score percentile transformation is a transductive descriptive diagnostic. Sensitivity results change model, representation, training composition or case mix as documented in the frozen tables.

## Data, licensing and citation

See `DATA_SOURCES.md` for GEO accessions, input representations and source rights. The MIT license covers original project code. Source data retain their original rights. Zenodo creator metadata uses the collective project name **Beyond Random Splitting**. Version 1.0.0 is the initial public reproducibility release.
