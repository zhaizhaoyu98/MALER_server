# MALER reviewer-validation workflow

This directory contains the reproducible evidence used for the revised
manuscript. It is deliberately separate from historical interactive analysis
views: results from these scripts must not be generalized to a legacy UI branch
until that branch has been migrated and integration-tested.

## Windows environment

The tested environment is `gene_edit` under Miniforge. The minimal Windows
specification is `environment_windows.yml` in the repository root.

Install/update with Mamba. Gunicorn is intentionally excluded on Windows; it is
only relevant to the Linux deployment path.

## Reproduce the checks

From the repository root in PowerShell:

```powershell
mamba run -n gene_edit python manage.py test mlserver -v 1
mamba run -n gene_edit python validation/run_reviewer_validation.py
mamba run -n gene_edit python validation/run_external_gse37745.py
mamba run -n gene_edit python validation/run_external_gse50081.py
mamba run -n gene_edit python validation/audit_external_data_quality.py
mamba run -n gene_edit python validation/audit_survival_data_quality.py
mamba run -n gene_edit python validation/run_survival_expanded_validation.py
mamba run -n gene_edit python validation/run_pca_sensitivity.py
mamba run -n gene_edit python validation/run_equal_budget_baselines.py
mamba run -n gene_edit python validation/benchmark_capacity.py
mamba run -n gene_edit python validation/audit_production_security.py
mamba run -n gene_edit python validation/make_validation_figures.py
mamba run -n gene_edit python validation/make_main_figure3.py
```

Set `MALER_MODEL_SIGNING_KEY` to a strong secret before generating signed model
bundles. Do not commit the key. Completed task JSON files are reused unless
`MALER_VALIDATION_FORCE` is set.

## Evaluation design

- Model development uses the repository-declared training partition only.
- Before fitting, TCGA barcodes are normalized to patient IDs. Every patient
  present in both declared partitions is excluded from both; within either
  partition, one aliquot is selected without consulting outcomes (primary
  tumour type 01, then vial A, then lexical order). The bundled CSVs remain
  unmodified, and each task JSON contains the complete patient-level audit.
  This correction yields binary 309/299, multiclass 482/468, regression
  367/162, and survival 160/133 development/test samples.
- Imputation, scaling, ranking, selection, and tuning are fitted within folds.
- Main estimates use 5-fold × 10 repeated outer validation with a 3-fold inner
  loop; the declared test partition is evaluated once. The public web UI
  defaults to 5-fold × 2 repeats and 3-fold inner selection; 5 × 10 is the
  review-validation script setting, not the web default.
- The no-feature-selection comparison uses the same outer and inner split
  protocol and model hyperparameter grids, but has fewer feature-size
  candidates. It is a matched-resampling comparison, not a compute-matched
  causal ablation of feature selection.
- Declared internal held-out confidence intervals use 1,000 bootstrap
  resamples; independent external-cohort and expanded-survival intervals use
  2,000 bootstrap resamples.
- GSE37745 histology filtering and GPL570 feature availability are determined
  without external outcomes.
- GSE50081 eligibility, model reconstruction, calibration-cohort use, metrics,
  and tie breaking were documented in `external_data/GSE50081_protocol.json`
  before the original downloaded matrix was parsed. Its patient-level
  reanalysis is not newly untouched because prior releases reported its
  outcomes; no GSE50081 outcomes enter the current model or threshold choice.

## Real external cohort

- Accession: GSE37745 / E-GEOD-37745
- Source: NCBI GEO and EMBL-EBI BioStudies
- Platform: Affymetrix HG-U133 Plus 2.0
- Official processed archive SHA-256:
  `3B331E2A531B314B503D0E65D6D2F2259AD32FDE1F40C0F3271998DC38753B27`
- Archive integrity: 196 members, ZIP CRC passed
- Target-matched analysis: 106 LUAD + 66 LUSC; 24 large-cell samples excluded
- Platform mapping after patient-level correction: 42 of 50 candidate genes
  (84%); 20 final features

The primary external transform is a within-sample percentile rank and does not
use the external cohort distribution. Cohort-wise z-scoring is retained only as
a clearly labeled transductive sensitivity analysis.

## Second external cohort and threshold calibration

- Accession: GSE50081 (UHN181), GPL570, submitter-processed log2 RMA
- Official Series Matrix SHA-256:
  `6F987E2F59A4542CBB563FA6CBD289B7F7AE35D034FD64ABB2735C844CEA99A3`
- Matrix: 54,675 unique probes × 181 unique samples; all values finite
- Prespecified target-matched analysis: 127 LUAD + 42 LUSC; 12 other or
  ambiguous histologies excluded without reference to model output
- Recalculated GSE37745-only threshold: 0.932106; GSE50081 outcomes were not
  used for its selection in this rerun.
- GSE50081 retrospective reanalysis: balanced accuracy 0.873, ROC-AUC 0.893,
  specificity 0.890, sensitivity 0.857, and MCC 0.707. Exact bootstrap CIs
  are in `results/external_gse50081.json`.

The uncalibrated 0.5 threshold is also retained (balanced accuracy 0.649,
specificity 0.370). This demonstrates that discrimination transferred but the
decision threshold required calibration; it is not hidden by the stronger
calibrated result.

## Expanded survival evidence

- The repository survival example is a TCGA development / CGGA external
  cross-cohort stress test. Seven TCGA aliquots collapse to 160 unique patients.
- Training-only model-family selection chose fold-local PCA plus regularized
  CoxPH (nested C-index 0.603±0.056). Its locked CGGA C-index was 0.523 (95% CI
  0.470–0.579), so cross-cohort transfer remains weak.
- On the real public GBSG2 positive-control benchmark, training-only selection
  chose gradient boosting (nested C-index 0.690±0.029). The declared holdout
  C-index was 0.700 (0.641–0.756), mean time-dependent AUC 0.768, and integrated
  Brier score 0.166.

## Direct reference implementation parity

`reference_parity_check.py` refits the locked binary Logistic Regression and
regression Ridge models using direct scikit-learn components, without importing
the MALER pipeline or metric helpers. It checks the archived input hash,
patient-level exclusions, selected features, and held-out metrics; for the
binary task it also compares all 299 patient predictions and probabilities.
The run passed at absolute tolerance `1e-10`; details are in
`results/reference_parity_check.json`. This is evidence for two selected
models, not all estimators or an independent algorithm reimplementation.

## Capacity evidence

Fresh Windows subprocesses measured peak working set for every case. The
largest tested training matrix (1,000 samples × 5,000 features) completed in
7.72 seconds with 344.6 MB peak working set. A 10,000-row prediction batch ran
at approximately 839,067 predictions/second for the in-process prediction
call only (not CSV upload, queueing, or end-to-end web throughput). These are
observed single-process measurements on this machine, not universal server
limits.

## Configuration-level security evidence

`audit_production_security.py` activates a simulated production configuration
and checks 10 controls: debug mode, secret strength, allowed hosts, secure
cookies, HTTPS redirect, HSTS, proxy SSL header, the independent model-signing
key, bounded uploads, and strict project-ID/path handling. All 10 checks pass;
`manage.py check --deploy` also reports no issues with high-entropy test
secrets. This does not inspect the live Aliyun host, its certificate chain,
network controls, backups, or penetration resistance.

## Main limitations

- TCGA-to-CGGA survival transfer remains weak despite improved development CV;
  the stronger GBSG2 result is a standard positive-control benchmark, not a
  replacement for disease-specific external validation.
- GSE50081 supports GSE37745-based threshold calibration, but neither cohort is
  prospective and neither establishes clinical utility.
- The automated usability smoke tests and execution-ready human-study protocol
  are not a completed human-participant usability study.
- Full legacy-UI migration, a supported dependency upgrade, a live Aliyun
  security/TLS audit, prospective validation, and independent penetration
  testing remain outside the evidence completed on this computer.
