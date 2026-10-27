# MALER reviewed local R2 source

## Current revision status on 30 September 2026

Supplementary Code R2 supplies this source snapshot, the exact retained
Python 3.7.11/Django 2.1.8 environment manifest, final-source product checks,
and the principal-result provenance and selection ledger. Both runtime versions
are unsupported; no migration is part of this revision. Public HTTPS is active
at <https://www.inbirg.com/maler/home>, but public uploads remain disabled (503)
and Aliyun source equivalence, end-to-end behavior, scheduled deletion, backups,
and reconciliation of old notices remain unverified. Use a restricted,
institutionally managed local installation for authorized research data.

All 73 Django tests passed: `conda run -n gene_edit python manage.py test
mlserver.tests mlserver.tests_submission --noinput`. A real local Edge workflow
completed upload through Predict, and all 40 displayed labels matched the signed
pipeline. Actual-product missingness checks audited 168 training-only imputer
fits. These are local checks, not Aliyun end-to-end evidence.

Users fix the scaler/feature/reduction methods and endpoint policy. Inner CV
searches subset/components and estimator parameters for each family. One-click
family selection compares outer means afterward and may be optimistic. Only
the chosen family is retuned/refitted, fully locked and held-out evaluated.
Grouped FSS/BSS and explicit-cutoff binary FSS/BSS are rejected; other supported
grouped paths keep supplied subjects disjoint. Additional ungrouped sequential
selector scores use enclosing-fold preprocessing. The positive class is explicit;
native or prespecified probability-cutoff policy is used consistently throughout.

The local web requests 5×2 outer/3 inner folds, principal scripts 5×10/3 plus
five-fold final tuning, and synthetic/browser checks 3×1/2. Actual budgets and
complete grids are saved. The cap of 24 concerns estimator parameter combinations
before subset/component multiplication, not total pipeline candidates.

The companion `evidence/primary_results` contains the preserved 50-fold results
from source `60351c5` and repeated independent/current-builder checks. This
checkout's `validation/results` retains earlier analysis configurations. The
current principal development metrics are 0.944/0.965 balanced accuracy,
0.902 R² and 0.536 C-index. Historical holdouts have unresolved split provenance;
GEO cohorts are retrospective and expanded-survival family selection is outside
the inner loop. The archive's root README gives current reproduction commands.

Result-token expiry is not physical deletion. Scheduled cleanup and backup
removal need independent operator verification. De-identified molecular data
can remain sensitive. Public uploads default to disabled. Do not reuse synthetic
validation signing keys or expose the retained runtime as an approved public
biomedical-upload service. The existing public tag `v1.0.0-review4` is frozen R1.

## Archived R1 documentation

The text below is preserved historical documentation for the earlier R1 release.
Its HTTP service, test counts and numerical results are not the current R2 status.

# MALER

MALER (Machine Learning Evaluation and Reporting server) is a guided Django
web application for building, evaluating, exporting, and applying machine-
learning models to numerical biological and biomedical data without requiring
users to write code. It supports binary classification, multiclass
classification, regression, and survival analysis.

This repository is the reproducibility source for the peer-review revision. It
contains the web application, leakage-controlled validation code, declared
example train/test matrices, machine-readable result files, external-cohort
provenance records, deployment examples, and figure-generation scripts.

Public mirrors:

- GitHub: <https://github.com/zhaizhaoyu98/MALER_server>
- Gitee: <https://gitee.com/zhaoyuzhai/MLSERVER>

## Public instance

The public HTTP demonstration was updated to the patient-corrected application
source at commit `35aa116` on 24 September 2026. The home, analysis, prediction,
and help pages returned HTTP 200 after the worker reload; both upload pages
displayed the authorization-confirmation field. This was a page-level smoke
test, not an end-to-end upload or a security assessment. The reproducibility
source and results remain tagged `v1.0.0-review4`:

- <http://www.inbirg.com/maler/home>

The public server is a research demonstration and is not a clinical decision
system. Do not upload direct identifiers or data that you are not authorized to
process.

## Reproducible Windows setup

The revision was tested with Python 3.7.11 and Django 2.1.8 to remain compatible
with the existing application and production runtime. Miniforge/Mamba is the
recommended installer.

```powershell
mamba env create -f environment_windows.yml
mamba run -n gene_edit python manage.py check
mamba run -n gene_edit python manage.py test mlserver -v 1
mamba run -n gene_edit python manage.py runserver 127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/maler/home>. Gunicorn and uWSGI are excluded
from the Windows environment because they are Linux deployment components.
The current local release test suite contains 50 Django tests, plus three
patient-partition tests (`python -m unittest validation.test_patient_partition`).

## Validation design

The primary reviewer-validation workflow enforces the repository-declared
development/held-out split. Imputation, scaling, feature selection, and model
tuning are fitted within training folds. Model selection uses repeated 5-fold
outer cross-validation (10 repeats) with 3-fold inner tuning; the held-out
partition is evaluated once after all choices are locked.

The main result and its audit trail can be regenerated with:

```powershell
mamba run -n gene_edit python validation/run_reviewer_validation.py
mamba run -n gene_edit python validation/make_main_figure3.py
```

The source binary-classification matrix contained 312 development and 302
held-out specimens. After an outcome-blind patient-level overlap and aliquot
audit, the corrected analysis has 309 development and 299 internal held-out
patients. The held-out balanced accuracy is 0.940 (95% bootstrap CI
0.911–0.965), ROC-AUC is 0.983 (0.967–0.995), and PR-AUC is 0.987.
The original split-generation seed and upstream filtering protocol cannot be
reconstructed, so the internal held-out estimate is not independent external
validation. These values are read from
`validation/results/binary_classification.json`; the figure script verifies the
dataset and model-bundle hashes and independently recomputes all reported test
metrics before drawing.

Additional scripts cover multiclass classification, regression, survival
analysis, feature-selection baselines, fold-local PCA sensitivity, capacity,
configuration-level security checks, and two real independent lung-cancer
GEO lung-cohort reanalyses. GSE50081 was reported before this correction and
is not a newly untouched final test. See [`validation/README.md`](validation/README.md) for the complete
protocol, accession identifiers, checksums, commands, and limitations.

## Repository map

- `mlserver/`: Django application, validated analysis service, templates, and
  example matrices.
- `validation/`: reproducible analysis, data-quality, benchmark, audit, and
  figure scripts.
- `validation/results/`: inspectable JSON/CSV outputs used in the revision.
- `deployment/`: Linux/Nginx/uWSGI examples and operational checklist.
- `environment_windows.yml`: minimal local Windows environment.
- `environment_server.yml`: legacy-compatible Linux deployment environment.

Large upstream GEO archives, platform annotation databases, temporary model
bundles, local caches, secrets, and partial calculations are intentionally not
versioned. Provenance records and official checksums are versioned so that
source downloads can be independently verified.

## Security and privacy boundary

MALER implements upload limits, strict path/project identifiers, private result
tokens, signed `.maler` model bundles, temporary-cache cleanup, and production
configuration checks. These controls do not constitute a penetration test or
clinical security certification. Configure unique Django and model-signing
secrets outside the repository and follow [`deployment/README.md`](deployment/README.md).

## Version and license

The patient-corrected peer-review source release is tagged `v1.0.0-review4`;
`v1.0.0-review3` is a historical baseline. The current local documentation
edits are not part of that existing tag. Source code is distributed
under the [MIT License](LICENSE). Third-party assets retain their respective
licenses.
