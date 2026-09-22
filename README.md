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

## Public instance

The maintained HTTP endpoint is:

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

The locked binary-classification example contains 312 development samples and
302 held-out samples. The held-out balanced accuracy is 0.954 (95% bootstrap CI
0.931–0.977), ROC-AUC is 0.987 (0.973–0.997), and PR-AUC is 0.989
(0.980–0.997). These values are read from
`validation/results/binary_classification.json`; the figure script verifies the
dataset and model-bundle hashes and independently recomputes all reported test
metrics before drawing.

Additional scripts cover multiclass classification, regression, survival
analysis, feature-selection baselines, fold-local PCA sensitivity, capacity,
configuration-level security checks, and two real independent lung-cancer
cohorts. See [`validation/README.md`](validation/README.md) for the complete
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

The peer-review release is tagged `v1.0.0-review1`. Source code is distributed
under the [MIT License](LICENSE). Third-party assets retain their respective
licenses.
