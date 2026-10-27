# MALER public research web application - R2

MALER builds, evaluates, exports and reuses machine-learning pipelines for
numerical biomedical research data. Public service: https://www.inbirg.com/maler/home.
Use authorized, non-sensitive research data; do not submit direct identifiers,
confidential records or sensitive patient data. De-identified molecular data
may still be sensitive. This is an exploratory research service.

## Current release and deployment

This release restores public Analysis and Predict controls and provides a
linked data-handling policy. Python 3.7.11/Django 2.1.8 are retained and
unsupported. HTTPS and application controls do not resolve runtime support.
The deployment record and actual production checks are in Cloud Evidence R2;
source identity is recorded in results and signed models.

Analysis and Predict file-size guidance reads the configured data-upload limit,
matching the policy notices and server checks. The old fixed 200-MB wording
was removed; the production data limit remains 50 MiB.

New public task files and upload buffers use capacity-limited volatile storage.
The deployed service uses a private 1.5-GiB tmpfs with no configured swap;
these new task files are not written into the old NAS task directories.
Host reboot or operational failure can remove them before the default 24-hour
access expiry. A five-minute cleanup removes expired active tasks. Provider,
legacy and other copies remain outside an all-copy deletion guarantee.
Download required results promptly. Historical NAS records remain historical.

## Run and test

Use the supplied exact environment manifest with the retained gene_edit runtime.
The release was tested in that environment; a fresh modern environment is not
claimed. For a controlled local installation, configure unique secrets, a
private task directory and appropriate access restrictions before accepting data.
Public uploads are disabled by default in settings; the operator enables the
flag only in the configured service environment.

```text
python manage.py test mlserver.tests mlserver.tests_submission mlserver.tests_audit mlserver.tests_public_online mlserver.tests_xgboost --noinput
python manage.py audit_security_configuration --strict
python manage.py check --deploy
```

The suite contains 95 tests. SVC probability calibration is seeded at 10;
its paired-fit regression check uses unrelated global RNG states. Registry
version 3 records this change. Registry version 4 adds a fold-local label adapter
for XGBoost 1.6.2 classification, including custom and one-click pipelines.
Original labels and probability-column order are preserved in the signed model;
Predict uses the saved mapping without refitting. Seven regression checks cover
text and nonconsecutive numeric labels, positive-class probabilities, nested CV,
feature selection, model serialization, evaluation-label rejection and margins.
XGBoost now participates on compatible text-labelled data, so one-click family
selection can differ from releases that excluded the failed family.
Unused Axios CDN tags were removed from Analysis/Predict after an actual
production browser run reported that external request blocked by ORB.
Earlier unseeded one-click traces retain their
original commit and can select different families. The four principal fixed
logistic/Ridge/SurvivalTree configurations are unaffected.
The new upload-admission middleware rejects
insufficient/unavailable task storage before multipart parsing. Public data and
model limits are 50/100 MiB; Nginx permits a 160-MiB multipart request, rate-limits
uploads and limits active computations. See deployment/README.md.

## Statistical scope and reproducibility

Users fix scaler/feature/reduction methods and binary endpoint policy. Inner CV
searches subset/components and estimator parameters within each family.
One-click compares per-family outer development means afterward; the winning
mean can be optimistic for that selection procedure. The selected fitted
configuration is locked before within-task held-out evaluation. Grouped and
explicit-cutoff binary FSS/BSS are unavailable; additional sequential-selector
scores reuse enclosing-fold preprocessing. Grouping requires meaningful supplied
subject identifiers. Predict reuses the fitted pipeline without refitting.

The web budget is 5 outer folds x 2 repeats / 3 inner folds. Principal four-task
scripts use 5 x 10 / 3, with five-fold final development tuning. The preserved
50-fold scientific outputs retain source 60351c5. Representative four-task
refits, missingness and earlier browser records retain their actual versions;
they are not relabelled as full reruns of this release. The supplied input
matrices have historical upstream-provenance limitations. GEO analyses are
retrospective, and the mapped gene universe has prior supervised selection.

The companion Supplementary Code R2 root README is the entry point for current
archive-reproduction commands and figure sources. validation/results contains
historical configurations. docs/history explicitly preserves earlier service
states and numbers; those records are not current deployment instructions.

MIT License; third-party assets retain their licenses.
