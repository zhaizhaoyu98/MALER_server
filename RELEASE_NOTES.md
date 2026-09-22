# MALER v1.0.0-review1

This release freezes the code and evidence prepared for the first peer-review
revision.

## Main changes

- Added a unified leakage-controlled analysis service for classification,
  regression, and survival workflows.
- Confined imputation, scaling, feature reduction, and hyperparameter tuning to
  training folds and reserved the declared test partition for one final audit.
- Added signed `.maler` model bundles and identifier-aligned external
  prediction.
- Added private result access tokens, upload limits, path validation, cache
  retention tooling, and deployment checks.
- Added explicit UI guidance for scaling, feature ranking, subset search,
  validation, privacy, and model export.
- Added reproducible internal, external-cohort, survival, PCA, equal-budget,
  capacity, and configuration-security evidence.
- Added publication-quality figure generation with hash and metric checks.

## Compatibility

The application remains on Python 3.7.11 and Django 2.1.8 for compatibility
with the existing Aliyun deployment. The Windows environment intentionally
omits Gunicorn/uWSGI. The Linux environment retains them for the legacy service
layout.

## Known limitations

- The software is a research tool and has not been prospectively validated for
  clinical decision-making.
- The completed usability evidence is automated; the supplied human-study
  protocol has not yet been executed with recruited participants.
- TCGA-to-CGGA survival transport remains weak and is reported as such.
- TLS/HTTPS configuration is not part of this release audit.
