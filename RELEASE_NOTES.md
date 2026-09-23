# MALER v1.0.0-review2

This release freezes the merged code, reproducibility evidence, and publication
statements prepared for the current peer-review revision.

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
- Added result-page notices for reordered prediction features and ignored
  unexpected features without changing prediction values.
- Restored ANOVA/MRMR controls when switching from survival back to another
  task and removed horizontal overflow from the analysis form.
- Made preview charts responsive, added a site icon, and removed unused
  third-party scripts that generated browser errors.
- Verified the local signed-model round trip from validated analysis export to
  external prediction upload using browser-driven controls.
- Removed the final routed legacy calculation helper; all public calculation
  routes now use the validated engine.
- Expanded the Django suite from 36 to 46 passing tests and reconciled the
  manuscript, response letter, tables, supplement, and repository documentation.

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
