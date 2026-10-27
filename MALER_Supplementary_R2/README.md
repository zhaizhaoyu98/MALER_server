# MALER R2 supplementary code and verification evidence

This single attachment contains the **Supplementary Code R2** and **Cloud Evidence R2** sections referenced in the manuscript and response letter. Upload this ZIP instead of the two earlier ZIPs. Submit the manuscript, Supplementary Information and publication figures separately.

## Contents

| Location | Contents |
| --- | --- |
| `Supplementary_Code_R2/source/` and `source_assets/` | Reviewed source and losslessly compressed large assets; all 389 original file contents are covered by the original SHA-256 manifest. |
| `Supplementary_Code_R2/evidence/primary_results/` | Principal results, fitted models, independent reference checks and scientific provenance. |
| `Supplementary_Code_R2/evidence/product_missingness/` | Controlled missingness inputs and results. |
| `Supplementary_Code_R2/figure_sources/` | Final figure-generation helpers and editable workflow source. |
| `Cloud_Evidence_R2/production_validation/` | Eight complete public HTTPS workflows, source/process identity, original captures, signed export/lock checks, 95-test records and protected deletion. |
| `Cloud_Evidence_R2/supporting_checks/` and `operations_checks/` | Selected XGBoost, HSTS, certificate-renewal, recovery and cleanup evidence. |
| `reviewer_evidence_index.csv` | All 77 evidence locations across 31 reviewer subitems. Document-only items are submitted separately. |
| `PROVENANCE.md` | Source identities, evidence scope and retained limitations. |

## Reproduction

Extract the entire ZIP and keep both section directories together. From `Supplementary_Code_R2/`, run:

```
python bootstrap_source.py
python reproduce.py verify
```

These commands use only the Python standard library. The bootstrap restores 15 required large files without Git, internet access or credentials, then verifies all 389 original source contents. Existing edited files are rejected by checksum and never overwritten. Five historical documentation/static-asset filenames have been normalized; the path map is in `source_assets.json`, and the original source manifest is unchanged. References inside preserved source notes may use original names; use the path map to locate them.

With the retained Python 3.7.11 environment and pinned dependencies:

```
python reproduce.py tests
python reproduce.py parity
python reproduce.py missingness
python reproduce.py figures
python reproduce.py browser-evidence
```

Environment specifications are in `source/environment_windows.yml` and `source/environment_server.yml`; recorded versions are in `evidence/environment_manifest_py37.json`. No environment upgrade is performed. New outputs go to `reproduction_runs/`, preserving all submitted evidence.

The checks cover the 95-test application suite, independent four-task refits, controlled missingness, nine figure outputs and retained browser evidence. The original missingness fixture uses manifest verification in this archive and records that its local Git-release requirement is disabled. Its model and imputation checks are unchanged. The application continues to report archive Git identity as unverified.

Production signatures were checked on the server. The private signing key is excluded, so offline archive checks cannot independently authenticate those signatures. The earlier local-browser fixture uses its public synthetic test key. Offline checks do not perform a new public-server test or rerun the historical 50-fold study.

## Presentation and provenance

Directory names, filenames and these submission guides use functional names without calendar dates. Original JSON, logs, screenshots and source notes retain their factual timestamps and source references. The reviewed source file contents are unchanged. Git history and duplicate historical evidence trees are excluded.

Source identity in this archive is established by the original 389-file SHA-256 manifest. Production deployment requires a proper versioned checkout and operator configuration. See `PROVENANCE.md` for the scientific and operational limits of these records.
