# Comprehensive audit of the R2 submission

The audit repairs signed-Predict endpoint semantics (zero cutoff), numeric or
missing-first outcome handling, delimiter/identifier parsing, invalid numeric
cells, and survival-pipeline reuse. All four signed task paths reuse fitted
preprocessing without fitting. Survival outputs risk scores and available
curves, with C-index for labeled cohorts. IPCW AUC/IBS are not available in
Predict because exports omit the development censoring reference; no substitute
is learned from the prediction cohort. Analysis metrics are unchanged.

Nine additional actual-route regression checks augment the earlier 73 tests.
Historical cloud/browser/scientific evidence retains its original version;
the source difference from 1b3c075 is no longer presentation only. Public
uploads stay disabled; Python 3.7.11/Django 2.1.8 and NAS governance are deferred.

Average precision is the noninterpolated sklearn statistic. Historical JSON
keys pr_auc and pr_auc_macro are retained for compatibility and denote AP.
The manuscript clarifies this definition and corrects the tie-order and
no-reduction figure cross-reference. The original submission package is kept.
