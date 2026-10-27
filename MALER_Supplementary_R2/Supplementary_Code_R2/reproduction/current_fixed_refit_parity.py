"""Check the current MALER builders against the preserved principal results.

This is a compatibility check, separate from the helper-independent reference.
It does not repeat model selection or the 50 outer evaluations.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sksurv.tree import SurvivalTree

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MALER_VALIDATION_OUTPUT_DIR', str(ROOT / 'evidence' / 'primary_results'))
os.environ.setdefault('MALER_FIGURE_OUTPUT_DIR', str(ROOT / 'evidence' / 'primary_figures'))
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'source' / 'validation')]
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ML_WebServer.settings')
import django
django.setup()
import reference_parity_check as ref
from mlserver.safe_ml import build_classification_pipeline, build_regression_pipeline, classification_metrics, regression_metrics
from mlserver.safe_survival import build_survival_pipeline, survival_metrics
from mlserver.source_version import source_identity

observations = {}
for task in ('binary_classification', 'multiclass_classification', 'regression', 'survival'):
    result = json.loads((ref.RESULTS / (task + '.json')).read_text())
    if task == 'survival':
        train, y, test, truth = ref._survival_input(result['dataset'])
        model = build_survival_pipeline(SurvivalTree(random_state=10, min_samples_leaf=10), k=50)
    else:
        filename = task + '_example.csv'
        train, y, test, truth = ref._matrix(filename, result['dataset'], 'PURITY' if task == 'regression' else 'label')
        if task == 'regression':
            y, truth = y.astype(float), truth.astype(float)
            model = build_regression_pipeline(Ridge(), k=50)
        else:
            model = build_classification_pipeline(LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=500, random_state=10), k=50)
    model.set_params(**result['final_best_params']).fit(train, y)
    selected = train.columns[model.named_steps['selector'].get_support()].tolist()
    assert selected == result['final_selected_features']
    if task == 'survival':
        metrics = survival_metrics(model, train, y, test, truth)
        names = ['c_index']
    elif task == 'regression':
        metrics = regression_metrics(truth, model.predict(test))
        names = ['r2', 'mae', 'mse', 'rmse']
    else:
        metrics = classification_metrics(truth, model.predict(test), scores=model.predict_proba(test), classes=model.classes_, positive_class='LUSC' if task == 'binary_classification' else None)
        names = ['balanced_accuracy', 'roc_auc', 'pr_auc'] if task == 'binary_classification' else ['balanced_accuracy', 'roc_auc_ovr_macro', 'pr_auc_macro']
    for name in names:
        np.testing.assert_allclose(metrics[name], result['test_metrics'][name], atol=1e-10, rtol=0)
    observations[task] = {'metrics': {n: metrics[n] for n in names}, 'selected_features_match': True}
report = {'source': source_identity(), 'tolerance': 1e-10, 'scope': __doc__, 'tasks': observations}
(ref.RESULTS / 'current_fixed_refit_parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'source': report['source'], 'all_four_tasks_passed': True}, indent=2))
