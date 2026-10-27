"""Inference-only rendering of current signed pipelines for all four tasks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sksurv.metrics import concordance_index_censored

from .safe_ml import (DataValidationError, align_prediction_frame, alignment_report,
                      classification_metrics, classification_prediction, regression_metrics)
from .safe_survival import _transform_for_survival_model, validate_survival_target
from .validated_analysis import read_uploaded_matrix


def _records(index, predicted, observed=None):
    rows = []
    for n, sample in enumerate(index):
        value = predicted[n]
        row = {'Name': str(sample), 'Label': value.item() if hasattr(value, 'item') else value}
        if observed is not None:
            value = observed[n]
            row['Label2'] = value.item() if hasattr(value, 'item') else value
        rows.append(row)
    return rows


def _class_labels(values, classes):
    resolved = []
    for value in values:
        matches = [c for c in classes if str(c) == str(value)]
        if not matches:
            try:
                matches = [c for c in classes if float(c) == float(value)]
            except (ValueError, TypeError):
                pass
        if len(matches) != 1:
            raise DataValidationError('Validation outcomes must match the signed model class labels.')
        resolved.append(matches[0])
    return np.asarray(resolved, dtype=np.asarray(classes).dtype)


def signed_prediction_result(projectid, bundle, input_path):
    from .views.predict_views import pred_val_split
    from .views.classification_cp_result_view_websocket import mkbar, mkheatmap
    from .views.regression_cp_result_view_websocket import mkvreportbarplot, mkvregpredplot
    model, features, method = bundle['model'], bundle['feature_names'], bundle['method']
    task = 'survival' if method == 'model_sur' else method
    data = read_uploaded_matrix(input_path)
    blind, validation = pred_val_split(data, datatype=task)
    positive = None
    if method == 'model_bclass':
        positive = next(c for c in model.classes_ if str(c) == str(bundle['positive_class']))
    policy, cutoff = bundle['threshold_policy'], bundle['threshold_value']
    result = {'predict_schema_version': 1, 'projectid': projectid,
              'method': {'model_bclass': 'Classification', 'model_mclass': 'Classification',
                         'model_reg': 'Regression', 'model_sur': 'Survival'}[method],
              'showtable': False, 'predict_reports_dict': None, 'surv_plot': None,
              'validation_reports_dict': None, 'val_describe_roc': None,
              'alignment_notices': [], 'signed_binary_metrics': None,
              'signed_validation_metrics': None, 'metric_rows': [], 'prediction_warnings': [],
              'positive_class': None if positive is None else str(positive),
              'threshold_policy': policy if positive is not None else None,
              'threshold_value': cutoff if policy == 'explicit_probability_threshold' else None}

    def align(frame, cohort):
        notice = alignment_report(frame, features, cohort=cohort)
        if notice:
            result['alignment_notices'].append(notice)
        return align_prediction_frame(frame, features)[0]

    if len(blind):
        aligned = align(blind, 'blind')
        if method in ('model_bclass', 'model_mclass'):
            prediction, _, _ = classification_prediction(model, aligned, positive, policy, cutoff)
        else:
            prediction = model.predict(aligned)
        result['predict_reports_dict'] = _records(aligned.index, prediction)
        result['showtable'] = True
        if task == 'survival':
            estimator = model.named_steps['model']
            if hasattr(estimator, 'predict_survival_function'):
                transformed = _transform_for_survival_model(model, aligned)
                survival = estimator.predict_survival_function(transformed)
                hazard = estimator.predict_cumulative_hazard_function(transformed)
                result['surv_plot'] = [
                    {'x': f.x.tolist(), 'y': f.y.tolist(), 'name': str(sample),
                     'mode': 'lines', 'line': {'shape': 'hv'},
                     'xaxis': 'x%d' % axis, 'yaxis': 'y%d' % axis}
                    for sample, sf, hf in zip(aligned.index, survival, hazard)
                    for axis, f in [(1, sf), (2, hf)]]
            else:
                result['prediction_warnings'].append(
                    'This estimator supplies risk scores but does not support survival curves.')
    if len(validation):
        outcome_count = 2 if task == 'survival' else 1
        aligned = align(validation.iloc[:, outcome_count:], 'validation')
        if method in ('model_bclass', 'model_mclass'):
            observed = _class_labels(validation.iloc[:, 0], model.classes_)
            predicted, scores, score_type = classification_prediction(model, aligned, positive, policy, cutoff)
            metrics = classification_metrics(observed, predicted, scores, model.classes_, positive, score_type)
            result['signed_binary_metrics'] = metrics if positive is not None else None
            bar = mkbar(pd.Series({k: metrics[k] for k in ('accuracy', 'precision', 'recall_sensitivity', 'f1')}))
            heatmap, annotation = mkheatmap(observed, predicted, list(model.classes_))
            roc = []
            if positive is not None and scores is not None and len(np.unique(observed)) == 2:
                raw = np.asarray(scores)
                positive_scores = raw[:, list(model.classes_).index(positive)] if raw.ndim == 2 else (
                    -raw if score_type == 'decision_function' and positive == model.classes_[0] else raw)
                fpr, tpr, _ = roc_curve(observed == positive, positive_scores)
                roc = [{'x': fpr.tolist(), 'y': tpr.tolist(), 'mode': 'lines', 'name': 'ROC for %s' % positive}]
            result['val_describe_roc'] = {'bar_dict': bar, 'heatmap_dict': heatmap,
                                          'heatmap_anno': annotation, 'valid_roc_traces': roc}
        elif method == 'model_reg':
            observed = pd.to_numeric(validation.iloc[:, 0], errors='raise').to_numpy()
            if not np.isfinite(observed).all():
                raise DataValidationError('Regression validation outcomes must be finite.')
            predicted = model.predict(aligned)
            metrics = regression_metrics(observed, predicted)
            report = pd.DataFrame([metrics], index=[bundle['name']])
            traces = []
            for n, key in enumerate(report.columns, 1):
                trace = mkvreportbarplot(report[[key]])[0]
                trace['xaxis'], trace['yaxis'] = 'x%d' % n, 'y%d' % n
                traces.append(trace)
            result['val_describe_roc'] = {'vreport_trace': traces,
                                         'vregpred_trace': mkvregpredplot([predicted], observed, 'validation')}
        else:
            target = validate_survival_target(validation.iloc[:, 0], validation.iloc[:, 1], require_event_mix=False)
            predicted = model.predict(aligned)
            observed = ['%d / %g' % (e, t) for e, t in zip(target['event'], target['time'])]
            metrics = {}
            try:
                metrics['c_index'] = float(concordance_index_censored(target['event'], target['time'], predicted)[0])
            except ValueError:
                result['prediction_warnings'].append('C-index is unavailable: this cohort has no comparable pairs.')
            result['prediction_warnings'].append(
                'Time-dependent AUC and integrated Brier score are unavailable in Predict: '
                'the signed export does not include the development censoring reference. '
                'Analysis reports these metrics when estimable. No validation-derived censoring distribution is substituted.')
        result['validation_reports_dict'] = _records(aligned.index, predicted, observed)
        result['signed_validation_metrics'] = metrics
        result['metric_rows'] = [{'name': {'pr_auc': 'Average precision (AP)',
                                          'pr_auc_macro': 'Macro average precision (AP)'}.get(k, k),
                                  'value': float(v)}
                                 for k, v in metrics.items()
                                 if isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v)]
    return result
