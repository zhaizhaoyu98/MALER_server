import json

from django.shortcuts import render
from django.http import HttpResponseRedirect

import os, shutil, pickle
import numpy as np
import pandas as pd

from mlserver.views.classification_oc_result_view_webscoket import JsonEncoder
from django.conf import settings
from mlserver.views.classification_oc_result_view_webscoket import get_file_md5
from django.contrib import messages
from mlserver.views.predict_views import pred_val_split, sur_pred_plot
from mlserver.views.preview_views import script_json, html_safe_records
from mlserver.views.classification_oc_result_view_webscoket import classification_process, JsonEncoder
from mlserver.views.regression_oc_result_view_websocket import regression_preprocess
from mlserver.views.classification_cp_result_view_websocket import pre_valid, mkbar, mkheatmap, valid_roc_info, mkroc
from mlserver.views.regression_cp_result_view_websocket import reg_cust_val, mkvregpredplot, mkvreportbarplot
from mlserver.views.survival_oc_result_view_websocket import sur_data_process, mk_surv_data, time_dependent_auc, mk_auc_line, mk_surv_layout
from mlserver.views.predict_views import _bundle_as_legacy_dict
from mlserver.safe_ml import DataValidationError, ModelBundleError, align_prediction_frame, alignment_report
from mlserver.safe_ml import classification_metrics, classification_prediction
from sklearn.metrics import roc_curve
from mlserver.validated_analysis import validate_sample_header
from mlserver.task_access import require_task_access


def _display_class(value, reverse_mapping):
    """Return a human-readable class without failing on string-trained models."""
    return reverse_mapping.get(value, value)


def _note_alignment(notices, cohort, frame, feature_names):
    """记录预测矩阵相对于训练特征契约的偏差.

    Args:
        notices (list, 必填): 收集偏差报告的列表.
        cohort (str, 必填): 数据来源标签, 例如 blind 或 validation.
        frame (pandas.DataFrame, 必填): 待检查的预测矩阵.
        feature_names (iterable, 必填): 训练时固定的特征名与顺序.

    Returns:
        Not Available

    See Also:
        alignment_report
    """
    report = alignment_report(frame, feature_names, cohort=cohort)
    if report:
        notices.append(report)


def _signed_predict_response(request, projectid, bundle_path, access_token):
    from mlserver.signed_prediction import signed_prediction_result
    directory = os.path.join(settings.MALER_TASK_ROOT, projectid)
    cache = os.path.join(directory, 'predict_pickle.pkl')
    payload = None
    if os.path.exists(cache):
        with open(cache, 'rb') as handle:
            stored = pickle.load(handle)
        if stored.get('predict_schema_version') == 1:
            payload = stored
    if payload is None:
        try:
            bundle = _bundle_as_legacy_dict(bundle_path)
            payload = signed_prediction_result(projectid, bundle, os.path.join(directory, 'data.csv'))
        except (DataValidationError, ModelBundleError, ValueError, OSError) as exc:
            messages.error(request, str(exc))
            return HttpResponseRedirect('/maler/predict')
        with open(cache + '.tmp', 'wb') as handle:
            pickle.dump(payload, handle)
        os.replace(cache + '.tmp', cache)
    context = dict(payload)
    context['access_token'] = access_token
    for key in ('predict_reports_dict', 'validation_reports_dict'):
        context[key] = script_json(html_safe_records(payload[key]))
    for key in ('surv_plot', 'val_describe_roc'):
        context[key] = script_json(payload[key], ensure_ascii=False, cls=JsonEncoder)
    return render(request, 'predict_result.html', context)


def predict_results(request, projectid):
    _, access_token = require_task_access(request, projectid)
    bundle_path = os.path.join(settings.MALER_TASK_ROOT, projectid, 'model.maler')
    if os.path.exists(bundle_path):
        return _signed_predict_response(request, projectid, bundle_path, access_token)
    # predict
    # if request.method == "POST":
    select_model = 'model_' + projectid.split('-')[1].lower()
    if not os.path.exists(os.path.join(settings.MALER_TASK_ROOT, projectid, 'predict_pickle.pkl')):
        # with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'rb') as f:
        #     preview_pickle = pickle.load(f)
        # if preview_pickle['status'] == 'Preview':
        #     print(preview_pickle['status'])
        #     preview_pickle['status'] = 'Running'
        #     with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'wb') as f:
        #         pickle.dump(preview_pickle, f)
        # else:
        #     status = 'Running'
        #     print(preview_pickle['status'])
        #     form_action = preview_pickle['form_action']
        #     display_samples_dict = preview_pickle['display_samples_dict']
        #     hist_trace = preview_pickle['hist_trace']
        #     inputdata_display = preview_pickle['inputdata_display']
        #     inputdata_columns = preview_pickle['inputdata_columns']
        #     title_str = preview_pickle['title_str']
        #     return render(request, 'status.html', {
        #         'projectid': projectid,
        #         # 'model_md5': model_md5,
        #         'form_action': form_action,
        #         'display_samples_dict': json.dumps(display_samples_dict),
        #         'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
        #         'inputdata_display': json.dumps(inputdata_display),
        #         'inputdata_columns': json.dumps(inputdata_columns),
        #         'title_str': title_str,
        #         'status': status,
        #     })
        bundle_path = os.path.join(settings.MALER_TASK_ROOT, projectid, 'model.maler')
        is_signed_bundle = os.path.exists(bundle_path)
        try:
            if is_signed_bundle:
                model_pickle = _bundle_as_legacy_dict(bundle_path)
            else:
                # Application-controlled legacy examples only. Arbitrary user
                # pickle uploads are rejected by predict_preview.
                with open(settings.MALER_TASK_ROOT + '/' + projectid + '/' + 'pickle.pkl', 'rb') as f:
                    model_pickle = pickle.load(f)
        except (OSError, ModelBundleError) as exc:
            messages.error(request, str(exc))
            return HttpResponseRedirect("/maler/predict")
        # model
        method, model_name, model, feature_names = \
            model_pickle['method'], model_pickle['name'], model_pickle['model'], model_pickle['feature_names']
        positive_class = None
        threshold_policy = model_pickle.get('threshold_policy', 'estimator_native_predict')
        threshold_value = model_pickle.get('threshold_value', 0.5)
        if is_signed_bundle and hasattr(model, 'classes_') and len(model.classes_) == 2:
            matches = [value for value in model.classes_
                       if str(value) == str(model_pickle.get('positive_class'))]
            if len(matches) != 1:
                messages.error(request, 'The bundle has no valid positive class.')
                return HttpResponseRedirect('/maler/predict')
            positive_class = matches[0]
        input_path = settings.MALER_TASK_ROOT + '/' + projectid + '/data.csv'
        try:
            with open(input_path, 'r', encoding='utf-8-sig', newline='') as handle:
                validate_sample_header(handle.readline())
        except (UnicodeDecodeError, ValueError) as exc:
            messages.error(request, str(exc))
            return HttpResponseRedirect('/maler/predict')
        inputdata = pd.read_csv(input_path, header=0, index_col=0, sep=r'/|,|\t').T
        # 预测矩阵与训练特征契约的偏差报告
        alignment_notices = []
        if select_model == 'model_sur':
            blind_set, validation_set = pred_val_split(inputdata, datatype='survival')
        else:
            blind_set, validation_set = pred_val_split(inputdata,datatype=select_model)
        if len(blind_set)>0:
            if select_model == 'model_bclass' or select_model == 'model_mclass':
                classes = model_pickle['classes']
                # blind_set = blind_set[5:20]  # 模拟的blind数据
                predict_reports = {}
                mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
                # print(blind_set[feature_names])
                try:
                    aligned_blind, unexpected_features = align_prediction_frame(blind_set, feature_names)
                    _note_alignment(alignment_notices, "blind", blind_set, feature_names)
                    if is_signed_bundle:
                        predict_reports[model_name], _, _ = classification_prediction(
                            model, aligned_blind, positive_class, threshold_policy,
                            threshold_value)
                    else:
                        predict_reports[model_name] = model.predict(aligned_blind)
                except DataValidationError as exc:
                    messages.error(request, str(exc))
                    return HttpResponseRedirect("/maler/predict")
                # predict_reports[model_name] = model.predict(blind_set[feature_names])

                predict_reports = pd.DataFrame(predict_reports, index=blind_set.index).applymap(
                    lambda x: _display_class(x, mapping))
                predict_reports_dict = predict_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
                showtable = True
                method = 'Classification'
                surv_plot = None
            elif select_model == 'model_reg':
                # blind_set = blind_set[5:20]  # 模拟的blind数据
                predict_reports = {}
                try:
                    aligned_blind, unexpected_features = align_prediction_frame(blind_set, feature_names)
                    _note_alignment(alignment_notices, "blind", blind_set, feature_names)
                    predict_reports[model_name] = model.predict(aligned_blind)
                except DataValidationError as exc:
                    messages.error(request, str(exc))
                    # return render(request, "predict.html")
                    return HttpResponseRedirect("/maler/predict")

                predict_reports = pd.DataFrame(predict_reports, index=blind_set.index)
                predict_reports_dict = predict_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
                showtable = True
                method = 'Regression'
                surv_plot = None
            else:
                try:
                    aligned_blind, unexpected_features = align_prediction_frame(blind_set, feature_names)
                    _note_alignment(alignment_notices, "blind", blind_set, feature_names)
                    if model_name != 'SurvivalSVM':
                        surv_plot = sur_pred_plot(model, aligned_blind, feature_names)
                    else:
                        surv_plot = None
                        # This estimator has no survival-curve prediction method.
                except DataValidationError as exc:
                    messages.error(request, str(exc))
                    # return render(request, "predict.html")
                    return HttpResponseRedirect("/maler/predict")
                showtable = False
                predict_reports_dict = None
                method = 'Survival'
        else:
            showtable = False
            predict_reports_dict = None
            surv_plot = None
        # print('predict_reports_dict: ',predict_reports_dict)
        #validation
        validation_reports_dict = None
        val_describe_roc = None
        signed_binary_metrics = None
        if len(validation_set) > 0:
            validation_reports = {}
            if select_model == 'model_bclass' or select_model == 'model_mclass':
                method = 'Classification'
                classes = model_pickle['classes']
                mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
                try:
                    if is_signed_bundle:
                        validation_label = np.asarray(validation_set.iloc[:, 0]).ravel()
                        validation_data = validation_set.iloc[:, 1:]
                    else:
                        validation_data, validation_label = classification_process(validation_set)
                    _note_alignment(alignment_notices, "validation", validation_data, feature_names)
                    validation_data, unexpected_features = align_prediction_frame(
                        validation_data, feature_names)
                    if is_signed_bundle:
                        validation_reports[model_name], validation_scores, score_type = classification_prediction(
                            model, validation_data, positive_class, threshold_policy,
                            threshold_value)
                        validation_prediction = np.asarray(validation_reports[model_name])
                    else:
                        validation_reports[model_name] = model.predict(validation_data)
                except DataValidationError as exc:
                    messages.error(request, str(exc))
                    return HttpResponseRedirect("/maler/predict")
                validation_reports = pd.DataFrame(validation_reports, index=validation_data.index).applymap(
                    lambda x: _display_class(x, mapping))
                validation_reports = pd.concat(
                    [validation_reports, pd.DataFrame(validation_label, index=validation_data.index)], axis=1)
                validation_reports_dict = validation_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label', 0: 'Label2'}).to_dict('records')
                # validation chart
                if is_signed_bundle:
                    validation_label2 = list(validation_label)
                    plot_classes = list(np.unique(np.concatenate([
                        np.asarray(validation_label2),
                        np.asarray(model.predict(validation_data)),
                    ])))
                else:
                    validation_label2 = list(map(lambda y: classes[y], validation_label))
                    plot_classes = list(classes)
                if is_signed_bundle:
                    validate_predict = validation_prediction
                    if positive_class is not None:
                        signed_binary_metrics = classification_metrics(
                            validation_label, validate_predict, scores=validation_scores,
                            classes=model.classes_, positive_class=positive_class,
                            score_type=score_type)
                        validate_report = pd.Series({
                            key: signed_binary_metrics[key]
                            for key in ('precision', 'recall_sensitivity', 'f1', 'accuracy')})
                    else:
                        validate_report = pd.Series({'accuracy': signed_binary_metrics['accuracy']}) if signed_binary_metrics else pd.Series({
                            'accuracy': float(np.mean(np.asarray(validation_label) == validate_predict))})
                else:
                    validate_predict, validate_report = pre_valid(
                        model, validation_data, validation_label2, feature_names)

                bar_dict = mkbar(validate_report)
                heatmap_dict, heatmap_anno = mkheatmap(validation_label2, validate_predict, plot_classes)
                if len(plot_classes) == 2:
                    if is_signed_bundle and positive_class is not None and validation_scores is not None:
                        score_array = np.asarray(validation_scores)
                        if score_array.ndim == 2:
                            positive_scores = score_array[:, list(model.classes_).index(positive_class)]
                        elif score_type == 'decision_function' and positive_class == model.classes_[0]:
                            positive_scores = -score_array
                        else:
                            positive_scores = score_array
                        fpr, tpr, _ = roc_curve(
                            np.asarray(validation_label) == positive_class, positive_scores)
                        valid_roc_traces = [{'x': fpr.tolist(), 'y': tpr.tolist(),
                                             'mode': 'lines', 'name': 'ROC for %s' % positive_class,
                                             'type': 'scatter'}]
                    else:
                        valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info(
                            model_name, model, validation_data, validation_label2, feature_names)
                        valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df,
                                                 valid_auc_mean_std, title=['validation'])
                else:
                    # The legacy plotting helper only implements binary ROC.
                    valid_roc_traces = []
                val_describe_roc = {
                    # validation
                    "bar_dict": (bar_dict),
                    "heatmap_dict": (heatmap_dict),
                    "heatmap_anno": (heatmap_anno),
                    "valid_roc_traces": (valid_roc_traces),
                    # 'line_chart_data': line_chart_data
                }
            elif select_model == 'model_reg':
                method = 'Regression'
                try:
                    if is_signed_bundle:
                        validation_label = pd.to_numeric(
                            validation_set.iloc[:, 0], errors='raise').to_numpy()
                        validation_data = validation_set.iloc[:, 1:]
                    else:
                        validation_data, validation_label = regression_preprocess(validation_set)
                    _note_alignment(alignment_notices, "validation", validation_data, feature_names)
                    validation_data, unexpected_features = align_prediction_frame(
                        validation_data, feature_names)
                    validation_reports[model_name] = model.predict(validation_data)
                except (DataValidationError, ValueError) as exc:
                    messages.error(request, str(exc))
                    return HttpResponseRedirect("/maler/predict")
                validation_reports = pd.concat(
                    [pd.DataFrame(validation_reports, index=validation_data.index), pd.DataFrame(validation_label, index=validation_data.index)], axis=1)
                validation_reports_dict = validation_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label', 0: 'Label2'}).to_dict('records')

                val_report = np.round(reg_cust_val([model], validation_data, validation_label, feature_names, 'validation'), 3)
                # val_report_dict = val_report.reset_index().rename(columns={'index': 'Method'}).to_dict('records')
                vreport_trace = []
                i = 0
                for m in list(val_report.columns):
                    subtrace = mkvreportbarplot(val_report[[m]])
                    subtrace[0]['xaxis'], subtrace[0]['yaxis'] = 'x' + str(i + 1), 'y' + str(i + 1)
                    i += 1
                    vreport_trace.append(subtrace[0])

                validate_predict = model.predict(validation_data[feature_names])
                vregpred_trace = mkvregpredplot([validate_predict], validation_label, 'validation')

                val_describe_roc = {
                    # validation
                    "vreport_trace": vreport_trace,
                    "vregpred_trace": vregpred_trace,
                }
            else:
                method = 'Survival'
                name = model_pickle['name']
                ytrain = model_pickle.get('ytrain', getattr(model, 'maler_ytrain_', None))
                validation_data, validation_label = sur_data_process(validation_set)
                try:
                    _note_alignment(alignment_notices, "validation", validation_data, feature_names)
                    validation_data, unexpected_features = align_prediction_frame(
                        validation_data, feature_names)
                    _note_alignment(alignment_notices, "input", inputdata, feature_names)
                    aligned_input, unexpected_features = align_prediction_frame(
                        inputdata, feature_names)
                    data_median = model.predict(pd.DataFrame(aligned_input.median()).T)[0]
                    if ytrain is None:
                        raise DataValidationError(
                            'The survival bundle does not contain the training outcome '
                            'required for time-dependent AUC validation.')
                except DataValidationError as exc:
                    messages.error(request, str(exc))
                    return HttpResponseRedirect("/maler/predict")
                vsurv_trace, vresultp = mk_surv_data(name, validation_data[feature_names], validation_label,
                                                     model, data_median)
                vsurv_layout = mk_surv_layout(name, vresultp)
                vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

                vlinedata = []
                va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(model, validation_data,
                                                                         validation_label, ytrain,
                                                                         feature_names, name)
                vlinetrace = mk_auc_line(name, va_times, rsf_auc, mean_auc, cindex)
                vlinedata.append(vlinetrace)

                val_describe_roc = {
                    # validation
                    'vsurv_data': vsurv_data,
                    'vlinedata': vlinedata,
                }
        predict_pickle = {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': predict_reports_dict,
            'surv_plot': surv_plot,
            'validation_reports_dict': validation_reports_dict,
            'val_describe_roc': val_describe_roc,
            'alignment_notices': alignment_notices,
            'signed_binary_metrics': signed_binary_metrics,
            'positive_class': str(positive_class) if positive_class is not None else None,
            'threshold_policy': threshold_policy if positive_class is not None else None,
            'threshold_value': threshold_value if positive_class is not None else None,
        }

        with open(settings.MALER_TASK_ROOT + '/' + projectid + '/predict_pickle.pkl',
                  'wb') as f:
            pickle.dump(predict_pickle, f)

        return render(request, 'predict_result.html', {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': script_json(html_safe_records(predict_reports_dict)),
            'surv_plot': script_json(surv_plot),
            'validation_reports_dict': script_json(html_safe_records(validation_reports_dict)),
            'val_describe_roc': script_json(val_describe_roc, ensure_ascii=False, cls=JsonEncoder),
            'alignment_notices': alignment_notices,
            'signed_binary_metrics': signed_binary_metrics,
            'positive_class': str(positive_class) if positive_class is not None else None,
            'threshold_policy': threshold_policy if positive_class is not None else None,
            'threshold_value': threshold_value if positive_class is not None else None,
        })
    else:
        with open(settings.MALER_TASK_ROOT + '/' + projectid + '/predict_pickle.pkl', 'rb') as f:
            predict_pickle = pickle.load(f)
        projectid, method, showtable, predict_reports_dict, surv_plot, validation_reports_dict, val_describe_roc \
            = predict_pickle['projectid'], \
              predict_pickle['method'], \
              predict_pickle['showtable'], \
              predict_pickle['predict_reports_dict'], \
              predict_pickle['surv_plot'], \
              predict_pickle['validation_reports_dict'], \
              predict_pickle['val_describe_roc']
        alignment_notices = predict_pickle.get('alignment_notices', [])
        return render(request, 'predict_result.html', {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': script_json(html_safe_records(predict_reports_dict)),
            'surv_plot': script_json(surv_plot),
            'validation_reports_dict': script_json(html_safe_records(validation_reports_dict)),
            'val_describe_roc': script_json(val_describe_roc, ensure_ascii=False, cls=JsonEncoder),
            'alignment_notices': alignment_notices,
            'signed_binary_metrics': predict_pickle.get('signed_binary_metrics'),
            'positive_class': predict_pickle.get('positive_class'),
            'threshold_policy': predict_pickle.get('threshold_policy'),
            'threshold_value': predict_pickle.get('threshold_value'),
        })
    # else:
    #     print('get!!!')
        # if not os.path.exists(os.path.join(settings.MALER_TASK_ROOT, projectid, 'predict_pickle.pkl')):
        #     with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'rb') as f:
        #         preview_pickle = pickle.load(f)
        #     if preview_pickle['status'] == 'Preview':
        #         print(preview_pickle['status'])
        #         preview_pickle['status'] = 'Running'
        #         with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'wb') as f:
        #             pickle.dump(preview_pickle, f)
        #     else:
        #         status = 'Running'
        #         print(preview_pickle['status'])
        #         form_action = preview_pickle['form_action']
        #         display_samples_dict = preview_pickle['display_samples_dict']
        #         hist_trace = preview_pickle['hist_trace']
        #         inputdata_display = preview_pickle['inputdata_display']
        #         inputdata_columns = preview_pickle['inputdata_columns']
        #         title_str = preview_pickle['title_str']
        #         return render(request, 'status.html', {
        #             'projectid': projectid,
        #             # 'model_md5': model_md5,
        #             'form_action': form_action,
        #             'display_samples_dict': json.dumps(display_samples_dict),
        #             'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
        #             'inputdata_display': json.dumps(inputdata_display),
        #             'inputdata_columns': json.dumps(inputdata_columns),
        #             'title_str': title_str,
        #             'status': status,
        #         })
