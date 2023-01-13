import json

from django.shortcuts import render
from django.http import HttpResponseRedirect

import os, shutil, pickle
import numpy as np
import pandas as pd

from mlserver.views.classification_oc_result_views import JsonEncoder
from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5
from django.contrib import messages
from mlserver.views.predict_views import pred_val_split, sur_pred_plot
from mlserver.views.classification_oc_result_views import classification_process, JsonEncoder
from mlserver.views.regression_oc_result_views import regression_preprocess
from mlserver.views.classification_cp_result_views import pre_valid, mkbar, mkheatmap, valid_roc_info, mkroc
from mlserver.views.regression_cp_result_views import reg_cust_val, mkvregpredplot, mkvreportbarplot
from mlserver.views.survival_oc_result_views import sur_data_process, mk_surv_data, time_dependent_auc, mk_auc_line, mk_surv_layout
def predict_results(request, projectid):
    # predict
    if request.method == "POST":
        select_model = 'model_' + projectid.split('-')[1].lower()
        # if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'apickle.pkl')):
        #     with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'rb') as f:
        #         preview_pickle = pickle.load(f)
        #     if preview_pickle['status'] == 'Preview':
        #         print(preview_pickle['status'])
        #         preview_pickle['status'] = 'Running'
        #         with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'wb') as f:
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
        with open(STATIC_ROOT + '/cache/' + projectid + '/' + 'pickle.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
        # model
        method, model_name, model, feature_names = \
            model_pickle['method'], model_pickle['name'], model_pickle['model'], model_pickle['feature_names']
        f.close()
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0, sep=r'/|,|\t').T
        print('projectid:', projectid)
        print('p:', STATIC_ROOT + '/cache/' + projectid + '/data.csv')
        if select_model == 'model_sur':
            blind_set, validation_set = pred_val_split(inputdata, datatype='survival')
        else:
            blind_set, validation_set = pred_val_split(inputdata)
        if len(blind_set)>0:
            if select_model == 'model_bclass' or select_model == 'model_mclass':
                classes = model_pickle['classes']
                # blind_set = blind_set[5:20]  # 模拟的blind数据
                predict_reports = {}
                mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
                # print(blind_set[feature_names])
                print(blind_set[feature_names].dtypes)
                # try:
                #     predict_reports[model_name] = model.predict(blind_set[feature_names])
                # except:
                #     title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                #     messages.success(request, title)
                #     return HttpResponseRedirect("/maler/predict")
                predict_reports[model_name] = model.predict(blind_set[feature_names])

                predict_reports = pd.DataFrame(predict_reports, index=blind_set.index).applymap(lambda x: mapping[x])
                predict_reports_dict = predict_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
                showtable = True
                method = 'Classification'
                surv_plot = None
            elif select_model == 'model_reg':
                # blind_set = blind_set[5:20]  # 模拟的blind数据
                predict_reports = {}
                try:
                    predict_reports[model_name] = model.predict(blind_set[feature_names])
                except:
                    title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                    messages.success(request, title)
                    # return render(request, "predict.html")
                    return HttpResponseRedirect("/maler/predict")

                predict_reports = pd.DataFrame(predict_reports, index=blind_set.index)
                predict_reports_dict = predict_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
                showtable = True
                method = 'Regression'
                surv_plot = None
            else:
                # blind_set = blind_set[5:20]  # 模拟的blind数据
                try:
                    if model_name != 'SurvivalSVM':
                        surv_plot = sur_pred_plot(model, blind_set, feature_names)
                    else:
                        surv_plot = None
                        print('The svm model does not support the prediction function')
                except:
                    title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                    messages.success(request, title)
                    # return render(request, "predict.html")
                    return HttpResponseRedirect("/maler/predict")
                showtable = False
                predict_reports_dict = None
                method = 'Survival'
        else:
            showtable = False
            predict_reports_dict = None
            surv_plot = None
        #validation
        validation_reports_dict = None
        val_describe_roc = None
        if len(validation_set) > 0:
            validation_reports = {}
            if select_model == 'model_bclass' or select_model == 'model_mclass':
                method = 'Classification'
                classes = model_pickle['classes']
                mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
                validation_data, validation_label = classification_process(validation_set)
                validation_reports[model_name] = model.predict(validation_data[feature_names])
                validation_reports = pd.DataFrame(validation_reports, index=validation_data.index).applymap(
                    lambda x: mapping[x])
                validation_reports = pd.concat(
                    [validation_reports, pd.DataFrame(validation_label, index=validation_data.index)], axis=1)
                validation_reports_dict = validation_reports.reset_index().rename(
                    columns={'index': 'Name', model_name: 'Label', 0: 'Label2'}).to_dict('records')
                # validation chart
                validation_label2 = list(map(lambda y: classes[y], validation_label))
                validate_predict, validate_report = pre_valid(model, validation_data, validation_label2, feature_names)

                bar_dict = mkbar(validate_report)
                heatmap_dict, heatmap_anno = mkheatmap(validation_label2, validate_predict, classes)
                valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info('validation', model,
                                                                                       validation_data,
                                                                                       validation_label2,
                                                                                       feature_names)
                valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std, title=['validation'])
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
                validation_data, validation_label = regression_preprocess(validation_set)
                validation_reports[model_name] = model.predict(validation_data[feature_names])
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
                ytrain = model_pickle['ytrain']
                validation_data, validation_label = sur_data_process(validation_set)
                data_median = model.predict(pd.DataFrame(inputdata[feature_names].median()).T)[0]
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
                print('val_describe_roc:', val_describe_roc)



        return render(request, 'predict_result.html', {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': json.dumps(predict_reports_dict),
            'surv_plot': json.dumps(surv_plot),
            'validation_reports_dict': json.dumps(validation_reports_dict),
            'val_describe_roc': json.dumps(val_describe_roc, ensure_ascii=False, cls=JsonEncoder),
        })
    else:
        return render(request, 'predict.html')

