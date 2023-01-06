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
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
        print('projectid:', projectid)
        print('p:', STATIC_ROOT + '/cache/' + projectid + '/data.csv')
        print('p2:', STATIC_ROOT + '/cache/' + projectid + '/data.csv')
        if select_model == 'model_sur':
            blind_set, validation_set = pred_val_split(inputdata, datatype='survival')
        else:
            blind_set, validation_set = pred_val_split(inputdata)

        if select_model == 'model_bclass' or select_model == 'model_mclass':
            classes = model_pickle['classes']
            # blind_set = blind_set[5:20]  # 模拟的blind数据
            predict_reports = {}
            mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
            try:
                predict_reports[model_name] = model.predict(blind_set[feature_names])
            except:
                title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                messages.success(request, title)
                return HttpResponseRedirect("/maler/predict")


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

        shutil.rmtree(os.path.join(STATIC_ROOT, 'cache', projectid))
        return render(request, 'predict_result.html', {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': json.dumps(predict_reports_dict),
            'surv_plot': json.dumps(surv_plot),
        })
    else:
        return render(request, 'predict.html')

