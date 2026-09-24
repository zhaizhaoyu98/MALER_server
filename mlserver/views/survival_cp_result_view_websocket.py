from django.shortcuts import render

import os, shutil, copy, pickle, json, random, string
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import cross_val_score,cross_validate, GridSearchCV, KFold,StratifiedKFold,RepeatedKFold
from sksurv.datasets import get_x_y
from sksurv.svm import FastKernelSurvivalSVM,FastSurvivalSVM
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest,ExtraSurvivalTrees,GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.statistics import logrank_test

from ML_WebServer.settings import STATIC_ROOT

from mlserver.views.classification_oc_result_view_webscoket import get_file_md5, df2bp, split_train_test, JsonEncoder,task_sendmail
from mlserver.views.classification_cp_result_view_websocket import md5_convert, surv_para_group,grid_para_split
from mlserver.views.survival_oc_result_view_websocket import sur_data_process, cox_selection, train_top3,pre_screening,\
    sur_RSKFold, train_estimator, mk_surv_data,mk_surv_layout, time_dependent_auc, mk_auc_line
from mlserver.views.featureselection_method import FSS_fun,BSS_fun
import warnings
from dwebsocket.decorators import accept_websocket
from concurrent.futures.thread import ThreadPoolExecutor
pools = ThreadPoolExecutor(100)
warnings.filterwarnings("ignore")
import time

def return_running_page(request,projectid):
    fsm = request.POST.get('fsm')
    form_action = request.POST.get('form_action')
    model_md5 = request.POST.get('model_md5')
    feature_select_method = request.POST.get('feature_select_method')
    to_mail = request.POST.get('to_mail')
    fn = request.POST.get('feature_norm')

    return render(request, 'survival_cp_result_ws.html', {
        'fsm': fsm,
        'form_action': form_action,
        'projectid': projectid,
        'model_md5': model_md5,
        'feature_select_method': feature_select_method,
        'to_mail': to_mail,
        'feature_norm': fn,
    })

def data_analysis(WebSocket,client_msg,projectid):
    # client_msg = json.loads(WebSocket.wait())
    # client_msg = str(client_msg, encoding="utf-8")
    # print('data_analysis: ', client_msg)
    # print(projectid)
    # client_msg['status'] = 1
    # time.sleep(5)
    # WebSocket.send(json.dumps(client_msg))
    print('start analysis')
    analysis_results = cp_sur_analysis(client_msg,projectid)
    analysis_results['status'] = 1
    WebSocket.send(json.dumps(analysis_results))
    print('finish!!')

@accept_websocket
def result_ws(request, projectid):
    if request.is_websocket():
        print('websocket on !!')
        WebSocket = request.websocket
        while True:
            if WebSocket.has_messages():
                client_msg = json.loads(WebSocket.wait())
                if client_msg != 'heartbeat':
                    print(client_msg)
                    # task1 = pools.submit(data_analysis,WebSocket,client_msg,projectid)
                    data_analysis(WebSocket,client_msg,projectid)
                    # print(task1.result())
                    # pools.shutdown()
                elif client_msg == 'heartbeat':
                    messages = {
                        'time': time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                        'status': 0,
                    }
                    time.sleep(2)
                    request.websocket.send(json.dumps(messages))

def cp_sur_analysis(client_msg,projectid):
    # try:
        feature_select_method = client_msg['feature_select_method']
        model_md5 = client_msg['model_md5']
        fsm = client_msg['fsm']
        fn = client_msg['feature_norm']
        with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
            model_set = pickle.load(f)
        print(model_set)
        sur_model, sur_model_name = model_set[model_md5]['model'], model_set[model_md5]['model_name']
        gridsearch_para = model_set[model_md5]['gridsearch_para']

        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'cp_cache.pkl')):
            if fn == 'N':
                scaler = None
                inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
                train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')
                x2, y2 = sur_data_process(train_set)
                if len(test_set) > 0:
                    validation_data, validation_label = sur_data_process(test_set)
                    ifval = True
                else:
                    ifval = False
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/normalization_data.pkl', 'rb') as f:
                    norm_info = pickle.load(f)
                x2, y2 = norm_info['train_set'], norm_info['train_set_label']
                ifval = norm_info['ifval']
                validation_data, validation_label = norm_info['test_set'], norm_info['test_set_label']
                scaler = norm_info['scaler']
                print('using normalized data to analysis!')

            cv = KFold(n_splits=5, shuffle=True, random_state=10)
            features = cox_selection(x2, y2)
            x3 = x2[features]
            train_index, test_index = sur_RSKFold(x3, y2)
            if feature_select_method != 'TopK':
                if feature_select_method == 'FSS':
                    sf, ms = FSS_fun(features, sur_model, x3, y2, cv, n_jobs=4)
                else:
                    sf, ms = BSS_fun(features, sur_model, x3, y2, cv, n_jobs=4)
                max_index = np.array(ms).argmax()
                max_score = max(ms)
                max_features = (sf[:max_index + 1])
                preds, tests, estimators = [], [], []
                for i in range(len(train_index)):
                    xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                    xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                    xtrain, xtest = xtrain[max_features], xtest[max_features]
                    estimator, test_acc, predict = train_estimator(sur_model, xtrain, ytrain, xtest, ytest)
                    tests.append(test_acc), estimators.append(estimator), preds.append(predict)
            else:
                clf_num, ms = pre_screening(x3, y2, sur_model, features)
                tests, estimators, mean_accs, preds, res = train_top3(sur_model, x3, y2, clf_num, train_index,
                                                                                 test_index, features)
                max_features = res



            test_acc_reports = pd.DataFrame(data=tests)
            test_acc_reports.columns = [sur_model_name]
            test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :], 3)

            # sur_pickle = {'x':x3[max_features],'y':y2}
            # with open('C:/Users/HP/Desktop/fsdownload/' + '/sur_data.pkl',
            #           'wb') as f:
            #     pickle.dump(sur_pickle, f)
            # 网格搜索gridSearchCV
            best_para = None
            if len(gridsearch_para) > 0:
                start = time.perf_counter()
                #构建tau截断时间
                lower, upper = np.percentile(y2['time'], [0, 100])
                sur_times = np.arange(lower, upper + 1)

                grid_search = gridsearch_bulid(sur_model, gridsearch_para, sur_model_name,sur_times)
                grid_search.fit(x3[max_features], y2)
                print("网格搜索最优参数：", grid_search.best_params_)
                print("网格搜索最优得分：", grid_search.best_score_)
                end = time.perf_counter()
                print('gridserach time: ', round(end - start, 2))
                # 比较
                grid_preds, grid_tests, grid_estimators = [], [], []

                for i in range(len(train_index)):
                    xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                    xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                    xtrain, xtest = xtrain[max_features], xtest[max_features]
                    grid_estimator, test_acc, predict = train_estimator(grid_search.best_estimator_.estimator, xtrain,
                                                                   ytrain, xtest, ytest)
                    grid_tests.append(test_acc), grid_estimators.append(grid_estimator), grid_preds.append(
                        predict)
                grid_acc_reports = pd.DataFrame(data=grid_tests)
                grid_describe = np.round(grid_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :], 3)
                grid_describe.columns = [sur_model_name]
                # 判断
                if grid_describe.loc['mean', sur_model_name] > test_acc_describe.loc['mean', sur_model_name]:
                    print('using gridsearch para')
                    test_acc_describe, tests, estimators = grid_preds, grid_tests, grid_estimators
                    f_describe = grid_describe
                elif grid_describe.loc['mean', sur_model_name] == test_acc_describe.loc['mean', sur_model_name]:
                    if grid_describe.loc['mean', sur_model_name] > test_acc_describe.loc['mean', sur_model_name]:
                        print('using gridsearch para')
                        preds, tests, estimators = grid_preds, grid_tests, grid_estimators
                        test_acc_describe_ = grid_describe
                    else:
                        print('raw')
                else:
                    print('raw')
                best_para = grid_search.best_params_
            test_acc_reports_dict = df2bp(test_acc_reports)
            test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
            test_acc_describe_dict = test_acc_describe_.to_dict('records')

            tmodels = copy.deepcopy(estimators[0])
            tmodels.fit(x3[max_features], y2)

            line_chart_data = []
            line_trace = {
                'mode': 'lines+markers',
                'name': sur_model_name,
                'type': 'scatter',
                'x': list(range(1, len(ms)+1)),
                'y': ms
            }
            line_chart_data.append(line_trace)

            parameter, test_acc, best_esti = [], [], []
            feature_names = []
            best_esti.append(tmodels)
            parameter.append(str(tmodels.get_params()))
            test_acc.append(test_acc_describe.iloc[0, 0])
            feature_names.append(str(max_features))
            final_reports = {'Mean C-index': test_acc,
                             'parameter': parameter,
                             'feature_names': feature_names, }
            final_reports = pd.DataFrame(final_reports, index=[sur_model_name]).reset_index().rename(
                columns={'index': 'Method'})
            final_reports_dict = final_reports.to_dict('records')

            data_median = tmodels.predict(pd.DataFrame(x3[max_features].median()).T)[0]
            surv_trace, resultp = mk_surv_data(sur_model_name, x3[max_features],y2,best_esti[0],data_median)
            surv_layout = mk_surv_layout(sur_model_name, resultp)
            surv_data = {'surv_trace': surv_trace, 'surv_layout': surv_layout}

            # validation
            vsurv_data,vlinedata = {}, []
            if len(validation_data)>0:
                vsurv_trace, vresultp = mk_surv_data(sur_model_name, validation_data[max_features],validation_label,best_esti[0],data_median)
                vsurv_layout = mk_surv_layout(sur_model_name, vresultp)
                vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

                vlinedata = []
                va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(best_esti[0], validation_data,
                                                                         validation_label, y2,
                                                                         max_features, sur_model_name)

                vlinetrace = mk_auc_line(sur_model_name, va_times, rsf_auc, mean_auc, cindex)
                vlinedata.append(vlinetrace)
            # pickle
            surv_pickle = {
                'sur_model_name': sur_model_name,
                'line_chart_data': line_chart_data,
                'test_acc_reports_dict': test_acc_reports_dict,
                'test_acc_describe_dict': test_acc_describe_dict,
                'final_reports_dict': final_reports_dict,
                'surv_data': surv_data,
                'vsurv_data': vsurv_data,
                'vlinedata': vlinedata,
                'ifval': ifval
            }
            # make cache
            cp_cache = {}
            para_str = feature_select_method + final_reports['Method'][0] + str(final_reports['parameter'][0]) +  \
                       str(gridsearch_para) + fn
            para_md5 = md5_convert(para_str)[:6]
            # add parameter md5 and feature select method
            final_reports['md5'], final_reports['fsm'] = para_md5, feature_select_method
            final_reports['grid_para'] = str(best_para)
            final_reports['grid_list'] = str(gridsearch_para)
            # normalization
            final_reports['fn'] = fn
            final_reports['scaler'] = str(scaler)
            final_reports_dict = final_reports.to_dict('records')

            cp_cache[para_md5] = surv_pickle
            cp_cache['reports'] = final_reports

            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                      'wb') as f:
                pickle.dump(cp_cache, f)

            #单个模型下载
            model_info = {}
            model_info['name'], model_info['model'], model_info['feature_names'] = sur_model_name, tmodels, max_features
            model_info['scaler'] = scaler
            with open(STATIC_ROOT + '/cache/' + projectid + '/' + para_md5 + '.pkl',
                      'wb') as f:
                pickle.dump(model_info, f)

        else:
            # 存在缓存时
            print('存在缓存！！！')
            # load pickle 加载缓存数据
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
            # 判断是否已经跑过该数据,如果是直接返回数据
            pd_reports = pd.DataFrame(cp_cache['reports'])
            for i in range(len(pd_reports.index)):
                select_md5 = 0
                pd_report = pd_reports.iloc[i, :]
                if (pd_report['parameter'] + pd_report['fsm'] + pd_report['grid_list'] + pd_report['fn']) == (
                        str(sur_model.get_params()) + feature_select_method + str(gridsearch_para) + fn):
                    select_md5 = pd_report['md5']
                    print('using cache!!!')
                    break
            if select_md5 != 0:
                final_reports_dict = cp_cache['reports'].to_dict('records')
                sur_model_name = cp_cache[select_md5]['sur_model_name']
                line_chart_data = cp_cache[select_md5]['line_chart_data']
                test_acc_reports_dict = cp_cache[select_md5]['test_acc_reports_dict']
                test_acc_describe_dict = cp_cache[select_md5]['test_acc_describe_dict']
                surv_data = cp_cache[select_md5]['surv_data']
                vsurv_data = cp_cache[select_md5]['vsurv_data']
                vlinedata = cp_cache[select_md5]['vlinedata']
                if len(cp_cache[select_md5]['vsurv_data']) != 0:
                    ifval = True
                else:
                    ifval = False
            else:
                print('using cache fail ! ')
                with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                    cp_cache = pickle.load(f)
                if fn == 'N':
                    inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
                    train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')
                    x2, y2 = sur_data_process(train_set)
                    scaler = 'None'
                    if len(test_set) > 0:
                        validation_data, validation_label = sur_data_process(test_set)
                        ifval = True
                    else:
                        ifval = False
                else:
                    with open(STATIC_ROOT + '/cache/' + projectid + '/normalization_data.pkl', 'rb') as f:
                        norm_info = pickle.load(f)
                    x2, y2 = norm_info['train_set'], norm_info['train_set_label']
                    ifval = norm_info['ifval']
                    validation_data, validation_label = norm_info['test_set'], norm_info['test_set_label']
                    scaler = norm_info['scaler']
                    print('using normalized data to analysis!')
                # inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
                # train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')
                #
                # x2, y2 = sur_data_process(train_set)
                # if len(test_set) > 0:
                #     validation_data, validation_label = sur_data_process(test_set)
                #     ifval = True
                # else:
                #     ifval = False
                # x2, vaildation_data, y2, vaildation_label = train_test_split(x, y, random_state=10, train_size=0.7,
                #                                                              stratify=y['Status'])
                cv = KFold(n_splits=5, shuffle=True, random_state=10)
                features = cox_selection(x2, y2)
                # features,ss2 = cox_selection(x2,y2)
                x3 = x2[features]
                train_index, test_index = sur_RSKFold(x3, y2)
                if feature_select_method != 'TopK':
                    if feature_select_method == 'FSS':
                        sf, ms = FSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
                    else:
                        sf, ms = BSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
                    max_index = np.array(ms).argmax()
                    max_score = max(ms)
                    max_features = (sf[:max_index + 1])

                    preds, tests, estimators = [], [], []
                    for i in range(len(train_index)):
                        xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                        xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                        xtrain, xtest = xtrain[max_features], xtest[max_features]
                        estimator, test_acc, predict = train_estimator(sur_model, xtrain, ytrain, xtest, ytest)
                        tests.append(test_acc), estimators.append(estimator), preds.append(predict)
                else:
                    clf_num, ms = pre_screening(x3, y2, sur_model, features)
                    tests, estimators, mean_accs, preds, res = train_top3(sur_model, x3, y2, clf_num, train_index,
                                                                          test_index, features)
                    max_features = res

                if select_md5 not in cp_cache.keys():

                    test_acc_reports = pd.DataFrame(data=tests)
                    test_acc_reports.columns = [sur_model_name]
                    test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :], 3)
                    # 网格搜索gridSearchCV
                    best_para = None
                    if len(gridsearch_para) > 0:
                        start = time.perf_counter()
                        # 构建tau截断时间
                        lower, upper = np.percentile(y2['time'], [0, 100])
                        sur_times = np.arange(lower, upper + 1)

                        grid_search = gridsearch_bulid(sur_model, gridsearch_para, sur_model_name, sur_times)
                        grid_search.fit(x3[max_features], y2)
                        print("网格搜索最优参数：", grid_search.best_params_)
                        print("网格搜索最优得分：", grid_search.best_score_)
                        end = time.perf_counter()
                        print('gridserach time: ', round(end - start, 2))
                        # 比较
                        grid_preds, grid_tests, grid_estimators = [], [], []

                        for i in range(len(train_index)):
                            xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                            xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                            xtrain, xtest = xtrain[max_features], xtest[max_features]
                            grid_estimator, test_acc, predict = train_estimator(grid_search.best_estimator_.estimator,
                                                                                xtrain,
                                                                                ytrain, xtest, ytest)
                            grid_tests.append(test_acc), grid_estimators.append(grid_estimator), grid_preds.append(
                                predict)
                        grid_acc_reports = pd.DataFrame(data=grid_tests)
                        grid_describe = np.round(grid_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :], 3)
                        grid_describe.columns = [sur_model_name]
                        # 判断
                        if grid_describe.loc['mean', sur_model_name] > test_acc_describe.loc['mean', sur_model_name]:
                            print('using gridsearch para')
                            test_acc_describe, tests, estimators = grid_preds, grid_tests, grid_estimators
                            f_describe = grid_describe
                        elif grid_describe.loc['mean', sur_model_name] == test_acc_describe.loc['mean', sur_model_name]:
                            if grid_describe.loc['mean', sur_model_name] > test_acc_describe.loc[ 'mean', sur_model_name]:
                                print('using gridsearch para')
                                preds, tests, estimators = grid_preds, grid_tests, grid_estimators
                                test_acc_describe_ = grid_describe
                            else:
                                print('raw')
                        else:
                            print('raw')
                        best_para = grid_search.best_params_
                    test_acc_reports_dict = df2bp(test_acc_reports)
                    test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                    test_acc_describe_dict = test_acc_describe_.to_dict('records')

                    tmodels = copy.deepcopy(estimators[0])
                    tmodels.fit(x3[max_features], y2)

                    select_str = feature_select_method + sur_model_name + str(tmodels.get_params()) + fn
                    select_md5 = md5_convert(select_str)[:6]



                    line_chart_data = []
                    line_trace = {
                        'mode': 'lines+markers',
                        'name': sur_model_name,
                        'type': 'scatter',
                        'x': list(range(1, len(ms)+1)),
                        'y': ms
                    }
                    line_chart_data.append(line_trace)

                    parameter, test_acc, best_esti = [], [], []
                    feature_names = []
                    best_esti.append(tmodels)
                    parameter.append(str(sur_model.get_params()))
                    test_acc.append(test_acc_describe.iloc[0, 0])
                    feature_names.append(str(max_features))
                    final_reports = {'Mean C-index': test_acc,
                                     'parameter': parameter,
                                     'feature_names': feature_names, }
                    final_reports = pd.DataFrame(final_reports, index=[sur_model_name]).reset_index().rename(
                        columns={'index': 'Method'})
                    final_reports['md5'], final_reports['fsm'] = select_md5, feature_select_method
                    final_reports['grid_para'] = str(best_para)
                    final_reports['grid_list'] = str(gridsearch_para)
                    # normalization
                    final_reports['fn'] = fn
                    final_reports['scaler'] = str(scaler)

                    final_reports_dict = final_reports.to_dict('records')

                    data_median = tmodels.predict(pd.DataFrame(x3[max_features].median()).T)[0]
                    surv_trace, resultp = mk_surv_data(sur_model_name, x3[max_features], y2, best_esti[0], data_median)
                    surv_layout = mk_surv_layout(sur_model_name, resultp)
                    surv_data = {'surv_trace': surv_trace, 'surv_layout': surv_layout}

                    # validation
                    vsurv_data,vlinedata = {}, []
                    if len(test_set)>0:
                        vsurv_trace, vresultp = mk_surv_data(sur_model_name, validation_data[max_features], validation_label,
                                                             best_esti[0], data_median)
                        vsurv_layout = mk_surv_layout(sur_model_name, vresultp)
                        vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

                        vlinedata = []
                        va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(best_esti[0], validation_data,
                                                                                 validation_label, y2,
                                                                                 max_features, sur_model_name)

                        vlinetrace = mk_auc_line(sur_model_name, va_times, rsf_auc, mean_auc, cindex)
                        vlinedata.append(vlinetrace)
                    # pickle
                    surv_pickle = {
                        'sur_model_name': sur_model_name,
                        'line_chart_data': line_chart_data,
                        'test_acc_reports_dict': test_acc_reports_dict,
                        'test_acc_describe_dict': test_acc_describe_dict,
                        'final_reports_dict': final_reports_dict,
                        'surv_data': surv_data,
                        'vsurv_data': vsurv_data,
                        'vlinedata': vlinedata
                    }
                    para_str = feature_select_method + final_reports['Method'][0] + str(final_reports['parameter'][0])+\
                               str(gridsearch_para) + fn
                    para_md5 = md5_convert(para_str)[:6]
                    final_reports = pd.concat([cp_cache['reports'], final_reports], axis=0).drop_duplicates(keep='last')
                    final_reports_dict = final_reports.to_dict('records')
                    cp_cache[para_md5] = surv_pickle
                    cp_cache['reports'] = final_reports

                    with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                              'wb') as f:
                        pickle.dump(cp_cache, f)
                    model_info = {}
                    model_info['name'], model_info['model'], model_info['feature_names'] = \
                        sur_model_name, tmodels, max_features
                    model_info['scaler'] = scaler
                    with open(STATIC_ROOT + '/cache/' + projectid + '/' + para_md5 + '.pkl',
                              'wb') as f:
                        pickle.dump(model_info, f)

                else:
                    final_reports_dict = cp_cache['reports'].to_dict('records')
                    sur_model_name = cp_cache[select_md5]['sur_model_name']
                    line_chart_data = cp_cache[select_md5]['line_chart_data']
                    test_acc_reports_dict = cp_cache[select_md5]['test_acc_reports_dict']
                    test_acc_describe_dict = cp_cache[select_md5]['test_acc_describe_dict']
                    surv_data = cp_cache[select_md5]['surv_data']
                    vsurv_data = cp_cache[select_md5]['vsurv_data']
                    vlinedata = cp_cache[select_md5]['vlinedata']


        #send email
        # to_mail = request.POST.get('to_mail')
        to_mail =client_msg['to_mail']
        if 'para_md5' in locals():  # 判断是否使用缓存，已有数据的变量名是select_md5
            url = 'maler/survival_cp_result/prev/'+ projectid + '_' + para_md5
            if to_mail != '':
                if re.match('^.*?@.*', to_mail):
                    task_sendmail(to_mail, url)
        analysis_results = {
            'projectid': projectid,
            'sur_model_name': sur_model_name,
            'line_chart_data': line_chart_data,
            'test_acc_reports_dict': test_acc_reports_dict,
            'test_acc_describe_dict': test_acc_describe_dict,
            'final_reports_dict': final_reports_dict,
            'surv_data': surv_data,
            'vsurv_data': vsurv_data,
            'vlinedata': vlinedata,
            'ifval': ifval,
        }
        return analysis_results
    # return render(request, 'survival_cp_result.html', {
    #     'projectid': projectid,
    #     'sur_model_name': sur_model_name,
    #     'line_chart_data': json.dumps(line_chart_data),
    #     'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
    #     'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
    #     'final_reports_dict': json.dumps(final_reports_dict),
    #     'surv_data': json.dumps(surv_data),
    #     'vsurv_data': json.dumps(vsurv_data),
    #     'vlinedata': json.dumps(vlinedata),
    # })
    # except Exception as e:
    #     print(repr(e))
    #     print('线程池任务报错！！！')
    #     analysis_results = {
    #         "error": repr(e)
    #     }
    #     return analysis_results


def show_prev_page(request, projectid_paramd5):
    if len(projectid_paramd5.split('-')[2]) != 4:
        if len(projectid_paramd5.split('_')) == 2:
            projectid = projectid_paramd5.split('_')[0]
            paramd5 = projectid_paramd5.split('_')[1]
            if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl'):
                status = 'Running'
                return render(request, 'status.html', {
                    'status': status,
                    'projectid': projectid,
                })
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
        else:
            projectid = projectid_paramd5
            if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl'):
                status = 'Running'
                # return render(request, 'status.html', {
                #     'status': status,
                #     'projectid': projectid,
                # })
                with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'rb') as f:
                    preview_pickle = pickle.load(f)
                feature_select_method = preview_pickle['feature_select_method']
                form_action = preview_pickle['form_action']
                display_samples_dict = preview_pickle['display_samples_dict']
                hist_trace = preview_pickle['hist_trace']
                inputdata_display = preview_pickle['inputdata_display']
                inputdata_columns = preview_pickle['inputdata_columns']
                title_str = preview_pickle['title_str']
                return render(request, 'status.html', {
                    'projectid': projectid,
                    'form_action': form_action,
                    'status': status,
                    'feature_select_method': feature_select_method,
                    # 'model_md5': model_md5,
                    'display_samples_dict': json.dumps(display_samples_dict),
                    'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
                    'inputdata_display': json.dumps(inputdata_display),
                    'inputdata_columns': json.dumps(inputdata_columns),
                    'title_str': title_str,
                })
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                    cp_cache = pickle.load(f)

                with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
                    model_pickle = pickle.load(f)
                status = 'Running'
                if len(model_pickle.keys()) > cp_cache['reports'].shape[0]:
                    with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'rb') as f:
                        preview_pickle = pickle.load(f)
                    feature_select_method = preview_pickle['feature_select_method']
                    form_action = preview_pickle['form_action']
                    display_samples_dict = preview_pickle['display_samples_dict']
                    hist_trace = preview_pickle['hist_trace']
                    inputdata_display = preview_pickle['inputdata_display']
                    inputdata_columns = preview_pickle['inputdata_columns']
                    title_str = preview_pickle['title_str']
                    return render(request, 'status.html', {
                        'projectid': projectid,
                        'form_action': form_action,
                        'status': status,
                        'feature_select_method': feature_select_method,
                        # 'model_md5': model_md5,
                        'display_samples_dict': json.dumps(display_samples_dict),
                        'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
                        'inputdata_display': json.dumps(inputdata_display),
                        'inputdata_columns': json.dumps(inputdata_columns),
                        'title_str': title_str,
                    })
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
            paramd5 = cp_cache['reports']['md5'][0]
        # projectid = projectid_paramd5.split('_')[0]
        # paramd5 = projectid_paramd5.split('_')[1]
        # # load pickle
        # with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
        #     cp_cache = pickle.load(f)

        final_reports_dict = cp_cache['reports'].to_dict('records')
        sur_model_name = cp_cache[paramd5]['sur_model_name']
        line_chart_data = cp_cache[paramd5]['line_chart_data']
        test_acc_reports_dict = cp_cache[paramd5]['test_acc_reports_dict']
        test_acc_describe_dict = cp_cache[paramd5]['test_acc_describe_dict']
        surv_data = cp_cache[paramd5]['surv_data']
        vsurv_data = cp_cache[paramd5]['vsurv_data']
        vlinedata = cp_cache[paramd5]['vlinedata']

        return render(request, 'survival_cp_result.html', {
            'projectid': projectid,
            'final_reports_dict': json.dumps(final_reports_dict),
            'sur_model_name': sur_model_name,
            'line_chart_data': json.dumps(line_chart_data),
            'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
            'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
            'surv_data': json.dumps(surv_data),
            'vsurv_data': json.dumps(vsurv_data),
            'vlinedata': json.dumps(vlinedata),
            'change_page': True,
        })
    else:
        projectid = projectid_paramd5
        if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl'):
            status = 'Running'
            return render(request, 'status.html', {
                'status': status,
                'projectid': projectid,
            })
        else:
            with open(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl', 'rb') as f:
                surv_pickle = pickle.load(f)

            line_chart_data = surv_pickle['line_chart_data']
            test_acc_reports_dict = surv_pickle['test_acc_reports_dict']
            test_acc_describe_dict = surv_pickle['test_acc_describe_dict']
            max_reports_dict = surv_pickle['max_reports_dict']
            surv_dict = surv_pickle['surv_dict']
            subplot_sur = surv_pickle['subplot_sur']
            para_dict = surv_pickle['para_dict']
            vsubplot_sur = surv_pickle['vsubplot_sur']
            vpara_dict = surv_pickle['vpara_dict']
            vlinedata = surv_pickle['vlinedata']
            return render(request, 'survival_oc_result.html', {
                'projectid': projectid,
                'line_chart_data': json.dumps(line_chart_data),
                'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
                'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
                'max_reports_dict': json.dumps(max_reports_dict),
                'surv_dict': json.dumps(surv_dict),
                'subplot_sur': json.dumps(subplot_sur),
                'para_dict': json.dumps(para_dict),
                'vsubplot_sur': json.dumps(vsubplot_sur),
                'vpara_dict': json.dumps(vpara_dict),
                'vlinedata': json.dumps(vlinedata),
            })



'''
CUSTOMIZED MODELS
'''
def Survival_svm(Alpha=1,Degree=3,Coef0=1,Gamma=None,Random_state=10,Max_iter=20,Optimizer="rbtree",Kernel='rbf'):
    sur_svm = FastKernelSurvivalSVM(alpha=Alpha,degree=Degree,coef0=Coef0,gamma=Gamma,random_state=Random_state,
                                    max_iter=Max_iter,optimizer=Optimizer,kernel=Kernel)
    return sur_svm

def Survival_tree(Splitter='best',Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features=None):
    sur_tree = SurvivalTree(splitter=Splitter,max_depth=Max_depth,min_samples_split=Min_samples_split,
                            min_samples_leaf=Min_samples_leaf,max_features=Max_features,random_state=10)
    return sur_tree

def Survival_extratrees(N_estimators=100,Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features='sqrt'):
    sur_extratrees = ExtraSurvivalTrees(n_estimators=N_estimators,max_depth=Max_depth,min_samples_split=Min_samples_split,
                                       min_samples_leaf=Min_samples_leaf,max_features=Max_features,
                                       n_jobs=1,random_state=10)
    return sur_extratrees

def Survival_randomforest(N_estimators=100,Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features=None):
    sur_randomforest = RandomSurvivalForest(n_estimators=N_estimators,max_depth=Max_depth,min_samples_split=Min_samples_split,
                                       min_samples_leaf=Min_samples_leaf,max_features=Max_features,
                                       n_jobs=1,random_state=10)
    return sur_randomforest

def Survival_gradientboosting(Loss='coxph',Learning_rate=0.1,N_estimators=100,Min_samples_split=2,
                              Min_samples_leaf=1,Max_depth=3,Max_features=None,Subsample=1.0):
    sur_gb = GradientBoostingSurvivalAnalysis(loss=Loss,learning_rate=Learning_rate,n_estimators=N_estimators,
                                              min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,
                                             max_depth=Max_depth,max_features=Max_features,subsample=Subsample)
    return sur_gb


'''
METHODS
'''

def select_sur_model(request):
    select_child_model = request.POST.get('select_child_model').replace('task_', '')
    # gridsearch para
    gridsearch_para = {}
    grid = request.POST.get('usr_grid')
    if select_child_model == 'survivalsvm':
        select_model_name = 'SurvivalSVM'
        kernel, optimizer = request.POST.get('survivalsvm_kernel'), \
                            request.POST.get('survivalsvm_optimizer')
        if grid == 'G':
            select_model = Survival_svm(Kernel=kernel, Optimizer=optimizer)
            alpha_grid, degree_grid, gamma_grid, coef0_grid = \
                                                request.POST.get('survivalsvm_alpha_grid'), \
                                                request.POST.get('survivalsvm_degree_grid'), \
                                                request.POST.get('survivalsvm_gamma_grid'), \
                                                request.POST.get('survivalsvm_coef0_grid')
            gs_para = {'estimator__alpha': alpha_grid,
                       'estimator__degree': degree_grid,
                       'estimator__gamma': gamma_grid,
                       'estimator__coef0': coef0_grid
                       }

            gridsearch_para = grid_para_split(gs_para, request)
        else:
            kernel, optimizer, alpha, degree, gamma, coef0 = request.POST.get('survivalsvm_kernel'), \
                                                            request.POST.get('survivalsvm_optimizer'), \
                                                            float(request.POST.get('survivalsvm_alpha')), \
                                                            request.POST.get('survivalsvm_degree'), \
                                                            request.POST.get('survivalsvm_gamma'), \
                                                            request.POST.get('survivalsvm_coef0')
            if kernel == 'linear':
                select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Optimizer=optimizer)
            elif kernel == 'ploy':
                if gamma == '': gamma = None
                degree = int(degree)
                coef0 = float(coef0)
                select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Degree=degree, Gamma=gamma, Coef0=coef0)
            elif kernel == 'rbf':
                if gamma == '': gamma = None
                select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Gamma=gamma)
            elif kernel == 'sigmoid':
                coef0 = float(coef0)
                select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Coef0=coef0)
            else:
                select_model = Survival_svm(Kernel=kernel, Alpha=alpha)
    elif select_child_model == 'survivaltree':
        select_model_name = 'SurvivalTree'
        splitter,max_features = request.POST.get('survivaltree_splitter'),request.POST.get('survivaltree_max_features'),
        if grid == 'G':
            select_model = Survival_tree(Splitter=splitter, Max_features=max_features)
            max_depth_grid, min_samples_split_grid, min_samples_leaf_grid = \
                request.POST.get('survivaltree_max_depth_grid'), \
                request.POST.get('survivaltree_min_samples_split_grid'), \
                request.POST.get('survivaltree_min_samples_leaf_grid')
            gs_para = {'estimator__max_depth': max_depth_grid,
                       'estimator__min_sample_split_grid': min_samples_split_grid,
                       'estimator__min_sample_leaf_grid': min_samples_leaf_grid
                       }
            gridsearch_para = grid_para_split(gs_para, request)
        else:
            splitter, max_depth, min_samples_split, min_samples_leaf, max_features = \
                request.POST.get('survivaltree_splitter'),request.POST.get('survivaltree_max_depth'), \
                request.POST.get('survivaltree_min_samples_split'), request.POST.get('survivaltree_min_samples_leaf'), \
                request.POST.get('survivaltree_max_features')
            max_depth, min_samples_split, min_samples_leaf, max_features = \
                surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
            select_model = Survival_tree(Splitter=splitter, Max_depth=max_depth,
                                         Min_samples_split=min_samples_split, Min_samples_leaf=min_samples_leaf,
                                         Max_features=max_features)
    elif select_child_model == 'extrasurvivaltrees':
        select_model_name = 'ExtraSurvivalTrees'
        max_features = request.POST.get('extrasurvivaltrees_max_features')
        if grid == 'G':
            select_model = Survival_extratrees(Max_features=max_features)
            max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, n_estimators_grid = \
                request.POST.get('extrasurvivaltrees_max_depth_grid'), request.POST.get(
                    'extrasurvivaltrees_min_samples_split_grid'), \
                request.POST.get('extrasurvivaltrees_min_samples_leaf_grid'),\
                request.POST.get('extrasurvivaltrees_n_estimators_grid')
            gs_para = {'estimator__max_depth': max_depth_grid,
                       'estimator__min_samples_split': min_samples_split_grid,
                       'estimator__min_samples_leaf': min_samples_leaf_grid,
                       'estimator__n_estimators': n_estimators_grid
                       }
            gridsearch_para = grid_para_split(gs_para, request, ['int']*4)
        else:
            max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators = \
                request.POST.get('extrasurvivaltrees_max_depth'), request.POST.get('extrasurvivaltrees_min_samples_split'), \
                request.POST.get('extrasurvivaltrees_min_samples_leaf'), request.POST.get('extrasurvivaltrees_max_features'), \
                int(request.POST.get('extrasurvivaltrees_n_estimators'))
            max_depth, min_samples_split, min_samples_leaf, max_features = \
                surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
            select_model = Survival_extratrees(Max_depth=max_depth, Min_samples_split=min_samples_split,
                                               Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                               N_estimators=n_estimators)
    elif select_child_model == 'randomsurvivalforest':
        select_model_name = 'RandomSurvivalForest'
        max_features = request.POST.get('randomsurvivalforest_max_features')
        if grid == 'G':
            select_model = Survival_randomforest(Max_features=max_features)
            max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, n_estimators_grid = \
                request.POST.get('randomsurvivalforest_max_depth_grid'), \
                request.POST.get('randomsurvivalforest_min_samples_split_grid'), \
                request.POST.get('randomsurvivalforest_min_samples_leaf_grid'), \
                request.POST.get('randomsurvivalforest_n_estimators_grid')
            gs_para = {'estimator__max_depth': max_depth_grid,
                       'estimator__min_samples_split': min_samples_split_grid,
                       'estimator__min_samples_leaf': min_samples_leaf_grid,
                       'estimator__n_estimators': n_estimators_grid
                       }
            gridsearch_para = grid_para_split(gs_para, request , ['int']*4)
        else:
            max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators = \
                request.POST.get('randomsurvivalforest_max_depth'), np.float(request.POST.get('randomsurvivalforest_min_samples_split')), \
                np.float(request.POST.get('randomsurvivalforest_min_samples_leaf')), request.POST.get('randomsurvivalforest_max_features'), \
                int(request.POST.get('randomsurvivalforest_n_estimators'))
            max_depth, min_samples_split, min_samples_leaf, max_features = \
                surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
            select_model = Survival_randomforest(Max_depth=max_depth, Min_samples_split=min_samples_split,
                                                 Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                                 N_estimators=n_estimators)
    else:
        select_model_name = 'GradientBoostingSurvival'
        loss, max_features, learning_rate,subsample = \
            request.POST.get('gradientboostingsurvival_loss'), \
            request.POST.get('gradientboostingsurvival_max_features'), \
            np.float(request.POST.get('gradientboostingsurvival_learning_rate')), \
            np.float(request.POST.get('gradientboostingsurvival_subsample'))
        if grid == "G":
            select_model = Survival_gradientboosting(Loss='coxph',  Max_features=max_features,
                                                     Learning_rate=learning_rate,Subsample=subsample)
            max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, n_estimators_grid= \
                request.POST.get('gradientboostingsurvival_max_depth_grid'), \
                request.POST.get('gradientboostingsurvival_min_samples_split_grid'),\
                request.POST.get('gradientboostingsurvival_min_samples_leaf_grid'), \
                request.POST.get('gradientboostingsurvival_n_estimators_grid'),
            gs_para = {'estimator__max_depth': max_depth_grid,
                       'estimator__min_sample_split_grid': min_samples_split_grid,
                       'estimator__min_sample_leaf_grid': min_samples_leaf_grid,
                       'estimator__n_estimators': n_estimators_grid
                       }
            gridsearch_para = grid_para_split(gs_para, request,['int']*4)
        else:
            loss, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators, learning_rate ,subsample= \
                request.POST.get('gradientboostingsurvival_loss'), \
                request.POST.get('gradientboostingsurvival_max_depth'), \
                request.POST.get('gradientboostingsurvival_min_samples_split'), \
                request.POST.get('gradientboostingsurvival_min_samples_leaf'), \
                request.POST.get('gradientboostingsurvival_max_features'), \
                int(request.POST.get('gradientboostingsurvival_n_estimators')), \
                np.float(request.POST.get('gradientboostingsurvival_learning_rate')), \
                np.float(request.POST.get('gradientboostingsurvival_subsample'))
            max_depth, min_samples_split, min_samples_leaf, max_features = \
                surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
            select_model = Survival_gradientboosting(Loss='coxph', Max_depth=max_depth, Min_samples_split=min_samples_split,
                                                     Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                                     N_estimators=n_estimators, Learning_rate=learning_rate,Subsample=subsample)
    return select_model, select_model_name,gridsearch_para


def gridsearch_bulid(svc, gridsearch_para, clf_name,sur_times):
    from sksurv.metrics import as_concordance_index_ipcw_scorer
    from sklearn.model_selection import GridSearchCV
    grid_cv = KFold(n_splits=5, shuffle=True, random_state=10)
    grid_model = copy.deepcopy(svc)
    param_grid = grid_space(svc, gridsearch_para, clf_name)
    grid_search = GridSearchCV(as_concordance_index_ipcw_scorer(grid_model,tau=sur_times[-1]),
                               param_grid, cv=grid_cv, n_jobs=1)
    return grid_search

def grid_space(model, gridsearch_para, clf_name):
    if clf_name == 'SurvivalSVM':
        search_space = {
            'estimator__alpha': gridsearch_para['estimator__alpha'],
        }
        if model.get_params()['kernel'] == 'rbf':
            search_space['estimator__gamma'] = list(gridsearch_para['estimator__gamma']) + ['scale', 'auto']
        elif model.get_params()['kernel'] == 'poly':
            search_space['estimator__gamma'] = list(gridsearch_para['estimator__gamma']) + ['scale', 'auto']
            search_space['estimator__coef0'] = gridsearch_para['estimator__coef0']
            search_space['estimator__degree'] = gridsearch_para['estimator__degree']
        elif model.get_params()['kernel'] == 'sigmoid':
            search_space['estimator__gamma'] = list(gridsearch_para['estimator__gamma']) + ['scale', 'auto']
            search_space['estimator__coef0'] = gridsearch_para['estimator__coef0']

    if clf_name == 'SurvivalTree':
        search_space = gridsearch_para
    if clf_name == 'ExtraSurvivalTrees':
        search_space = gridsearch_para
    if clf_name == 'RandomSurvivalForest':
        # 全部转为整数
        gridsearch_para_convert = convert_float_to_int(gridsearch_para)
        search_space = gridsearch_para_convert
    if clf_name == 'GradientBoostingSurvival':
        search_space = gridsearch_para
    return search_space

def convert_float_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_float_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_float_to_int(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.astype(int)
    elif isinstance(obj, float):
        return int(obj)
    else:
        return obj
