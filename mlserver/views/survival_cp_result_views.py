from django.shortcuts import render

import os, shutil, copy, pickle, json
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score,cross_validate ,train_test_split, GridSearchCV, KFold,StratifiedKFold,RepeatedKFold
from sksurv.datasets import get_x_y
from sksurv.svm import FastKernelSurvivalSVM,FastSurvivalSVM
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest,ExtraSurvivalTrees,GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.statistics import logrank_test

from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5, df2bp
from mlserver.views.survival_oc_result_views import sur_data_process, cox_selection, \
    sur_RSKFold, FSS_fun, train_estimator, mk_surv_data,mk_surv_layout, time_dependent_auc,mk_auc_line
import warnings
warnings.filterwarnings("ignore")


def survival_cp_result(request):
    feature_select_method = request.POST.get('feature_select_method')
    print('feature_select_method: ', feature_select_method)
    select_model = request.POST.get('select_model')
    select_child_model = request.POST.get('select_child_model').replace('task_','')
    # select_child_model = 'survivalsvm'
    sur_model, sur_model_name = select_sur_model(select_child_model)
    '''
    IMPORRT DATA
    '''
    # file load
    upload_file = request.FILES.get('upload_profile')
    f = open(os.path.join(STATIC_ROOT, 'cache', upload_file.name), 'wb')
    for line in upload_file.chunks():
        f.write(line)
    f.close()

    upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', upload_file.name))
    projectid = 'SC-' + upload_file_md5[:6] + '-' + feature_select_method
    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
        newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        shutil.move(STATIC_ROOT + '/cache/' + upload_file.name, newpath)
        '''
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'load_breast_cancer.csv', header=0, index_col=0).T
        '''
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + upload_file.name, header=0, index_col=0).T
        x, y = sur_data_process(data)
        x2, vaildation_data, y2, vaildation_label = train_test_split(x, y, random_state=10, train_size=0.7,
                                                                     stratify=y['Status'])
        cv = KFold(n_splits=5, shuffle=True, random_state=10)
        features = cox_selection(x2.values, y2, x2.columns)
        # features,ss2 = cox_selection(x2,y2)
        x3 = x2[features]
        train_index, test_index = sur_RSKFold(x3, y2)

        sf, ms = FSS_fun(features, sur_model, x3, y2, cv, n_jobs=6)
        max_index = np.array(ms).argmax()
        max_score = max(ms)
        max_features = (sf[:max_index + 1])
        preds, tests, res = [], [], []
        for i in range(len(train_index)):
            xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
            xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
            xtrain, xtest = xtrain[max_features], xtest[max_features]
            estimator, test_acc, predict = train_estimator(sur_model, xtrain, ytrain, xtest, ytest)
            tests.append(test_acc), res.append(estimator), preds.append(predict)

        line_chart_data = []
        line_trace = {
            'mode': 'lines+markers',
            'name': sur_model_name,
            'type': 'scatter',
            'x': list(range(1, 50)),
            'y': ms
        }
        line_chart_data.append(line_trace)

        test_acc_reports = pd.DataFrame(data=tests)
        test_acc_reports.columns = [sur_model_name]
        test_acc_reports_dict = df2bp(test_acc_reports)
        test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                     3)
        test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        test_acc_describe_dict = test_acc_describe_.to_dict('records')

        tmodels = copy.deepcopy(sur_model)
        tmodels.fit(x3[max_features], y2)
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
        vsurv_trace, vresultp = mk_surv_data(sur_model_name, vaildation_data[max_features],vaildation_label,best_esti[0],data_median)
        vsurv_layout = mk_surv_layout(sur_model_name, vresultp)
        vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

        vlinedata = []
        va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(best_esti[0], vaildation_data,
                                                                 vaildation_label, y2,
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

        with open(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl',
                  'wb') as f:
            pickle.dump(surv_pickle, f)
    else:
        with open(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl', 'rb') as f:
            surv_pickle = pickle.load(f)

        sur_model_name = surv_pickle['sur_model_name']
        line_chart_data = surv_pickle['line_chart_data']
        test_acc_reports_dict = surv_pickle['test_acc_reports_dict']
        test_acc_describe_dict = surv_pickle['test_acc_describe_dict']
        final_reports_dict = surv_pickle['final_reports_dict']
        surv_data = surv_pickle['surv_data']
        vsurv_data = surv_pickle['vsurv_data']
        vlinedata = surv_pickle['vlinedata']

        print(surv_data)
    return render(request, 'survival_cp_result.html', {
        'projectid': projectid,
        'sur_model_name': sur_model_name,
        'line_chart_data': json.dumps(line_chart_data),
        'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
        'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
        'final_reports_dict': json.dumps(final_reports_dict),
        'surv_data': json.dumps(surv_data),
        'vsurv_data':json.dumps(vsurv_data),
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
                                       n_jobs=4,random_state=10)
    return sur_extratrees

def Survival_randomforest(N_estimators=100,Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features=None):
    sur_randomforest = RandomSurvivalForest(n_estimators=N_estimators,max_depth=Max_depth,min_samples_split=Min_samples_split,
                                       min_samples_leaf=Min_samples_leaf,max_features=Max_features,
                                       n_jobs=4,random_state=10)
    return sur_randomforest

def Survival_gradientboosting(Loss='coxph',Learning_rate=0.1,N_estimators=100,Min_samples_split=2,
                              Min_samples_leaf=1,Max_depth=3,Max_features=None):
    sur_gb = GradientBoostingSurvivalAnalysis(loss=Loss,learning_rate=Learning_rate,n_estimators=N_estimators,
                                              min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,
                                             max_depth=Max_depth,max_features=Max_features)
    return sur_gb


'''
METHODS
'''
def select_sur_model(select_child_model):
    if select_child_model == 'survivalsvm':
        select_model = Survival_svm()
        select_model_name = 'SurvivalSVM'
    elif select_child_model == 'survivaltree':
        select_model = Survival_tree()
        select_model_name = 'SurvivalTree'
    elif select_child_model == 'extrasurvivaltrees':
        select_model = Survival_extratrees()
        select_model_name = 'ExtraSurvivalTrees'
    elif select_child_model == 'randomsurvivalforest':
        select_model = Survival_randomforest()
        select_model_name = 'RandomSurvivalForest'
    else:
        select_model = Survival_gradientboosting()
        select_model_name = 'GradientBoostingSurvival'
    return select_model, select_model_name
