
import math
import os
# from djangoProject import settings
from ML_WebServer import settings
from django.core.mail import send_mail
import re

# if os.path.exists(temp_cache_directory + 'result.zip'):
#     print('Task run complete!' + temp_cache_directory)
#     if re.match('^.*?@.*', to_mail):
#         subject = 'Task run complete!'
#         content = '<a style="font-size: 28px; font-weight: 700; text-align: center;" href="' + endpoint + '/timecourse/analysis/omics_data_analysis_page?analysisId=' + analysis_id + '">--》Click to see the results《--</a>'
#         from_mail = 'Task run notification<' + settings.EMAIL_HOST_USER + '>'
#         print('发送邮件' + content)
#         try:
#             send_mail(subject, message=None, from_email=from_mail, recipient_list=[to_mail], fail_silently=False,
#                       html_message=content)
#             print('Sending an email succeeded！')
#         except Exception as ee:
#             print('发送邮件异常')
#             print(ee)


# This legacy module is not a public route. Never send a message at import time.
# The supported validated result view sends an optional, configured completion
# notice only after a successful analysis.

# try:
#
# except Exception as ee:
#     print('发送邮件异常')


# # bayeSearch
            # import skopt
            # from skopt.space import Real, Integer, Categorical
            # from skopt import BayesSearchCV
            # def randomforest_bayeSCV(model, cv):
            #     max_dep = list(range(3, 20))
            #     max_dep.append(None)
            #     randomforest_model_para = {
            #         'n_estimators': Integer(10, 200),
            #         #          'max_depth'   : Integer(3, 50),
            #         'max_depth': max_dep,
            #         'min_samples_split': Integer(2, 50),
            #         'min_samples_leaf': Integer(1, 50),
            #         # 'max_features': ['auto', 'sqrt', 'log2'],
            #     }
            #     opt = BayesSearchCV(
            #         estimator=model,
            #         search_spaces=randomforest_model_para,
            #         n_iter=200,
            #         cv=cv,
            #         n_jobs=4,
            #         # n_points=2,
            #         random_state=10,
            #         # verbose=2
            #     )
            #     return opt
            #
            # rf = copy.deepcopy(svc)
            # rf_opt = randomforest_bayeSCV(rf, cv2)
            # rf_opt.fit(data3[max_features], label3)
            #
            # print("val. score: %s" % rf_opt.best_score_)
            # print("test score: %s" % rf_opt.score(validation_data.loc[:, max_features], validation_label))
            # print("best params: %s" % str(rf_opt.best_params_))
            # print('raw params: %s' % str(svc.get_params()))
            #
            # validate_predict2, validate_report2 = pre_valid(rf_opt.best_estimator_, validation_data, validation_label,
            #                                               max_features)
            #
            # # GridSearchCV
            # from sklearn.model_selection import GridSearchCV
            # max_dep = list(range(3, 20))
            # max_dep.append(None)
            # param_grid = {
            #     'n_estimators': range(50, 500, 10),
            #     'max_depth': max_dep,
            #     #          'min_samples_split': range(2, 50),
            #     #          'min_samples_leaf':  range(1, 50),
            #     #          'max_features' :     ['auto','sqrt','log2'],
            # }
            # # 网格搜索
            # start = time.perf_counter()
            # grid_search = GridSearchCV(rf, param_grid, cv=cv2, scoring='accuracy', verbose=2, n_jobs=4)
            # grid_search.fit(data3[max_features], label3)
            # print("网格搜索最优参数：", grid_search.best_params_)
            # print("网格搜索最优得分：", grid_search.best_score_)
            # end = time.perf_counter()
            # print('analusis time: ',round(end - start, 2))
            # grid_search.best_estimator_.score(validation_data.loc[:, max_features], validation_label)
            #
            # #raw
            # raw = cross_val_score(estimator=rf, X=data3[max_features], y=label3, cv=cv2)
            # cross_val_score(estimator=svc,X=data3[max_features], y=label3,cv=cv2,error_score='raise',n_jobs=4).mean()
            #
            # np.mean(raw)
            # rf.fit(data3[max_features],label3)
            # rf.score(validation_data.loc[:, max_features], validation_label)
            #
            # ###
            # feature_names2 = list(features)
            # selected_feature = []
            # max_scores = []
            # features_num = min([len(features), 20])  # 判断特征数目是否大于20
            # for i in range(features_num):
            #     cv_scores = []
            #     for feature in feature_names2:
            #         # train_feature = [feature] + selected_feature
            #         train_feature = selected_feature + [feature]
            #         data1 = pd.DataFrame(data.loc[:, train_feature])
            #         cv_score = cross_val_score(svc, data1, label, cv=cv2, n_jobs=4, error_score='raise').mean()
            #         cv_scores.append(cv_score)
            #     max_index = np.array(cv_scores).argmax()
            #     max_score = max(cv_scores)
            #     # if max(cv_scores) > 0.974:
            #     #     break
            #     max_scores.append(max_score)
            #     selected_feature.append(feature_names2[max_index])
            #     print(max_score,' ',feature_names2[max_index])
            #     feature_names2.remove(feature_names2[max_index])
            # ###FSS
            # cross_val_score(svc, data3[max_features], label3, cv=cv2, n_jobs=4, error_score='raise').mean()
from django.shortcuts import render
from django.http import HttpResponse
import os, hashlib, shutil, pickle, time ,json, random, string ,copy
import pandas as pd
import numpy as np

# from sklearnex import patch_sklearn, unpatch_sklearn
# patch_sklearn()
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB,BernoulliNB,ComplementNB,MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_curve,auc,classification_report,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_view_webscoket import df2bp, mkroc, mkradar, JsonEncoder ,task_sendmail
from mlserver.views.featureselection_method import mrmr_fs,FSS_fun,BSS_fun,train_estimator,train_top3,selectkbest_top20,pre_screening
import warnings
from dwebsocket.decorators import accept_websocket
warnings.filterwarnings("ignore")
def data_analysis(WebSocket,client_msg,projectid):
    # client_msg = json.loads(WebSocket.wait())
    # client_msg = str(client_msg, encoding="utf-8")
    # print('data_analysis: ', client_msg)
    # print(projectid)
    # client_msg['status'] = 1
    # time.sleep(5)
    # WebSocket.send(json.dumps(client_msg))
    print('start analysis')
    analysis_results = cp_analysis(client_msg,projectid)
    WebSocket.send(json.dumps(analysis_results))
    print('finish!!')

from dwebsocket.decorators import accept_websocket
from concurrent.futures.thread import ThreadPoolExecutor
pools = ThreadPoolExecutor(100)

@accept_websocket
def result_ws(request, projectid):
    if request.is_websocket():
        WebSocket = request.websocket
        while True:
            if WebSocket.has_messages():
                client_msg = json.loads(WebSocket.wait())
                if client_msg != 'heartbeat':
                    print(client_msg)
                    task1 = pools.submit(data_analysis,WebSocket,client_msg,projectid)
                elif client_msg == 'heartbeat':
                    messages = {
                        'time': time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                        'status': 0,
                    }
                    request.websocket.send(json.dumps(messages))


