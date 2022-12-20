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
from mlserver.views.classification_oc_result_views import df2bp, mkroc, mkradar
import warnings
warnings.filterwarnings("ignore")

title = ["Naive Bayes","SVM","RandomForest","Logistic","KNN","XGBoost","lightGBM",'Adaboost',"DecisionTree","GBDT"]

def result(request):
    select_model = request.POST.get('select_model')
    if select_model == 'model_bclass':
        prefix_id = 'BCC-'
        ifmarco = False
    else:
        prefix_id = 'MCC-'
        ifmarco = True
    # Feature selection methods
    feature_select_method = request.POST.get('feature_select_method')
    print('feature_select_method: ', feature_select_method)

    # select child model
    select_child_model = request.POST.get('select_child_model').replace("task_", "")
    print(select_child_model)
    file_upload_type = request.POST.get('file_upload_type')

    '''
    MODULE PARAMETERS
    '''
    # if select_child_model == 'svm':
    #     svc = get_svc_model(request)

    svc, clf_name = select_class_model(request)
    # get project id
    projectid = request.POST.get('projectid')
    if projectid == '': projectid='None'

    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
        '''
        IMPORT DATA
        '''
        # label load
        # obj_label = request.FILES.get('upload_label')
        # print(obj_label.name)
        # f = open(os.path.join(STATIC_ROOT, 'cache', obj_label.name), 'wb')
        # for line in obj_label.chunks():
        #     f.write(line)
        # f.close()
        # random token
        # token = ''.join(random.sample(string.digits + string.ascii_letters, 6))
        token = request.POST.get('random_token')
        if file_upload_type == 'user_data':
            '''
            IMPORRT DATA
            '''
            obj_file = request.FILES.get('upload_file')
            f = open(os.path.join(STATIC_ROOT, 'cache', obj_file.name), 'wb')
            for line in obj_file.chunks():
                f.write(line)
            f.close()

            filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_file.name))

            projectid = prefix_id + filemd5[:6] + '-' + token
            newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
            os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            shutil.move(STATIC_ROOT + '/cache/' + obj_file.name, newpath)
            # shutil.move(STATIC_ROOT + '/cache/' + obj_label.name, newpath)
            # rename
            os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + obj_file.name, \
                      STATIC_ROOT + '/cache/' + projectid + '/' + "express_data.csv")
            # os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + obj_label.name, \
            #           STATIC_ROOT + '/cache/' + projectid + '/' + "label.csv")
            inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "express_data.csv", header=0,
                                    index_col=0).T
        else:
            if select_model == 'model_bclass':
                filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/binary_classification_example.csv'))
                filename = 'binary_classification_example.csv'
            else:
                filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/multiclass_classification_example.csv'))
                filename = 'multiclass_classification_example.csv'
            projectid = prefix_id + filemd5[:6] + '-' + token
            os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
            shutil.copy(STATIC_ROOT + '/cache/example/' + filename, newpath)
            os.rename(newpath + '/' + filename, \
                      newpath + '/' + "express_data.csv")
            inputdata = pd.read_csv(STATIC_ROOT + '/cache/example/' + filename, header=0, index_col=0).T
        # profile load
        # obj_file = request.FILES.get('upload_file')
        # f = open(os.path.join(STATIC_ROOT, 'cache', obj_file.name), 'wb')
        # for line in obj_file.chunks():
        #     f.write(line)
        # f.close()


        '''
        CALCULATE PROJECTID
        '''
        # filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_file.name))



        print(projectid)
        '''
            feature_select_method = 'TopK'
            select_model = 'model_bclass'
            select_child_model = 'svm'
            svm_kernel, svm_c, svm_degree, svm_coef = 'rbf',1,2,0
            projectid = '911c4d-c3ceec-TopK-RWZIlT'
            data = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'express_data.csv', header=0, index_col=0).T
            label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'label.csv', header=0, index_col=0)
        '''

        # read files

        # label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'label_3columns.csv', header=0, index_col=0)
        # data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'express_data.csv', header=0, index_col=0).T
        # label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'label.csv', header=0, index_col=0)

        # label = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "label.csv", header=0, index_col=0)
        # line_chart_data = 'null'
        # preprocess
        # label = np.array(label).ravel()
        # label2, classes = label_pre(label)

        train_set, test_set, blind_set = split_train_test(inputdata)
        data, label = classification_process(train_set)
        label3, classes = label_pre(label)
        if len(test_set) > 0:
            validation_data, validation_label = classification_process(test_set)
            validation_label, ll = label_pre(validation_label)

        features = selectkbest_top20(data, label3, k=50)
        data3 = data.loc[:, features]
        # data3, validation_data, label3, validation_label = train_test_split(data2, label2,
        #                                                                     random_state=10,
        #                                                                     train_size=0.9)
        train_index, test_index = RSKFold(data3, label3)  # 十次五折交叉验证

        # features = selectkbest_top20(data, label2, k=50)
        # data2 = data.loc[:, features]
        # 拆分验证集
        # data3, validation_data, label3, validation_label = train_test_split(data2, label2, random_state=10,
        #                                                                     train_size=0.9)
        # train_index, test_index = RSKFold(data3, label3)  # 十次五折交叉验证
        # clf_name = select_child_model.upper()
        if feature_select_method == 'TopK':
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=10)
            clf_num, ms = pre_screening(data3, label3, svc, features, cv=cv)
            test_accs, estimators, mean_accs, predicts, f_names = train_top3(svc, data3, label3, clf_num,
                                                                             train_index, test_index, features)  ##
            line_chart_data = []
            line_trace = {
                'mode': 'lines+markers',
                'name': clf_name,
                'type': 'scatter',
                'x': list(range(1, len(ms) + 1)),
                'y': ms
            }
            line_chart_data.append(line_trace)

            final_reports, f_describe = customized_report(clf_name, estimators, data3, label3, predicts,
                                                          test_index, f_names, test_accs)

            final_reports_dict = df2bp(final_reports)
            f_describe = np.round(f_describe.loc[("mean", 'min', 'max', 'std'), :],
                                  3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
            f_describe_dict = f_describe.to_dict('records')
            # ROC
            mean_FPR, mean_TPR_df, auc_mean_std = get_ROC_info(clf_name, estimators, data3, label3, test_index, f_names,
                                                               final_reports, predicts)
            roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=[clf_name])

            # 最优分类器表格展示
            parameter, train_acc, test_acc, best_esti = [], [], [], []
            precision, AUC, recall, f1_score = [], [], [], []
            feature_names = []

            maxauc_index = np.array(test_accs).argmax()
            best_esti.append(estimators[maxauc_index])
            parameter.append(str(estimators[maxauc_index].get_params()))

            test_acc.append(final_reports["test_accuracy"][maxauc_index])
            precision.append(final_reports["precision"][maxauc_index])
            recall.append(final_reports["recall"][maxauc_index])
            f1_score.append(final_reports["f1-score"][maxauc_index])
            AUC.append(final_reports["AUC"][maxauc_index])
            feature_names.append(str(list(f_names)))
            max_reports = {'parameter': parameter,
                           'feature_names': feature_names,
                           'test_acc': test_acc,
                           'precision': precision,
                           'AUC': AUC,
                           'recall': recall,
                           'f1-score': f1_score}
            max_reports = pd.DataFrame(max_reports, index=[clf_name])
            max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
            max_reports = max_reports.reset_index().rename(
                columns={'index': 'Method', 'f1-score': 'f1score'})

            '''validation'''
            validate_predict,validate_report = pre_valid(best_esti[0],validation_data,validation_label,f_names)
            # barplot
            bar_dict = mkbar(validate_report)
            # heatmap
            heatmap_dict, heatmap_anno = mkheatmap(validation_label, validate_predict, classes)
            # roc
            valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info(clf_name, best_esti[0], validation_data, validation_label, f_names)
            valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std, title=[clf_name])

            report_describe_roc = {
                'final_reports': final_reports,
                'f_describe': f_describe,
                'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
                # validation
                'bar_dict': bar_dict,
                'heatmap_dict': heatmap_dict,
                'heatmap_anno': heatmap_anno,
                'valid_roc_traces': valid_roc_traces,
                'line_chart_data': line_chart_data
                # 'report': max_reports
            }
        elif feature_select_method == 'FSS' or feature_select_method == 'BSS':
            print("run BSS")
            cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=10)
            start = time.perf_counter()
            if feature_select_method == 'FSS':
                selected_feature, max_scores = FSS_fun(features, svc, data3, label3, cv2)
            else:
                selected_feature, max_scores = BSS_fun(features, svc, data3, label3, cv2)
            # 得到最值
            max_index = max_scores.index(np.nanmax(max_scores))
            max_score = max(max_scores)
            max_features = selected_feature[:max_index + 1]
            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = data3.iloc[train_index[i], :], label3[train_index[i]]
                xtest, ytest = data3.iloc[test_index[i], :], label3[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(svc, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)
            end = time.perf_counter()
            print(round(end - start, 2))
            line_chart_data = []
            trace = {
                'mode': 'lines+markers',
                'name': clf_name,
                'type': 'scatter',
                'x': list(range(1, 21)),
                'y': max_scores
            }
            line_chart_data.append(trace)

            final_reports, f_describe = customized_report(clf_name, res, data3, label3, preds, test_index,
                                                          max_features, tests)
            final_reports_dict = df2bp(final_reports)
            f_describe = np.round(f_describe.loc[("mean", 'min', 'max', 'std'), :],
                                  3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
            f_describe_dict = f_describe.to_dict('records')

            # ROC
            mean_FPR, mean_TPR_df, auc_mean_std = get_ROC_info(clf_name, res, data3, label3, test_index, max_features,
                                                               final_reports, preds)
            roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=[clf_name])

            # 最优分类器表格展示
            parameter, train_acc, test_acc, best_esti = [], [], [], []
            precision, AUC, recall, f1_score = [], [], [], []
            feature_names = []

            maxauc_index = np.array(tests).argmax()
            best_esti.append(res[maxauc_index])
            parameter.append(str(res[maxauc_index].get_params()))

            test_acc.append(final_reports["test_accuracy"][maxauc_index])
            precision.append(final_reports["precision"][maxauc_index])
            recall.append(final_reports["recall"][maxauc_index])
            f1_score.append(final_reports["f1-score"][maxauc_index])
            AUC.append(final_reports["AUC"][maxauc_index])
            feature_names.append(str(list(max_features)))
            max_reports = {'parameter': parameter,
                           'feature_names': feature_names,
                           'test_acc': test_acc,
                           'precision': precision,
                           'AUC': AUC,
                           'recall': recall,
                           'f1-score': f1_score}
            max_reports = pd.DataFrame(max_reports, index=[clf_name])
            max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
            max_reports = max_reports.reset_index().rename(
                columns={'index': 'Method', 'f1-score': 'f1score'})

            '''validation'''
            validate_predict, validate_report = pre_valid(best_esti[0], validation_data, validation_label, max_features)
            # barplot
            bar_dict = mkbar(validate_report)
            # heatmap
            heatmap_dict, heatmap_anno = mkheatmap(validation_label, validate_predict, classes)
            # roc
            valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info(clf_name, best_esti[0],
                                                                                   validation_data, validation_label,
                                                                                   max_features)
            valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std, title=[clf_name])

            report_describe_roc = {
                'final_reports': final_reports,
                'f_describe': f_describe,
                'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
                # validation
                'bar_dict': bar_dict,
                'heatmap_dict': heatmap_dict,
                'heatmap_anno': heatmap_anno,
                'valid_roc_traces': valid_roc_traces,
                'line_chart_data': line_chart_data
                # 'report': max_reports
            }

            # report_describe_roc = {
            #     'final_reports': final_reports,
            #     'f_describe': f_describe,
            #     'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
            #     # 'report': max_reports
            # }

        # make cache
        cp_cache = {}
        para_str = feature_select_method + max_reports['Method'][0] + str(max_reports['parameter'][0])
        para_md5 = md5_convert(para_str)[:6]
        # add parameter md5 and feature select method
        max_reports['md5'],max_reports['fsm'] = para_md5, feature_select_method
        max_reports_dict = max_reports.to_dict('records')

        cp_cache[para_md5] = report_describe_roc
        cp_cache['reports'] = max_reports

        with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                  'wb') as f:
            pickle.dump(cp_cache, f)

    else:
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "express_data.csv", header=0, index_col=0).T
        # label = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "label.csv", header=0, index_col=0)
        # line_chart_data = 'null'
        # preprocess
        # label = np.array(label).ravel()
        # label3, classes = label_pre(label)

        train_set, test_set, blind_set = split_train_test(inputdata)
        data, label = classification_process(train_set)
        label3, classes = label_pre(label)

        if len(test_set) > 0:
            validation_data, validation_label = classification_process(test_set)
            validation_label, ll = label_pre(validation_label)

        features = selectkbest_top20(data, label3, k=50)
        data3 = data.loc[:, features]
        # 拆分验证集
        # data3, validation_data, label3, validation_label = train_test_split(data2, label2, random_state=10,
        #                                                                     train_size=0.9)
        train_index, test_index = RSKFold(data3, label3)  # 十次五折交叉验证
        # clf_name = select_child_model.upper()
        if feature_select_method == 'TopK':
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=10)
            clf_num, ms = pre_screening(data3, label3, svc, features, cv=cv)
            test_accs, estimators, mean_accs, predicts, f_names = train_top3(svc, data3, label3, clf_num,
                                                                             train_index, test_index, features)  ##

            line_chart_data = []
            line_trace = {
                'mode': 'lines+markers',
                'name': clf_name,
                'type': 'scatter',
                'x': list(range(1, len(ms) + 1)),
                'y': ms
            }
            line_chart_data.append(line_trace)

            maxauc_index = np.array(test_accs).argmax()
            select_str = feature_select_method + clf_name + str(estimators[maxauc_index].get_params())
            select_md5 = md5_convert(select_str)[:6]
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)

            if select_md5 not in cp_cache.keys():
                final_reports, f_describe = customized_report(clf_name, estimators, data3, label3, predicts,
                                                              test_index, f_names, test_accs)

                final_reports_dict = df2bp(final_reports)
                f_describe = np.round(f_describe.loc[("mean", 'min', 'max', 'std'), :],
                                      3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                f_describe_dict = f_describe.to_dict('records')

                mean_FPR, mean_TPR_df, auc_mean_std = get_ROC_info(clf_name, estimators, data3, label3, test_index,
                                                                   f_names,
                                                                   final_reports, predicts)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=[clf_name])

                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []

                maxauc_index = np.array(test_accs).argmax()
                best_esti.append(estimators[maxauc_index])
                parameter.append(str(estimators[maxauc_index].get_params()))

                test_acc.append(final_reports["test_accuracy"][maxauc_index])
                precision.append(final_reports["precision"][maxauc_index])
                recall.append(final_reports["recall"][maxauc_index])
                f1_score.append(final_reports["f1-score"][maxauc_index])
                AUC.append(final_reports["AUC"][maxauc_index])
                feature_names.append(str(list(f_names)))

                max_reports = {'parameter': parameter,
                               'feature_names': feature_names,
                               'test_acc': test_acc,
                               'precision': precision,
                               'AUC': AUC,
                               'recall': recall,
                               'f1-score': f1_score}
                max_reports = pd.DataFrame(max_reports, index=[clf_name])
                max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                    max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                max_reports = max_reports.reset_index().rename(
                    columns={'index': 'Method', 'f1-score': 'f1score'})

                para_str = feature_select_method + max_reports['Method'][0] + str(max_reports['parameter'][0])
                para_md5 = md5_convert(para_str)[:6]
                print(para_md5)
                # add parameter md5 and feature select method
                max_reports['md5'], max_reports['fsm'] = para_md5, feature_select_method

                '''validation'''
                validate_predict, validate_report = pre_valid(best_esti[0], validation_data, validation_label, f_names)
                # barplot
                bar_dict = mkbar(validate_report)
                # heatmap
                heatmap_dict, heatmap_anno = mkheatmap(validation_label, validate_predict, classes)
                # roc
                valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info(clf_name, best_esti[0],
                                                                                       validation_data,
                                                                                       validation_label, f_names)
                valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std, title=[clf_name])

                report_describe_roc = {
                    'final_reports': final_reports,
                    'f_describe': f_describe,
                    'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
                    # validation
                    'bar_dict': bar_dict,
                    'heatmap_dict': heatmap_dict,
                    'heatmap_anno': heatmap_anno,
                    'valid_roc_traces': valid_roc_traces,
                    'line_chart_data': line_chart_data
                }

                max_reports = pd.concat([cp_cache['reports'], max_reports], axis=0).drop_duplicates(keep='last')
                max_reports_dict = max_reports.to_dict('records')
                cp_cache[para_md5] = report_describe_roc
                cp_cache['reports'] = max_reports

                with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                          'wb') as f:
                    pickle.dump(cp_cache, f)
            else:
                final_reports_dict = df2bp(cp_cache[select_md5]['final_reports'])
                f_describe_dict = cp_cache[select_md5]['f_describe'].to_dict('records')
                roc_traces = mkroc(
                    cp_cache[select_md5]['roc']['mean_FPR'],
                    cp_cache[select_md5]['roc']['mean_TPR_df'],
                    cp_cache[select_md5]['roc']['auc_mean_std'],
                    title=[clf_name]
                )
                max_reports_dict = cp_cache['reports'].to_dict('records')
                bar_dict = cp_cache[select_md5]['bar_dict']
                heatmap_dict = cp_cache[select_md5]['heatmap_dict']
                heatmap_anno = cp_cache[select_md5]['heatmap_anno']
                valid_roc_traces = cp_cache[select_md5]['valid_roc_traces']
                line_chart_data = cp_cache[select_md5]['line_chart_data']
        elif feature_select_method == 'FSS' or feature_select_method == 'BSS':
            cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=10)
            start = time.perf_counter()
            if feature_select_method == 'FSS':
                selected_feature, max_scores = FSS_fun(features, svc, data3, label3, cv2)
            else:
                selected_feature, max_scores = BSS_fun(features, svc, data3, label3, cv2)
            # 得到最值
            max_index = max_scores.index(np.nanmax(max_scores))
            max_score = max(max_scores)
            max_features = selected_feature[:max_index + 1]
            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = data3.iloc[train_index[i], :], label3[train_index[i]]
                xtest, ytest = data3.iloc[test_index[i], :], label3[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(svc, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)
            end = time.perf_counter()
            print(round(end - start, 2))
            line_chart_data = []
            trace = {
                'mode': 'lines+markers',
                'name': clf_name,
                'type': 'scatter',
                'x': list(range(1, 21)),
                'y': max_scores
            }
            line_chart_data.append(trace)

            maxauc_index = np.array(tests).argmax()
            select_str = feature_select_method + clf_name + str(res[maxauc_index].get_params())
            select_md5 = md5_convert(select_str)[:6]
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)

            if select_md5 not in cp_cache.keys():

                final_reports, f_describe = customized_report(clf_name, res, data3, label3, preds, test_index,
                                                              max_features, tests)
                final_reports_dict = df2bp(final_reports)
                f_describe = np.round(f_describe.loc[("mean", 'min', 'max', 'std'), :],
                                      3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                f_describe_dict = f_describe.to_dict('records')

                # ROC
                mean_FPR, mean_TPR_df, auc_mean_std = get_ROC_info(clf_name, res, data3, label3, test_index, max_features,
                                                                   final_reports, preds)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=[clf_name])

                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []

                maxauc_index = np.array(tests).argmax()
                best_esti.append(res[maxauc_index])
                parameter.append(str(res[maxauc_index].get_params()))

                test_acc.append(final_reports["test_accuracy"][maxauc_index])
                precision.append(final_reports["precision"][maxauc_index])
                recall.append(final_reports["recall"][maxauc_index])
                f1_score.append(final_reports["f1-score"][maxauc_index])
                AUC.append(final_reports["AUC"][maxauc_index])
                feature_names.append(str(list(max_features)))
                max_reports = {'parameter': parameter,
                               'feature_names': feature_names,
                               'test_acc': test_acc,
                               'precision': precision,
                               'AUC': AUC,
                               'recall': recall,
                               'f1-score': f1_score}
                max_reports = pd.DataFrame(max_reports, index=[clf_name])
                max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                    max_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                max_reports = max_reports.reset_index().rename(
                    columns={'index': 'Method', 'f1-score': 'f1score'})

                para_str = feature_select_method + max_reports['Method'][0] + str(max_reports['parameter'][0])
                para_md5 = md5_convert(para_str)[:6]
                print(para_md5)
                # add parameter md5 and feature select method
                max_reports['md5'], max_reports['fsm'] = para_md5, feature_select_method

                '''validation'''
                validate_predict, validate_report = pre_valid(best_esti[0], validation_data, validation_label, max_features)
                # barplot
                bar_dict = mkbar(validate_report)
                # heatmap
                heatmap_dict, heatmap_anno = mkheatmap(validation_label, validate_predict, classes)
                # roc
                valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std = valid_roc_info(clf_name, best_esti[0],
                                                                                       validation_data,
                                                                                       validation_label, max_features)
                valid_roc_traces = mkroc(valid_mean_FPR, valid_mean_TPR_df, valid_auc_mean_std, title=[clf_name])

                report_describe_roc = {
                    'final_reports': final_reports,
                    'f_describe': f_describe,
                    'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
                    # validation
                    'bar_dict': bar_dict,
                    'heatmap_dict': heatmap_dict,
                    'heatmap_anno': heatmap_anno,
                    'valid_roc_traces': valid_roc_traces,
                    'line_chart_data': line_chart_data
                }

                # report_describe_roc = {
                #     'final_reports': final_reports,
                #     'f_describe': f_describe,
                #     'roc': {'mean_FPR': mean_FPR, 'mean_TPR_df': mean_TPR_df, 'auc_mean_std': auc_mean_std},
                #     # 'report': max_reports
                # }
                max_reports = pd.concat([cp_cache['reports'], max_reports], axis=0).drop_duplicates(keep='last')
                max_reports_dict = max_reports.to_dict('records')
                cp_cache[para_md5] = report_describe_roc
                cp_cache['reports'] = max_reports

                with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                          'wb') as f:
                    pickle.dump(cp_cache, f)
            else:
                final_reports_dict = df2bp(cp_cache[select_md5]['final_reports'])
                f_describe_dict = cp_cache[select_md5]['f_describe'].to_dict('records')
                roc_traces = mkroc(
                    cp_cache[select_md5]['roc']['mean_FPR'],
                    cp_cache[select_md5]['roc']['mean_TPR_df'],
                    cp_cache[select_md5]['roc']['auc_mean_std'],
                    title=[clf_name]
                )
                max_reports_dict = cp_cache['reports'].to_dict('records')
                bar_dict = cp_cache[select_md5]['bar_dict']
                heatmap_dict = cp_cache[select_md5]['heatmap_dict']
                heatmap_anno = cp_cache[select_md5]['heatmap_anno']
                valid_roc_traces = cp_cache[select_md5]['valid_roc_traces']
                line_chart_data = cp_cache[select_md5]['line_chart_data']
    if select_model == 'model_bclass':
        ifmarco = False
    elif select_model == 'model_mclass':
        ifmarco = True
    print('ifmarco:', ifmarco)
    return render(request, 'classification_cp_result.html', {
        'projectid': projectid,
        'final_reports_dict': final_reports_dict,
        'f_describe_dict': f_describe_dict,
        'roc_traces': json.dumps(roc_traces),
        'max_reports_dict': max_reports_dict,
        'ifmarco': ifmarco,
        'model_name': clf_name,
        'bar_dict': bar_dict,
        'heatmap_dict': json.dumps(heatmap_dict),
        'heatmap_anno': json.dumps(heatmap_anno),
        'valid_roc_traces': json.dumps(valid_roc_traces),
        'line_chart_data': json.dumps(line_chart_data),
    })

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
                return render(request, 'status.html', {
                    'status': status,
                    'projectid': projectid,
                })
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
            paramd5 = cp_cache['reports']['md5'][0]

        final_reports_dict = df2bp(cp_cache[paramd5]['final_reports'])
        f_describe_dict = cp_cache[paramd5]['f_describe'].to_dict('records')
        clf_name = cp_cache['reports'].loc[cp_cache['reports']['md5'] == paramd5]['Method'] \
            .to_list()[0]
        roc_traces = mkroc(
            cp_cache[paramd5]['roc']['mean_FPR'],
            cp_cache[paramd5]['roc']['mean_TPR_df'],
            cp_cache[paramd5]['roc']['auc_mean_std'],
            title=[clf_name]
        )
        max_reports_dict = cp_cache['reports'].to_dict('records')
        bar_dict = cp_cache[paramd5]['bar_dict']
        heatmap_dict = cp_cache[paramd5]['heatmap_dict']
        heatmap_anno = cp_cache[paramd5]['heatmap_anno']
        valid_roc_traces = cp_cache[paramd5]['valid_roc_traces']
        line_chart_data = cp_cache[paramd5]['line_chart_data']
        ifmarco = False
        return render(request, 'classification_cp_result.html', {
            'projectid': projectid,
            'final_reports_dict': final_reports_dict,
            'f_describe_dict': f_describe_dict,
            'roc_traces': json.dumps(roc_traces),
            'max_reports_dict': max_reports_dict,
            'ifmarco': ifmarco,
            'model_name': clf_name,
            'bar_dict': bar_dict,
            'heatmap_dict': json.dumps(heatmap_dict),
            'heatmap_anno': json.dumps(heatmap_anno),
            'valid_roc_traces': json.dumps(valid_roc_traces),
            'line_chart_data':json.dumps(line_chart_data),
            'change_page': True,
        })
    else:
        projectid = projectid_paramd5
        if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/classification_pickle.pkl'):
            status = 'Running'
            return render(request, 'status.html', {
                'status': status,
                'projectid': projectid,
            })
        else:
            with open(STATIC_ROOT + '/cache/' + projectid + '/classification_pickle.pkl', 'rb') as f:
                classification_pickle = pickle.load(f)

            test_acc_reports_dict, df_AUCs_dict, precision_reports_dict, \
            recall_reports_dict, f1_score_reports_dict, roc_traces, ifmarco \
                = classification_pickle['test_acc_reports_dict'], \
                  classification_pickle['df_AUCs_dict'], \
                  classification_pickle['precision_reports_dict'], \
                  classification_pickle['recall_reports_dict'], \
                  classification_pickle['f1_score_reports_dict'], \
                  classification_pickle['roc_traces'], \
                  classification_pickle['ifmarco']

            test_acc_describe_dict, df_AUCs_describe_dict, precision_describe_dict, \
            recall_describe_dict, f1_score_describe_dict, final_reports_dict, line_chart_data, radar_dict \
                = classification_pickle['test_acc_describe_dict'], \
                  classification_pickle['df_AUCs_describe_dict'], \
                  classification_pickle['precision_describe_dict'], \
                  classification_pickle['recall_describe_dict'], \
                  classification_pickle['f1_score_describe_dict'], \
                  classification_pickle['final_reports_dict'], \
                  classification_pickle['line_chart_data'], \
                  classification_pickle['radar_dict']

            vbar_trace, heatmap_data, heatmap_anno, valid_roc_traces, radar_range = classification_pickle['vbar_trace'], \
                                                                                    classification_pickle['heatmap_data'], \
                                                                                    classification_pickle['heatmap_anno'], \
                                                                                    classification_pickle[
                                                                                        'valid_roc_traces'], \
                                                                                    classification_pickle['radar_range']

            return render(request, 'classification_oc_result.html', {
                'projectid': projectid,
                'test_acc_reports_dict': test_acc_reports_dict,
                'test_acc_describe_dict': test_acc_describe_dict,
                'df_AUCs_dict': df_AUCs_dict,
                'df_AUCs_describe_dict': df_AUCs_describe_dict,
                'precision_reports_dict': precision_reports_dict,
                'precision_describe_dict': precision_describe_dict,
                'recall_reports_dict': recall_reports_dict,
                'recall_describe_dict': recall_describe_dict,
                'f1_score_reports_dict': f1_score_reports_dict,
                'f1_score_describe_dict': f1_score_describe_dict,
                'roc_traces': json.dumps(roc_traces),
                'final_reports_dict': json.dumps(final_reports_dict),
                'ifmarco': ifmarco,
                'line_chart_data': line_chart_data,
                'radar_dict': radar_dict,
                'radar_range': radar_range,
                'vbar_trace': json.dumps(vbar_trace),
                'heatmap_data': json.dumps(heatmap_data),
                'heatmap_anno': json.dumps(heatmap_anno),
                'valid_roc_traces': json.dumps(valid_roc_traces),
            })

# get ajax customize parameters ROC RADAR
def get_cp_combination(request):
    projectid = request.POST.get('projectid')
    print(projectid)
    # load pickle
    with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
        cp_cache = pickle.load(f)
    md5_list = list(cp_cache.keys())
    md5_list.remove('reports')
    # make roc data
    mean_FPR = cp_cache[md5_list[0]]['roc']['mean_FPR']
    combine_mean_TPR_df = pd.DataFrame()
    combine_auc_mean_std = pd.DataFrame()
    for md5 in md5_list:
        mean_TPR_df = cp_cache[md5]['roc']['mean_TPR_df']
        mean_TPR_df.columns = [md5]
        combine_mean_TPR_df = pd.concat([combine_mean_TPR_df,mean_TPR_df],axis=1)
        auc_mean_std = cp_cache[md5]['roc']['auc_mean_std']
        auc_mean_std.columns = [md5]
        combine_auc_mean_std = pd.concat([combine_auc_mean_std,auc_mean_std],axis=1)
    roc_traces = mkroc(mean_FPR, combine_mean_TPR_df, combine_auc_mean_std, title=md5_list)
    # make radar data
    final_report = cp_cache['reports'][['md5', 'test_acc', 'precision', 'AUC', 'recall', 'f1score']].set_index('md5').T
    radar_dict = mkradar(final_report)
    return HttpResponse(json.dumps({
        'roc_traces': roc_traces,
        'radar_dict': radar_dict,
    }))

'''
MODEL FUNCTIONS
'''
def svm(kernels='rbf',degrees=3,c=1,coef=0,Gamma = 'scale',):
    clf = SVC(kernel=kernels,degree=degrees,cache_size=5000,random_state=10,
              probability=False,max_iter=1000,C=c,coef0=coef,gamma=Gamma)
    return clf

def naivebayes(name,Alpha=1.0):
    naivebayes_model = {'GaussianNB':GaussianNB(),
                       'BernoulliNB':BernoulliNB(alpha=Alpha),
                       'ComplementNB':ComplementNB(alpha=Alpha),
                       'MultinomialNB':MultinomialNB(alpha=Alpha),}
    return naivebayes_model[name]

def kneighbors(Weight='uniform',N_neighbors=5,P=2,Agorithm='auto',Metric='minkowski',Leaf_size=30):
    knn = KNeighborsClassifier(n_jobs=4,weights=Weight,n_neighbors=N_neighbors,p=P,algorithm=Agorithm,
                              metric=Metric,leaf_size=Leaf_size)
    return knn

def logistic_reg(Penalty='l2',CC=1.0,Fit_intercept=True,Solver='lbfgs',L1_ratio=0.5):
    lr = LR(random_state=10,penalty=Penalty,C=CC,fit_intercept=Fit_intercept,solver=Solver,
            n_jobs=4,max_iter=1000,l1_ratio=L1_ratio)
    return lr

def decisiontree(Criterion='gini',Splitter='best',Max_depth=None,Min_samples_split=2,Min_samples_leaf=1,Max_features=None):
    dt = DecisionTreeClassifier(random_state=10,criterion=Criterion,splitter=Splitter,max_depth=Max_depth,
                            min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,max_features=Max_features)
    return dt

def randomforest(Criterion='gini',Max_depth=None,Min_samples_split=2,Min_samples_leaf=1,Max_features='auto',N_estimators=100):
    rf = RFC(random_state=10,criterion=Criterion,max_depth=Max_depth,n_jobs=4,max_features=Max_features,
             min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,n_estimators=N_estimators)
    return rf

def xgboost(Learning_rate=0.3,N_estimators=100,Min_child_weight=1,Subsample=1,Colsample_bytree=1,
                   Gamma=0,Reg_alpha=1,Reg_lambda=1):
    xgb = XGBClassifier(learning_rate=Learning_rate,n_estimators=N_estimators,min_child_weight=Min_child_weight,
                        subsample=Subsample,colsample_bytree=Colsample_bytree,gamma=Gamma,reg_alpha=Reg_alpha,
                        reg_lambda=Reg_lambda,n_jobs=4,random_state=10)
    return xgb

def lightgbm(Boosting_type='gbdt',N_estimators=100,Min_child_samples=20,Reg_alpha=0,Reg_lambda=0,Subsample=1,
             Colsample_bytree=1,Num_leaves=31,Max_bin=255,Learning_rate=0.1):
    lgbm = LGBMClassifier(boosting_type=Boosting_type,n_estimators=N_estimators,min_child_samples=Min_child_samples,
                      reg_alpha=Reg_alpha,reg_lambda=Reg_lambda,subsample=Subsample,colsample_bytree=Colsample_bytree,
                      num_leaves=Num_leaves,max_bin=Max_bin,random_state=10,n_jobs=4,learning_rate=Learning_rate)
    return lgbm

def adaboost(N_estimators=50, Learning_rate=1.0,Algorithm='SAMME.R',Max_depth=1):
    ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=Max_depth),n_estimators=N_estimators,
                                 learning_rate=Learning_rate,algorithm=Algorithm,random_state=10)
    return ada_clf

def gbdt(Loss='deviance',Learning_rate=0.1,Subsample=1.0,Criterion='friedman_mse',Min_samples_split=2,Max_features='auto',
         Min_samples_leaf=1,Max_depth=3,Validation_fraction=0.1,N_iter_no_change=None,N_estimators=100):
    gbdt_clf = GradientBoostingClassifier(loss=Loss,learning_rate=Learning_rate,subsample=Subsample,criterion=Criterion,
                                    min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,max_depth=Max_depth,max_features=Max_features,
                                    validation_fraction=Validation_fraction,n_iter_no_change=N_iter_no_change,
                                    n_estimators=N_estimators,random_state=10)
    return gbdt_clf
'''
METHODS
'''
def select_class_model(request):
    select_child_model = request.POST.get('select_child_model').replace('task_', '')
    if select_child_model == 'naivebayes':
        select_model_name = 'Naive Bayes'
        alpha, name = float(request.POST.get('naivebayes_alpha')), request.POST.get('naivebayes_name')
        select_model = naivebayes(name,Alpha=alpha)
    elif select_child_model == 'svm':
        select_model_name = 'SVM'
        kernel, c, degree, coef, gamma = request.POST.get('svm_kernel'), \
                                         int(request.POST.get('svm_c')), \
                                         request.POST.get('svm_degree'), \
                                         request.POST.get('svm_coef'), \
                                         request.POST.get('svm_gamma')
        if kernel == 'linear':
            select_model = svm(kernels=kernel, c=c)
        elif kernel == 'poly':
            degree, coef = int(degree), int(coef)
            select_model = svm(kernels=kernel, degrees=degree, c=c, coef=coef, Gamma=gamma)
        elif kernel == 'sigmoid':
            coef = int(coef)
            select_model = svm(kernels=kernel, c=c, coef=coef, Gamma=gamma)
        else:
            select_model = svm(kernels=kernel, c=c, Gamma=gamma)

    elif select_child_model == 'randomforest':
        select_model_name = 'RandomForest'
        criterion, max_depth, max_features, min_samples_split, \
        min_samples_leaf, n_estimators = request.POST.get('randomforest_criterion'), \
                                           request.POST.get('randomforest_max_depth'), \
                                           request.POST.get('randomforest_max_features'), \
                                           request.POST.get('randomforest_min_samples_split'), \
                                           request.POST.get('randomforest_min_samples_leaf'), \
                                           int(request.POST.get('randomforest_n_estimators'))
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = randomforest(Criterion=criterion,Max_depth=max_depth,Min_samples_split=min_samples_split,
                                    Min_samples_leaf=min_samples_leaf,Max_features=max_features,
                                    N_estimators=n_estimators)

    elif select_child_model == 'logistic':
        select_model_name = 'Logistic'
        penalty, solver, fit_intercept, C, l1_ratio = request.POST.get('logistic_penalty'), \
                                                        request.POST.get('logistic_solver'), \
                                                        bool(request.POST.get('logistic_fit_intercept')), \
                                                        request.POST.get('logistic_C'), \
                                                        request.POST.get('logistic_l1_ratio')
        # if penalty == 'none': penalty = None
        C = float(C)
        if penalty == 'elasticnet':
            l1_ratio = float(l1_ratio)
            select_model = logistic_reg(Penalty=penalty, CC=C, Fit_intercept=fit_intercept, Solver=solver, L1_ratio=l1_ratio)
        else:
            select_model = logistic_reg(Penalty=penalty, CC=C, Fit_intercept=fit_intercept, Solver=solver)
    elif select_child_model == 'knn':
        select_model_name = 'KNN'
        algorithm, metric, weights, n_neighbors, leaf_size, p = request.POST.get('knn_algorithm'), \
                                                                request.POST.get('knn_metric'), \
                                                                request.POST.get('knn_weights'), \
                                                                int(request.POST.get('knn_n_neighbors')), \
                                                                request.POST.get('knn_leaf_size'), \
                                                                request.POST.get('knn_p')

        if algorithm == 'ball_tree' or algorithm == 'kd_tree':
            leaf_size = int(leaf_size)
            if metric == 'minkowski':
                p = int(p)
                select_model = kneighbors(Weight=weights, N_neighbors=n_neighbors, P=p, Agorithm=algorithm,
                                          Metric=metric, Leaf_size=leaf_size)
            else:
                select_model = kneighbors(Weight=weights, N_neighbors=n_neighbors, Agorithm=algorithm,
                                          Metric=metric, Leaf_size=leaf_size)
        else:
            if metric == 'minkowski':
                p = int(p)
                select_model = kneighbors(Weight=weights, N_neighbors=n_neighbors, P=p, Agorithm=algorithm,
                                          Metric=metric)
            else:
                select_model = kneighbors(Weight=weights, N_neighbors=n_neighbors, Agorithm=algorithm,
                                          Metric=metric)

    elif select_child_model == 'xgboost':
        select_model_name = 'XGBoost'
        learning_rate, n_estimators,min_child_weight, subsample, colsample_bytree, Gamma, \
        reg_alpha, reg_lambda = float(request.POST.get('xgboost_learning_rate')), \
                                int(request.POST.get('xgboost_n_estimators')), \
                                int(request.POST.get('xgboost_min_child_weight')), \
                                int(request.POST.get('xgboost_subsample')), \
                                int(request.POST.get('xgboost_colsample_bytree')), \
                                int(request.POST.get('xgboost_Gamma')), \
                                int(request.POST.get('xgboost_reg_alpha')), \
                                int(request.POST.get('xgboost_reg_lambda'))
        select_model = xgboost(Learning_rate=learning_rate,N_estimators=n_estimators,Min_child_weight=min_child_weight,
                               Subsample=subsample,Colsample_bytree=colsample_bytree,Gamma=Gamma,Reg_alpha=reg_alpha,
                               Reg_lambda=reg_lambda)
    elif select_child_model == 'lightgbm':
        select_model_name = 'lightGBM'
        boosting_type, reg_alpha, reg_lambda, min_child_samples, subsample, colsample_bytree, learning_rate, n_estimators, \
        num_leaves, max_bin = request.POST.get('lightgbm_boosting_type'), \
                                float(request.POST.get('lightgbm_reg_alpha')), \
                                float(request.POST.get('lightgbm_reg_lambda')), \
                                int(request.POST.get('lightgbm_min_child_samples')), \
                                float(request.POST.get('lightgbm_subsample')), \
                                float(request.POST.get('lightgbm_colsample_bytree')), \
                                float(request.POST.get('lightgbm_learning_rate')), \
                                int(request.POST.get('lightgbm_n_estimators')), \
                                int(request.POST.get('lightgbm_num_leaves')), \
                                int(request.POST.get('lightgbm_max_bin'))
        select_model = lightgbm(Boosting_type=boosting_type,N_estimators=n_estimators,
                                Min_child_samples=min_child_samples,Reg_alpha=reg_alpha,Reg_lambda=reg_lambda,
                                Subsample=subsample,Colsample_bytree=colsample_bytree,Num_leaves=num_leaves,
                                Max_bin=max_bin,Learning_rate=learning_rate)
    elif select_child_model == 'adaboost':
        select_model_name = 'Adaboost'
        algorithm, n_estimators, learning_rate, max_depth = request.POST.get('adaboost_algorithm'), \
                                                             int(request.POST.get('adaboost_n_estimators')), \
                                                             float(request.POST.get('adaboost_learning_rate')), \
                                                             int(request.POST.get('adaboost_max_depth'))
        select_model = adaboost(N_estimators=n_estimators, Learning_rate=learning_rate,Algorithm=algorithm,Max_depth=max_depth)
    elif select_child_model == 'decisiontree':
        select_model_name = 'DecisionTree'
        criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features = \
                                                        request.POST.get('decisiontree_criterion'),\
                                                        request.POST.get('decisiontree_splitter'), \
                                                        request.POST.get('decisiontree_max_depth'), \
                                                        request.POST.get('decisiontree_min_samples_split'), \
                                                        request.POST.get('decisiontree_min_samples_leaf'), \
                                                        request.POST.get('decisiontree_max_features')
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = decisiontree(Criterion=criterion,Splitter=splitter,Max_depth=max_depth,
                                    Min_samples_split=min_samples_split,Min_samples_leaf=min_samples_leaf,
                                    Max_features=max_features)
    else:
        select_model_name = 'GBDT'
        criterion, loss, learning_rate, subsample, min_samples_split, min_samples_leaf, max_depth, max_features, \
        n_estimators, validation_fraction, n_iter_no_change = request.POST.get('gbdt_criterion'), \
                                                                             request.POST.get('gbdt_loss'), \
                                                                             float(request.POST.get('gbdt_learning_rate')), \
                                                                             float(request.POST.get('gbdt_subsample')), \
                                                                             request.POST.get('gbdt_min_samples_split'), \
                                                                             request.POST.get('gbdt_min_samples_leaf'), \
                                                                             request.POST.get('gbdt_max_depth'), \
                                                                             request.POST.get('gbdt_max_features'), \
                                                                             int(request.POST.get('gbdt_n_estimators')), \
                                                                             float(request.POST.get('gbdt_validation_fraction')), \
                                                                             request.POST.get('gbdt_n_iter_no_change')
        if n_iter_no_change == '': n_iter_no_change = None
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = gbdt(Loss=loss,Learning_rate=learning_rate,Subsample=subsample,
                            Criterion=criterion,Min_samples_split=min_samples_split,Max_features=max_features,
                            Min_samples_leaf=min_samples_leaf,Max_depth=max_depth,Validation_fraction=validation_fraction,
                            N_iter_no_change=n_iter_no_change,N_estimators=n_estimators)
    return select_model, select_model_name
'''
file preprocess
'''
def split_train_test(data,datatype='other'):
    train_set,test_set,blind_set = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    num = (1,2)[datatype == 'survival']  #datatype == 'survival'时选第三列，否则为第二列
    blind_set = data[data.iloc[:,:num].isna().T.any()]
    if len(blind_set)>0:
        blind_set = blind_set.drop(labels=blind_set.columns[num], axis=1)
    else:
        blind_set = pd.DataFrame()
    data2 = data[~data.index.isin(blind_set.index)]
    if 'training' in np.unique(data2.iloc[:,num]):
        train_set = data2[data2.iloc[:,num]=='training']
        train_set = train_set.drop(labels=train_set.columns[num], axis=1)
    else:
        train_set = data2.drop(labels=data2.columns[num], axis=1)
    if 'testing' in np.unique(data2.iloc[:,num]):
        test_set = data2[data2.iloc[:,num]=='testing']
        test_set = test_set.drop(labels=test_set.columns[num], axis=1)
    return train_set,test_set,blind_set

def classification_process(data):
    data = data.apply(pd.to_numeric,errors='ignore')#转成数值型
    x=data.iloc[:,data.columns!=data.columns[0]]
    if np.any(x.isnull()) == True:
        x=x.fillna(x.mean())  #填充缺失值
    y=np.array(data.iloc[:,0]).ravel()
    return x,y

'''
ml function
'''
#预处理部分
def label_pre(ml_label):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(ml_label)
    ml_label2 = le.transform(ml_label)#转为数值标签
    ml_label3 = le.inverse_transform(np.unique(ml_label2))
    classes = dict(zip(ml_label3,np.unique(ml_label2)))
    return ml_label2,classes

def selectkbest_top20(data,label,k=20,score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k='all').fit(data,label)
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(data.columns)
    df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    df_feature_scores.columns = ['Feature', 'Score']
    feature_names=df_feature_scores.sort_values(by='Score', ascending=False)[:k]['Feature']
    return feature_names

#n次k折数据拆分
def RSKFold (data,label,n=10,k=5):
    train_index = []
    test_index = []
    kf = RepeatedStratifiedKFold(n_splits=k,n_repeats=n,random_state=10)
    for train, test in kf.split(data,label):
        train_index.append(train)
        test_index.append(test)
    return train_index,test_index

#SVM分类器
#kernel = rbf linear poly sigmoid
# def svm(kernels='rbf',degrees=3,c=1,coef=0,max_iters=100):
#     clf = SVC(kernel=kernels,degree=degrees,cache_size=5000,random_state=10,
#               probability=False,max_iter=max_iters,C=c,coef0=coef)
#     return clf
#初筛
def pre_screening(data2,label,model,features,cv=2):
    #第一步筛选
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    clf = model
    cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=cv,).mean() for i in range(1,21)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num, cv_scores

#top3训练
def train_estimator(clf,xtrain,ytrain,xtest,ytest):
    clf = copy.deepcopy(clf)
    res = clf.fit(xtrain,ytrain)
    predict = res.predict(xtest)
    test_acc = accuracy_score(ytest,predict,normalize=True,)
    return res,test_acc,predict

def train_top3(clf,data,label,clf_num,train_index,test_index,feature_names):
    test_accs,estimators,predicts,f_names = {},{},{},{}
    mean_accs = []
    for j in range(len(clf_num)):    #top3分类器
        preds,tests,res,f_name = [],[],[],[]
        for i in range(len(train_index)):
            xtrain,ytrain = data.iloc[train_index[i],:],label[train_index[i]]
            xtest,ytest = data.iloc[test_index[i],:],label[test_index[i]]
            xtrain,xtest = xtrain.loc[:,feature_names[:clf_num[j]]],xtest.loc[:,feature_names[:clf_num[j]]]
            estimator,test_acc,predict = train_estimator(clf,xtrain,ytrain,xtest,ytest)
            tests.append(test_acc),res.append(estimator),preds.append(predict)
        mean_accs.append(np.mean(tests))
        test_accs[clf_num[j]] = tests
        estimators[clf_num[j]] = res
        predicts[clf_num[j]] = preds
        f_names[clf_num[j]] = feature_names[:clf_num[j]]
    #选择得分最高的topk
    topk = clf_num[mean_accs.index(max(mean_accs))]
    test_accs = test_accs[topk]
    estimators = estimators[topk]
    predicts = predicts[topk]
    f_names = f_names[topk]
    return test_accs,estimators,mean_accs,predicts,f_names

def customized_report(clf_name,estimator,data,label,predict,test_index,f_names,test_accs):
    from sklearn.preprocessing import label_binarize
    reports,final_report = {},{}
    AUC = []
    if len(np.unique(label)) > 2:#判断二分类还是多分类
        for i in range(len(predict)):
            xtest = data[f_names].iloc[test_index[i]]
            ytest = label[test_index[i]]
            report = classification_report(ytest,predict[i],output_dict=True)
            reports[i] = pd.DataFrame.from_dict(report)['macro avg'].iloc[:-1]
            if clf_name=="SVM" :
                proba = estimator[i].decision_function(xtest)
                y = label_binarize(ytest, classes=np.unique(ytest))
                fpr,tpr,Auc = macro_roc(estimator[i],xtest,y,proba,len(np.unique(label)))
            else:
                proba = estimator[i].predict_proba(xtest)
                Auc = roc_auc_score(ytest,proba,multi_class="ovr",average="macro")
            AUC.append(Auc)
    else:
        for i in range(len(predict)):
            xtest = data[f_names].iloc[test_index[i]]
            ytest = label[test_index[i]]
            report = classification_report(ytest,predict[i],output_dict=True)

            reports[i] = pd.DataFrame.from_dict(report).iloc[:-1,1]
            #auc
            if clf_name == 'SVM' :
                proba = estimator[i].decision_function(xtest)
                fprs, tprs, threshold = roc_curve(ytest,proba)
                Auc = auc(fprs,tprs)
            else:
                proba = estimator[i].predict_proba(xtest)[:,1]
                Auc = roc_auc_score(ytest,proba)
            AUC.append(Auc)
    final_report['precision'] = pd.DataFrame.from_dict(reports).T.loc[:,'precision']
    final_report['recall'] = pd.DataFrame.from_dict(reports).T.loc[:,'recall']
    final_report['f1-score'] = pd.DataFrame.from_dict(reports).T.loc[:,'f1-score']
    final_reports = pd.concat(final_report,axis=1)
    final_reports['test_accuracy'] = test_accs
    final_reports['AUC'] = AUC
    f_describe = final_reports.describe().loc[("mean",'min','max','std'),:]
    return final_reports,f_describe

# def macro_roc(estimator,xtest,ytest,proba,n_classes):
#     mean_fpr = np.linspace(0, 1, 100)
#     fpr,tpr,roc_auc = {},{},{}
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(ytest[:,i], proba[:,i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         #plt.plot(fpr[i], tpr[i])
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#     macro_fpr = (all_fpr)
#     macro_tpr = (mean_tpr)
#     macro_roc_auc = (auc(all_fpr,mean_tpr))
#     print(mean_tpr)
#     return macro_fpr,macro_tpr,macro_roc_auc

def macro_roc(estimator,xtest,ytest,proba,n_classes):
    mean_fpr = np.linspace(0, 1, 100)
    fpr,tpr,roc_auc = {},{},{}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytest[:,i], proba[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    macro_fpr = (all_fpr)
    macro_tpr = (mean_tpr)
    macro_roc_auc = (auc(all_fpr,mean_tpr))
    return macro_fpr,macro_tpr,macro_roc_auc,fpr,tpr,roc_auc

def get_ROC_info(clf_name,estimator,data,label,test_index,f_names,reports,predicts):
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    auc_mean_std = pd.DataFrame()
    tprs = []
    if len(np.unique(label)) > 2:
        for i in range(len(predicts)):  # 50重复次数
            xtest = data[f_names].iloc[test_index[i]]
            ytest = label[test_index[i]]
            y = label_binarize(ytest, classes=np.unique(ytest))
            if clf_name == 'SVM':
                proba = estimator[i].decision_function(xtest)
            else:
                proba = estimator[i].predict_proba(xtest)
            fpr, tpr, roc_auc, cfpr, ctpr, croc_auc = macro_roc(estimator[i], xtest, y, proba,
                                                                len(np.unique(label)))
            # fpr,tpr,roc_auc = macro_roc(estimator[i],xtest,y,proba,len(np.unique(ytest)))
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        # 对曲线进行插值，因为每个曲线的样本不一样，所以获取到的fpr和tpr也不一样长度，所以需要进行插值
        # 插值原理，获取所有fpr的值，然后将每个交叉验证的roc都插值成和fpr的值一样多的长度。并不会改变每个roc曲线的形状
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(reports['AUC'])
        std_auc = np.std(reports["AUC"])
    else:
        for i in range(len(test_index)):
            xtest = data[f_names].iloc[test_index[i]]
            ytest = label[test_index[i]]
            if clf_name == 'SVM':
                proba = estimator[i].decision_function(xtest)
            else:
                proba = estimator[i].predict_proba(xtest)[:, 1]
            fpr, tpr, threshold = roc_curve(ytest, proba)
            Auc = reports['AUC'][i]
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(reports['AUC'])
        std_auc = np.std(reports["AUC"])
    mean_TPR_df[clf_name] = mean_tpr
    auc_mean_std[clf_name] = [mean_auc, std_auc]
    auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, auc_mean_std

def FSS_fun(feature_names,clf,data,label,cv):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    for i in range(20):
        cv_scores = []
        for feature in feature_names2:
            train_feature = [feature] + selected_feature
            data1 = pd.DataFrame(data.loc[:,train_feature])
            cv_score = cross_val_score(clf,data1,label,cv=cv,n_jobs=4).mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        feature_names2.remove(feature_names2[max_index])
    return selected_feature,max_scores
def BSS_fun(feature_names,clf,data,label,cv):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    for i in range(49):
        cv_scores = []
        for feature in feature_names2:
            train_feature = feature_names2[:] #切片，独立于原列表
            train_feature.remove(feature)
            data1 = pd.DataFrame(data.loc[:,train_feature])
            cv_score = cross_val_score(clf,data1,label,cv=cv,n_jobs=4).mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        del feature_names2[max_index]
    selected_feature.append(feature_names2[0])  ###
    selected_feature.reverse() #反向排序
    max_scores.reverse()
    return selected_feature,max_scores

def pre_valid(estimator,vdata,vlabel,max_features):
    validate_predict = estimator.predict(vdata[max_features])
    if len(np.unique(vlabel))>2:
        validate_report = pd.DataFrame(classification_report(vlabel,validate_predict,output_dict=True)).T
        validate_report = validate_report.loc['macro avg',['precision','recall','f1-score']]
    else:
        validate_report = pd.DataFrame(classification_report(vlabel,validate_predict,output_dict=True)).T.iloc[1,:3]
    validate_report['accuracy'] = estimator.score(vdata[max_features],vlabel)
    return validate_predict,validate_report
'''
general functions
'''
def get_file_md5(file_name):
    """
    计算文件的md5
    :param file_name:
    :return:
    """
    m = hashlib.md5()   #创建md5对象
    with open(file_name,'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)  #更新md5对象

    return m.hexdigest()    #返回md5对象

def random_str():
    import random
    import string

    # 指定随机数长度
    r_num = 4

    # 生成数字 + 字母（字符串序列）
    token = string.ascii_letters + string.digits
    '''
        string.ascii_letters:生成大小写字母（type:字符串）
        string.digits:生成数字（type:字符串）
    '''

    # 随机选择 指定长度 随机码（字符串列表）
    token = random.sample(token, r_num)

    # 生成 数字 + 字母 随机数
    token = ''.join(token)

    # 加强版（一行代码）
    token = ''.join(random.sample(string.digits + string.ascii_letters, r_num))

def get_svc_model(request):
    kernel = request.POST.get('svm_kernel')
    c = request.POST.get('svm_c')
    degree = request.POST.get('svm_degree')
    coef = request.POST.get('svm_coef')
    if kernel == 'linear' or kernel == 'rbf':
        c = int(c)
        svc = svm(kernels=kernel, c=c, max_iters=500)
    elif kernel == 'poly':
        c, degree, coef = int(c),int(degree),int(coef)
        svc = svm(kernels=kernel, c=c, max_iters=500, coef=coef, degrees=degree)
    elif kernel == 'sigmoid':
        c, coef = int(c),int(coef)
        svc = svm(kernels=kernel, c=c, max_iters=500, coef=coef)
    return svc

def md5_convert(string):
    m = hashlib.md5()
    m.update(string.encode())
    return m.hexdigest()

def mkbar(validate_report):
    data = [{
        'x': list(validate_report.index),
        'y': list(validate_report),
        'type': 'bar',
        'text': list(np.round(validate_report,3)),
        'textposition': 'auto',
        'hoverinfo': 'none',
        'marker': {
            'color': 'rgb(158,202,225)',
            'opacity': 0.6,
            'line': {
                'color': 'rgb(8,48,107)',
                'width': 1.5
            }
        }
    }]
    return data

def mkheatmap(validation_label, validate_predict, classes):
    cm = confusion_matrix(validation_label, validate_predict).tolist()
    data = [{
        'z': cm,
        'x': list(classes),
        'y': list(classes),
        'type': 'heatmap',
        'hoverongaps': False,
        'showscale': False,
        'colorscale': [['0', 'rgb(255,240,230)'], ['0.5', 'rgb(250,130,50)'], ['1', 'rgb(130,40,0)']]
    }]
    annotation = []
    for i in range(len(list(classes))):
        for j in range(len(list(classes))):
            if cm[i][j] < 30:
                fontcolor = 'black'
            else:
                fontcolor = 'white'
            result = {
                'x': list(classes)[j],
                'y': list(classes)[i],
                'text': cm[i][j],
                'showarrow': False,
                'font': {
                    'color': fontcolor
                }
            }
            annotation.append(result)
    return data, annotation

def valid_roc_info(clf_name,estimator,vdata,vlabel,max_features):
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    # auc_mean_std = pd.DataFrame()
    tprs = []

    if len(np.unique(vlabel)) > 2:
        if clf_name == "SVM":
            proba = estimator.decision_function(vdata[max_features])
        else:
            proba = estimator.predict_proba(vdata[max_features])
        y = label_binarize(vlabel, classes=np.unique(vlabel))
        fpr, tpr, roc_auc, cfpr, ctpr, croc_auc = macro_roc(estimator, vdata, y, proba, len(np.unique(vlabel)))
        # mean_auc = np.mean(reports['AUC'])
        # std_auc = np.std(reports["AUC"])
    else:
        if clf_name == "SVM":
            proba = estimator.decision_function(vdata[max_features])
        else:
            proba = estimator.predict_proba(vdata[max_features])[:,1]
        fpr, tpr, threshold = roc_curve(vlabel, proba)
        roc_auc = auc(fpr, tpr)
        # Auc = reports['AUC'][i]
        # mean_auc = np.mean(reports['AUC'])
        # std_auc = np.std(reports["AUC"])
    interp_tpr = np.interp(mean_FPR, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_TPR_df[clf_name] = mean_tpr
    # auc_mean_std[clf_name] = [mean_auc, std_auc]
    # auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, roc_auc

def surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features):
    if max_depth == '': max_depth = None
    if max_depth != None: max_depth = np.int(max_depth)
    if max_features == '': max_features = None
    if max_features != 'auto' and max_features != 'sqrt' and max_features != 'log2' and max_features != None:
        max_features = np.float(max_features)
    # min_samples_leaf must be at least 1 or in (0, 0.5]
    if 0 < np.float(min_samples_leaf) <= 0.5:
        min_samples_leaf = np.float(min_samples_leaf)
    elif 1 <= np.float(min_samples_leaf):
        min_samples_leaf = np.int(min_samples_leaf)
    # min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0]
    if 0 < np.float(min_samples_split) <= 1.0:
        min_samples_split = np.float(min_samples_split)
    elif 1 <= np.float(min_samples_split):
        min_samples_split = np.int(min_samples_split)
    return max_depth, min_samples_split, min_samples_leaf, max_features
