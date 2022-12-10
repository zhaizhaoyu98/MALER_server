from django.shortcuts import render
from django.http import Http404, StreamingHttpResponse
import os, hashlib, shutil, pickle, time ,json
import pandas as pd
import numpy as np

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_curve,auc,classification_report,roc_auc_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from ML_WebServer.settings import STATIC_ROOT
import warnings
warnings.filterwarnings("ignore")

title = ["Naive Bayes","SVM","RandomForest","Logistic","KNN","XGBoost","lightGBM",'Adaboost',"DecisionTree","GBDT"]

def result(request):

    start_time = time.time()
    # model
    model = [GaussianNB(), SVC(cache_size=5000, probability=False), RFC(n_jobs=4, random_state=10),
             LR(max_iter=5000, n_jobs=4), KNeighborsClassifier(n_jobs=4), XGBClassifier(n_jobs=7, random_state=10),
             LGBMClassifier(importance_type='gain', n_jobs=4), AdaBoostClassifier(),
             DecisionTreeClassifier(random_state=10), GradientBoostingClassifier(random_state=10)]

    # Feature selection methods
    feature_select_method = request.POST.get('feature_select_method')
    print('feature_select_method: ', feature_select_method)
    '''
    IMPORRT DATA
    '''
    # label load
    obj_label = request.FILES.get('upload_label')
    print(obj_label.name)
    f = open(os.path.join(STATIC_ROOT, 'cache', obj_label.name), 'wb')
    for line in obj_label.chunks():
        f.write(line)
    f.close()

    # profile load
    obj_profile = request.FILES.get('upload_profile')
    f = open(os.path.join(STATIC_ROOT, 'cache', obj_profile.name), 'wb')
    for line in obj_profile.chunks():
        f.write(line)
    f.close()

    '''
    CALCULATE PROJECTID
    '''
    # calculate projectid(profile md5 + label md5)

    labelmd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_label.name))
    profilemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_profile.name))
    print('labelmd5:', labelmd5)
    print('profilemd5:', profilemd5)
    projectid = labelmd5[:6] + '-' + profilemd5[:6] + '-' + feature_select_method
    print(projectid)
    # projectid = '911c4d-c3ceec-FSS'
    # projectid = '3eb2d0-533bd2'
    # projectid = '1e57e8-c3ceec'
    # make project folder in cache
    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
        newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        shutil.move(STATIC_ROOT + '/cache/' + obj_profile.name, newpath)
        shutil.move(STATIC_ROOT + '/cache/' + obj_label.name, newpath)
        # read files
        # data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'express_data.csv', header=0, index_col=0).T
        # label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'label.csv', header=0, index_col=0)
        # label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'label_3columns.csv', header=0, index_col=0)
        # data = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'multi_express_data.csv', header=0, index_col=0).T
        # label = pd.read_csv(STATIC_ROOT + '/cache/' + '/' + projectid + '/' + 'multi_label.csv', header=0, index_col=0)
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + obj_profile.name, header=0, index_col=0).T
        label = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + obj_label.name, header=0, index_col=0)
        line_chart_data = 'null'

        if label.shape[1] == 2:
            blind_label, train_label, validation_label, blind_data, train_data, validation_data = label_data_split(
                label, data)
            label = train_label.iloc[:, 0]
            data = train_data.iloc[:, ]
            print(label.shape)

        label = np.array(label).ravel()
        label2, classes = label_pre(label)
        label_num = len(np.unique(label2))

        if feature_select_method == 'TopK':

            train_index, test_index = RSKFold(data, label2)  # 十次五折交叉验证
            '''
            DETERMINE BINARY OR MULTIPLE CLASSIFICATION
            '''
            if label_num == 2:
                # top3分类器初筛
                clf_nums = []
                for i in range(len(title)):
                    clf_num = pre_screening(data, label2, model[i])
                    clf_nums.append(clf_num)

                test_accs, estimators, mean_accs, predicts, f_names = {}, {}, {}, {}, {}
                for i in range(len(title)):
                    start = time.perf_counter()
                    test_accs[i], estimators[i], mean_accs[i], predicts[i], f_names[i] = train_top3(title[i], data,
                                                                                                    label2, clf_nums[i],
                                                                                                    train_index, test_index)
                    end = time.perf_counter()
                    print(round(end-start,2))

                # 评价指标acc，auc，precision，recall，f1-score
                test_acc_reports = pd.DataFrame(data=test_accs)
                test_acc_reports.columns = title
                test_acc_reports_dict = df2bp(test_acc_reports)
                test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :], 3).reset_index().rename(columns={'index':'Method'})  # 测试集准确率指数
                test_acc_describe_dict = test_acc_describe.to_dict('records')

                df_AUCs, df_AUCs_describe = TopK_all_AUC(estimators,data,label2,f_names,test_index,'binary') #auc
                df_AUCs_dict = df2bp(df_AUCs)
                df_AUCs_describe = np.round(df_AUCs_describe, 3).reset_index().rename(columns={'index':'Method'})
                df_AUCs_describe_dict = df_AUCs_describe.to_dict('records')

                precision_reports, precision_describe = classif_report(predicts,label2,test_index,'precision')
                precision_reports_dict = df2bp(precision_reports)
                precision_describe = np.round(precision_describe, 3).reset_index().rename(columns={'index':'Method'})
                precision_describe_dict = precision_describe.to_dict('records')

                recall_reports, recall_describe = classif_report(predicts,label2,test_index,'recall')
                recall_reports_dict = df2bp(recall_reports)
                recall_describe = np.round(recall_describe, 3).reset_index().rename(columns={'index':'Method'})
                recall_describe_dict = recall_describe.to_dict('records')

                f1_score_reports, f1_score_describe = classif_report(predicts,label2,test_index,'f1-score')
                f1_score_reports_dict = df2bp(f1_score_reports)
                f1_score_describe = np.round(f1_score_describe, 3).reset_index().rename(columns={'index':'Method'})
                f1_score_describe_dict = f1_score_describe.to_dict('records')

                # radar plot
                test_acc_radar = test_acc_describe.loc[test_acc_describe['Method'] == 'mean']
                test_acc_radar['Method'] = 'Test Accuracy'
                auc_radar = df_AUCs_describe.loc[df_AUCs_describe['Method'] == 'mean']
                auc_radar['Method'] = 'AUC'
                precision_radar = precision_describe.loc[precision_describe['Method'] == 'mean']
                precision_radar['Method'] = 'Precision'
                recall_radar = recall_describe.loc[recall_describe['Method'] == 'mean']
                recall_radar['Method'] = 'Recall'
                f1_score_radar = f1_score_describe.loc[f1_score_describe['Method'] == 'mean']
                f1_score_radar['Method'] = 'F1-score'
                mean_method_model = pd.concat([test_acc_radar,auc_radar,precision_radar,recall_radar,f1_score_radar])
                radar_dict = mkradar(mean_method_model)
                # ROC
                mean_FPR, mean_TPR_df, auc_mean_std = get_TopK_ROC_info(estimators, data, label2, f_names, test_index, df_AUCs, title=title)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=title)


                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []
                for i in range(len(estimators.keys())):
                    maxauc_index = test_acc_reports.iloc[:, i].argmax()
                    best_esti.append(estimators[i][maxauc_index])
                    parameter.append(str(estimators[i][maxauc_index].get_params()))
                    # train_acc.append(trains[maxauc_index][i])
                    test_acc.append(test_accs[i][maxauc_index])
                    precision.append(precision_reports.iloc[maxauc_index, i])
                    recall.append(recall_reports.iloc[maxauc_index, i])
                    f1_score.append(f1_score_reports.iloc[maxauc_index, i])
                    AUC.append(df_AUCs.iloc[maxauc_index, i])
                    feature_names.append(str(list(f_names[i][maxauc_index].values)))
                final_reports = {'parameter': parameter,
                                 # 'train_acc': train_acc,
                                 'feature_names': feature_names,
                                 'test_acc': test_acc,
                                 'precision': precision,
                                 'AUC': AUC,
                                 'recall': recall,
                                 'f1-score': f1_score}
                final_reports = pd.DataFrame(final_reports, index=title)
                final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                final_reports_dict = final_reports.reset_index().rename(columns={'index':'Method', 'f1-score':'f1score'}).to_dict('records')

                ifmarco = False
                acc_auc_precision_recall_f1score_roc_final = {'test_acc_reports_dict': test_acc_reports_dict,
                                                        'test_acc_describe_dict': test_acc_describe_dict,
                                                        'df_AUCs_dict': df_AUCs_dict,
                                                        'df_AUCs_describe_dict': df_AUCs_describe_dict,
                                                        'precision_reports_dict': precision_reports_dict,
                                                        'precision_describe_dict': precision_describe_dict,
                                                        'recall_reports_dict': recall_reports_dict,
                                                        'recall_describe_dict': recall_describe_dict,
                                                        'f1_score_reports_dict': f1_score_reports_dict,
                                                        'f1_score_describe_dict': f1_score_describe_dict,
                                                        'roc_traces': roc_traces,
                                                        'radar_dict': radar_dict,
                                                        'final_reports_dict': final_reports_dict,
                                                        'ifmarco': ifmarco,
                                                        'line_chart_data':line_chart_data}

                with open(STATIC_ROOT + '/cache/' + projectid + '/acc_auc_precision_recall_f1score_roc_final.pkl', 'wb') as f:
                    pickle.dump(acc_auc_precision_recall_f1score_roc_final, f)

                # generate model pickle files
                for i in range(final_reports.shape[0]):
                    t = title[i].replace(' ', '_')
                    model = best_esti[i]
                    with open(STATIC_ROOT + '/cache/' + projectid + '/' + t + '.pkl', 'wb') as f:
                        pickle.dump(model, f)

                print('analysis time: ', time.time() - start_time)
                # no speed up analysis time:  75.98924517631531
                # speed up analysis time:  16.308971881866455
            elif label_num > 2:
                clf_nums = []
                for i in range(len(title)):
                    start = time.perf_counter()
                    clf_num = multi_label_pre_screening(data, label2, model[i])
                    clf_nums.append(clf_num)
                    end = time.perf_counter()
                    print(title[i], ":", round(end - start, 2), "s")

                test_accs, estimators, mean_accs, predicts, f_names = {}, {}, {}, {}, {}
                for i in range(len(title)):
                    start = time.perf_counter()
                    test_accs[i], estimators[i], mean_accs[i], predicts[i], f_names[i] = train_top3(title[i], data,
                                                                                                    label2, clf_nums[i],
                                                                                                    train_index, test_index)
                    end = time.perf_counter()
                    print(title[i], ":", round(end - start, 2), "s")

                # 评价指标acc，auc，precision，recall，f1-score
                test_acc_reports = pd.DataFrame(data=test_accs)
                test_acc_reports.columns = title
                test_acc_reports_dict = df2bp(test_acc_reports)
                test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                             3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                test_acc_describe_dict = test_acc_describe.to_dict('records')

                df_AUCs, df_AUCs_describe = TopK_all_AUC(estimators, data, label2, f_names, test_index, 'multiple')
                df_AUCs_dict = df2bp(df_AUCs)
                df_AUCs_describe_dict = np.round(df_AUCs_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                precision_reports, precision_describe = multi_classif_report(predicts, label2, test_index, 'precision')
                precision_reports_dict = df2bp(precision_reports)
                precision_describe_dict = np.round(precision_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                recall_reports, recall_describe = multi_classif_report(predicts, label2, test_index, 'recall')
                recall_reports_dict = df2bp(recall_reports)
                recall_describe_dict = np.round(recall_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                f1_score_reports, f1_score_describe = multi_classif_report(predicts, label2, test_index, 'f1-score')
                f1_score_reports_dict = df2bp(f1_score_reports)
                f1_score_describe_dict = np.round(f1_score_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                # ROC
                mean_FPR, mean_TPR_df, auc_mean_std = multi_label_get_TopK_ROC_info(estimators, data, label2, f_names, test_index, df_AUCs, df_AUCs_describe,
                                                                   title=title)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=title)

                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []
                for i in range(len(estimators.keys())):
                    maxauc_index = test_acc_reports.iloc[:, i].argmax()
                    best_esti.append(estimators[i][maxauc_index])
                    parameter.append(str(estimators[i][maxauc_index].get_params()))
                    # train_acc.append(trains[maxauc_index][i])
                    test_acc.append(test_accs[i][maxauc_index])
                    precision.append(precision_reports.iloc[maxauc_index, i])
                    recall.append(recall_reports.iloc[maxauc_index, i])
                    f1_score.append(f1_score_reports.iloc[maxauc_index, i])
                    AUC.append(df_AUCs.iloc[maxauc_index, i])
                    feature_names.append(str(list(f_names[i][maxauc_index].values)))
                final_reports = {'parameter': parameter,
                                 # 'train_acc': train_acc,
                                 'feature_names': feature_names,
                                 'test_acc': test_acc,
                                 'precision': precision,
                                 'AUC': AUC,
                                 'recall': recall,
                                 'f1-score': f1_score}
                final_reports = pd.DataFrame(final_reports, index=title)
                final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                    final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                final_reports_dict = final_reports.reset_index().rename(
                    columns={'index': 'Method', 'f1-score': 'f1score'}).to_dict('records')
                # if multi label or not
                ifmarco = True
                acc_auc_precision_recall_f1score_roc_final = {'test_acc_reports_dict': test_acc_reports_dict,
                                                              'test_acc_describe_dict': test_acc_describe_dict,
                                                              'df_AUCs_dict': df_AUCs_dict,
                                                              'df_AUCs_describe_dict': df_AUCs_describe_dict,
                                                              'precision_reports_dict': precision_reports_dict,
                                                              'precision_describe_dict': precision_describe_dict,
                                                              'recall_reports_dict': recall_reports_dict,
                                                              'recall_describe_dict': recall_describe_dict,
                                                              'f1_score_reports_dict': f1_score_reports_dict,
                                                              'f1_score_describe_dict': f1_score_describe_dict,
                                                              'roc_traces': roc_traces,
                                                              'final_reports_dict': final_reports_dict,
                                                              'ifmarco': ifmarco,
                                                              'line_chart_data':line_chart_data}

                with open(STATIC_ROOT + '/cache/' + projectid + '/acc_auc_precision_recall_f1score_roc_final.pkl',
                          'wb') as f:
                    pickle.dump(acc_auc_precision_recall_f1score_roc_final, f)

                # generate model pickle files
                for i in range(final_reports.shape[0]):
                    t = title[i].replace(' ', '_')
                    model = best_esti[i]
                    with open(STATIC_ROOT + '/cache/' + projectid + '/' + t + '.pkl', 'wb') as f:
                        pickle.dump(model, f)

                print('analysis time: ', time.time() - start_time)

            else:
                print('error')
                return render(request, 'ERROR.html', {
                    'error_msg': 'Invalid input!'
                })
        elif feature_select_method == 'FSS' or feature_select_method == 'BSS':
            '''
            DETERMINE BINARY OR MULTIPLE CLASSIFICATION
            '''
            feature_names = selectkbest_top20(data, label2, k=50)
            data2 = data.loc[:, feature_names]
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=10)
            # 所有分类器
            selected_feature, max_scores = [], []
            for each_model in model:
                start = time.perf_counter()
                if feature_select_method == 'FSS':
                    sf, ms = FSS_fun(feature_names, each_model, cv, data2, label2)
                else:
                    sf, ms = BSS_fun(feature_names, each_model, cv, data2, label2)
                selected_feature.append(sf), max_scores.append(ms)
                end = time.perf_counter()
                print(round(end - start, 3))
            line_chart_data = []
            for f in range(len(max_scores)):
                if len(np.argwhere(np.isnan(max_scores[f]))) == 1:
                    xnum = list(range(1, 21))
                    xnum.pop(np.argwhere(np.isnan(max_scores[f]))[0][0])
                    ynum = max_scores[f]
                    ynum.pop(np.argwhere(np.isnan(max_scores[f]))[0][0])
                    trace = {
                        'mode': 'lines+markers',
                        'name': title[f],
                        'type': 'scatter',
                        'x': xnum,
                        'y': ynum
                    }
                else:
                    trace = {
                        'mode': 'lines+markers',
                        'name': title[f],
                        'type': 'scatter',
                        'x': list(range(1, 21)),
                        'y': max_scores[f]
                    }
                line_chart_data.append(trace)

            max_indexs, max_score, max_features = [], [], []
            for i in range(len(max_scores)):
                max_index = max_scores[i].index(np.nanmax(max_scores[i]))
                max_indexs.append(max_index)
                max_score.append(max(max_scores[i]))
                max_features.append(selected_feature[i][:max_index + 1])
            train_index, test_index = RSKFold(data2, label2)
            test_accs, estimators, predicts = {}, {}, {}
            for j in range(len(title)):
                preds, tests, res = [], [], []
                start = time.perf_counter()
                for i in range(len(train_index)):
                    xtrain, ytrain = data2.iloc[train_index[i], :], label2[train_index[i]]
                    xtest, ytest = data2.iloc[test_index[i], :], label2[test_index[i]]
                    xtrain, xtest = xtrain[max_features[j]], xtest[max_features[j]]
                    estimator, test_acc, predict = train_estimator(title[j], xtrain, ytrain, xtest, ytest)
                    tests.append(test_acc), res.append(estimator), preds.append(predict)
                test_accs[j] = tests
                estimators[j] = res
                predicts[j] = preds
                end = time.perf_counter()
                print(round(end - start, 2))
            if label_num == 2:

                # 评价指标acc，auc，precision，recall，f1-score
                test_acc_reports = pd.DataFrame(data=test_accs)
                test_acc_reports.columns = title
                test_acc_reports_dict = df2bp(test_acc_reports)
                test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                             3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                test_acc_describe_dict = test_acc_describe.to_dict('records')

                df_AUCs, df_AUCs_describe = FSS_BSS_all_AUC(estimators, data, label2, test_index, max_features,'binary')  # auc
                df_AUCs_dict = df2bp(df_AUCs)
                df_AUCs_describe_dict = np.round(df_AUCs_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                precision_reports, precision_describe = classif_report(predicts, label2, test_index, 'precision')
                precision_reports_dict = df2bp(precision_reports)
                precision_describe_dict = np.round(precision_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                recall_reports, recall_describe = classif_report(predicts, label2, test_index, 'recall')
                recall_reports_dict = df2bp(recall_reports)
                recall_describe_dict = np.round(recall_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                f1_score_reports, f1_score_describe = classif_report(predicts, label2, test_index, 'f1-score')
                f1_score_reports_dict = df2bp(f1_score_reports)
                f1_score_describe_dict = np.round(f1_score_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                # ROC
                mean_FPR, mean_TPR_df, auc_mean_std = get_FSS_BSS_ROC_info(estimators, data, label2, max_features, test_index,
                                                                   df_AUCs, title=title)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=title)

                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []
                for i in range(len(estimators)):
                    maxauc_index = test_acc_reports.iloc[:, i].argmax()
                    best_esti.append(estimators[i][maxauc_index])
                    parameter.append(str(estimators[i][maxauc_index].get_params()))
                    # train_acc.append(trains[maxauc_index][i])
                    test_acc.append(test_accs[i][maxauc_index])
                    precision.append(precision_reports.iloc[maxauc_index, i])
                    recall.append(recall_reports.iloc[maxauc_index, i])
                    f1_score.append(f1_score_reports.iloc[maxauc_index, i])
                    AUC.append(df_AUCs.iloc[maxauc_index, i])
                    feature_names.append(str(max_features[i]))
                final_reports = {'parameter': parameter,
                                 # 'train_acc':train_acc,
                                 'feature_names': feature_names,
                                 'test_acc': test_acc,
                                 'precision': precision,
                                 'AUC': AUC,
                                 'recall': recall,
                                 'f1-score': f1_score}
                final_reports = pd.DataFrame(final_reports, index=title)
                final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                    final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                final_reports_dict = final_reports.reset_index().rename(
                    columns={'index': 'Method', 'f1-score': 'f1score'}).to_dict('records')

                ifmarco = False
                acc_auc_precision_recall_f1score_roc_final = {'test_acc_reports_dict': test_acc_reports_dict,
                                                              'test_acc_describe_dict': test_acc_describe_dict,
                                                              'df_AUCs_dict': df_AUCs_dict,
                                                              'df_AUCs_describe_dict': df_AUCs_describe_dict,
                                                              'precision_reports_dict': precision_reports_dict,
                                                              'precision_describe_dict': precision_describe_dict,
                                                              'recall_reports_dict': recall_reports_dict,
                                                              'recall_describe_dict': recall_describe_dict,
                                                              'f1_score_reports_dict': f1_score_reports_dict,
                                                              'f1_score_describe_dict': f1_score_describe_dict,
                                                              'roc_traces': roc_traces,
                                                              'final_reports_dict': final_reports_dict,
                                                              'ifmarco': ifmarco,
                                                              'line_chart_data': line_chart_data}

                with open(STATIC_ROOT + '/cache/' + projectid + '/acc_auc_precision_recall_f1score_roc_final.pkl',
                          'wb') as f:
                    pickle.dump(acc_auc_precision_recall_f1score_roc_final, f)

                # generate model pickle files
                for i in range(final_reports.shape[0]):
                    t = title[i].replace(' ', '_')
                    model = best_esti[i]
                    with open(STATIC_ROOT + '/cache/' + projectid + '/' + t + '.pkl', 'wb') as f:
                        pickle.dump(model, f)

                print('analysis time: ', time.time() - start_time)
            elif label_num > 2:
                test_acc_reports = pd.DataFrame(data=test_accs)
                test_acc_reports.columns = title
                test_acc_reports_dict = df2bp(test_acc_reports)
                test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                             3).reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
                test_acc_describe_dict = test_acc_describe.to_dict('records')

                df_AUCs, df_AUCs_describe = FSS_BSS_all_AUC(estimators, data2, label2, test_index, max_features,
                                                            'multiple')  # auc
                df_AUCs_dict = df2bp(df_AUCs)
                df_AUCs_describe_dict = np.round(df_AUCs_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                precision_reports, precision_describe = multi_classif_report(predicts, label2, test_index, 'precision')
                precision_reports_dict = df2bp(precision_reports)
                precision_describe_dict = np.round(precision_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                recall_reports, recall_describe = multi_classif_report(predicts, label2, test_index, 'recall')
                recall_reports_dict = df2bp(recall_reports)
                recall_describe_dict = np.round(recall_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                f1_score_reports, f1_score_describe = multi_classif_report(predicts, label2, test_index, 'f1-score')
                f1_score_reports_dict = df2bp(f1_score_reports)
                f1_score_describe_dict = np.round(f1_score_describe, 3).reset_index().rename(
                    columns={'index': 'Method'}).to_dict('records')

                # ROC
                mean_FPR, mean_TPR_df, auc_mean_std = multi_label_get_FSS_BSS_ROC_info(estimators, data2, label2,
                                                                           test_index,
                                                                           df_AUCs, max_features,df_AUCs_describe, title=title)
                roc_traces = mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=title)

                # 最优分类器表格展示
                parameter, train_acc, test_acc, best_esti = [], [], [], []
                precision, AUC, recall, f1_score = [], [], [], []
                feature_names = []
                for i in range(len(estimators)):
                    maxauc_index = test_acc_reports.iloc[:, i].argmax()
                    best_esti.append(estimators[i][maxauc_index])
                    parameter.append(str(estimators[i][maxauc_index].get_params()))
                    # train_acc.append(trains[maxauc_index][i])
                    test_acc.append(test_accs[i][maxauc_index])
                    precision.append(precision_reports.iloc[maxauc_index, i])
                    recall.append(recall_reports.iloc[maxauc_index, i])
                    f1_score.append(f1_score_reports.iloc[maxauc_index, i])
                    AUC.append(df_AUCs.iloc[maxauc_index, i])
                    feature_names.append(str(max_features[i]))

                final_reports = {'parameter': parameter,
                                 # 'train_acc':train_acc,
                                 'feature_names': feature_names,
                                 'test_acc': test_acc,
                                 'precision': precision,
                                 'AUC': AUC,
                                 'recall': recall,
                                 'f1-score': f1_score}
                final_reports = pd.DataFrame(final_reports, index=title)
                final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']] = np.round(
                    final_reports[['test_acc', 'precision', 'AUC', 'recall', 'f1-score']], 3)
                final_reports_dict = final_reports.reset_index().rename(
                    columns={'index': 'Method', 'f1-score': 'f1score'}).to_dict('records')

                ifmarco = True
                acc_auc_precision_recall_f1score_roc_final = {'test_acc_reports_dict': test_acc_reports_dict,
                                                              'test_acc_describe_dict': test_acc_describe_dict,
                                                              'df_AUCs_dict': df_AUCs_dict,
                                                              'df_AUCs_describe_dict': df_AUCs_describe_dict,
                                                              'precision_reports_dict': precision_reports_dict,
                                                              'precision_describe_dict': precision_describe_dict,
                                                              'recall_reports_dict': recall_reports_dict,
                                                              'recall_describe_dict': recall_describe_dict,
                                                              'f1_score_reports_dict': f1_score_reports_dict,
                                                              'f1_score_describe_dict': f1_score_describe_dict,
                                                              'roc_traces': roc_traces,
                                                              'final_reports_dict': final_reports_dict,
                                                              'ifmarco': ifmarco,
                                                              'line_chart_data': line_chart_data}

                with open(STATIC_ROOT + '/cache/' + projectid + '/acc_auc_precision_recall_f1score_roc_final.pkl',
                          'wb') as f:
                    pickle.dump(acc_auc_precision_recall_f1score_roc_final, f)

                # generate model pickle files
                for i in range(final_reports.shape[0]):
                    t = title[i].replace(' ', '_')
                    model = best_esti[i]
                    with open(STATIC_ROOT + '/cache/' + projectid + '/' + t + '.pkl', 'wb') as f:
                        pickle.dump(model, f)

                print('analysis time: ', time.time() - start_time)
            else:
                print('error')
                return render(request, 'ERROR.html', {
                    'error_msg': 'Invalid input!'
                })
    else:
        with open(STATIC_ROOT + '/cache/' + projectid + '/acc_auc_precision_recall_f1score_roc_final.pkl', 'rb') as f:
            acc_auc_precision_recall_f1score_roc = pickle.load(f)

        test_acc_reports_dict, df_AUCs_dict, precision_reports_dict, \
        recall_reports_dict, f1_score_reports_dict, roc_traces, ifmarco \
            = acc_auc_precision_recall_f1score_roc['test_acc_reports_dict'],\
              acc_auc_precision_recall_f1score_roc['df_AUCs_dict'],\
              acc_auc_precision_recall_f1score_roc['precision_reports_dict'],\
              acc_auc_precision_recall_f1score_roc['recall_reports_dict'],\
              acc_auc_precision_recall_f1score_roc['f1_score_reports_dict'],\
              acc_auc_precision_recall_f1score_roc['roc_traces'], \
              acc_auc_precision_recall_f1score_roc['ifmarco']

        test_acc_describe_dict, df_AUCs_describe_dict, precision_describe_dict, \
        recall_describe_dict, f1_score_describe_dict, final_reports_dict, line_chart_data, radar_dict \
            = acc_auc_precision_recall_f1score_roc['test_acc_describe_dict'], \
              acc_auc_precision_recall_f1score_roc['df_AUCs_describe_dict'], \
              acc_auc_precision_recall_f1score_roc['precision_describe_dict'], \
              acc_auc_precision_recall_f1score_roc['recall_describe_dict'], \
              acc_auc_precision_recall_f1score_roc['f1_score_describe_dict'], \
              acc_auc_precision_recall_f1score_roc['final_reports_dict'], \
              acc_auc_precision_recall_f1score_roc['line_chart_data'], \
              acc_auc_precision_recall_f1score_roc['radar_dict']


    return render(request, 'result.html', {
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
    })

def download_model(request, projectid_model):
    projectid = projectid_model.split('_')[0]
    model = projectid_model.split('_')[1].replace(' ','_')
    # print(projectid,model)
    file_path = (STATIC_ROOT + '/cache/' + projectid + '/' + model + '.pkl')
    try:
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + projectid + '_' + os.path.basename(file_path)
        return response
    except Exception:
        raise Http404


# def get_model(request):
#     model = request.POST.get('model').replace(' ','_')
#     projectid = request.POST.get('projectid')
#     print(model, projectid)
#     file_path = (STATIC_ROOT + '/cache/' + projectid + '/' + model + '.pkl')
#     try:
#         response = StreamingHttpResponse(open(file_path, 'rb'))
#         response['content_type'] = "application/octet-stream"
#         response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
#         return response
#     except Exception:
#         raise Http404


'''
machine learning functions
'''
def selectkbest_top20(data,label,k=20,score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k='all').fit(data,label)
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(data.columns)
    df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    df_feature_scores.columns = ['Feature', 'Score']
    feature_names=df_feature_scores.sort_values(by='Score', ascending=False)[:k]['Feature']
    return feature_names
#初筛
def pre_screening(data,label,model):
    #第一步筛选
    selector = SelectKBest(score_func=f_classif,k=100).fit(data,label)
    data2 = data.iloc[:,selector.get_support(indices=True)]
    feature_names = selectkbest_top20(data2,label,k=20,score_func=f_classif)
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    cv_scores = list()
    clf = model
    for i in range(1,21):
        cv_score = cross_val_score(clf,data2[:,:i],label,cv=2,).mean()
        cv_scores.append(cv_score)
    # from collections import Counter  #使用字典方法
    # clf_num = Counter(cv_scores).most_common()[:3] #输出顺序以及准确率的值
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num

def multi_label_pre_screening(data2,label,model):
    #第一步筛选
    feature_names = selectkbest_top20(data2,label,k=20,score_func=f_classif)
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    clf = model
    cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=2,).mean() for i in range(1,21)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num

#标签预处理
def label_pre(ml_label):
    le = LabelEncoder().fit(ml_label)
    ml_label2 = le.transform(ml_label)#转为数值标签
    ml_label3 = le.inverse_transform(np.unique(ml_label2))
    #classes = list(map(lambda x, y: {x:y},np.unique(ml_label2),)) #lambda 做map映射
    classes = dict(zip(ml_label3,np.unique(ml_label2)))
    return ml_label2,classes

# #特征筛选
# def selectkbest(data,label,k=100):
#     features = data.columns
#     select = SelectKBest(score_func=chi2,k=k)
#     z = select.fit_transform(data,label)# 拟合数据
#     filter = select.get_support()  #select.get_support(indices=True)返回索引
#     return features[filter]

#n次k折数据拆分
def RSKFold (data,label,n=10,k=5):
    train_index = []
    test_index = []
    kf = RepeatedStratifiedKFold(n_splits=k,n_repeats=n,random_state=10)
    for train, test in kf.split(data,label):
        train_index.append(train)
        test_index.append(test)
    return train_index,test_index

#top3训练
def train_estimator(clf_name,xtrain,ytrain,xtest,ytest):
    model = [GaussianNB(),SVC(cache_size=5000,probability=True),RFC(n_jobs=4,random_state=10),
             LR(max_iter=5000,n_jobs=4),KNeighborsClassifier(n_jobs=4),XGBClassifier(n_jobs=7,random_state=10),
             LGBMClassifier(importance_type='gain',n_jobs=4),AdaBoostClassifier(),
             DecisionTreeClassifier(random_state=10),GradientBoostingClassifier(random_state=10)]
    models = dict(zip(title,model))
    clf = models[clf_name]
    res = clf.fit(xtrain,ytrain)
    predict = res.predict(xtest)
    test_acc = accuracy_score(ytest,predict,normalize=True,)
    return res,test_acc,predict

def train_top3(clf, data, label, clf_num, train_index, test_index):
    test_accs, estimators, predicts, f_names = dict(), dict(), dict(), dict()
    mean_accs = []
    for j in range(len(clf_num)):  # top3分类器
        preds, tests, res, f_name = [], [], [], []
        xtrains, xtests, ytrains, ytests = [], [], [], []
        for i in range(len(train_index)):
            xtrain, ytrain = data.iloc[train_index[i], :], label[train_index[i]]
            xtest, ytest = data.iloc[test_index[i], :], label[test_index[i]]
            # 对训练集进行特征筛选
            selector = SelectKBest(f_classif, k=100).fit(xtrain, ytrain)
            feature_names = selectkbest_top20(xtrain.iloc[:, selector.get_support(indices=True)], ytrain, k=20)

            xtrain, xtest = xtrain.loc[:, feature_names[:clf_num[j]]], xtest.loc[:, feature_names[:clf_num[j]]]
            estimator, test_acc, predict = train_estimator(clf, xtrain, ytrain, xtest, ytest)
            tests.append(test_acc), res.append(estimator), f_name.append(feature_names[:clf_num[j]]), preds.append(
                predict)
        mean_accs.append(np.mean(tests))
        test_accs[clf_num[j]] = tests
        estimators[clf_num[j]] = res
        predicts[clf_num[j]] = preds
        f_names[clf_num[j]] = f_name
    # 选择得分最高的topk
    topk = clf_num[mean_accs.index(max(mean_accs))]
    test_accs = test_accs[topk]
    estimators = estimators[topk]
    predicts = predicts[topk]
    f_names = f_names[topk]
    return test_accs, estimators, mean_accs, predicts, f_names

def defalut_ml(xtrain,ytrain,xtest,ytest,njobs = 8):
    tests = []
    trains = []
    estimators = []
    predicts = []
    # title = ["Naive Bayes","SVM","RandomForest","Logistic","KNN","XGBoost"]
    model = [GaussianNB(),SVC(cache_size=2000,probability=True),RFC(n_jobs=njobs),
             LR(max_iter=1000),KNeighborsClassifier(n_jobs=njobs)  ,XGBClassifier(n_jobs=njobs)]
    for estimator in model:
        res = estimator.fit(xtrain,ytrain)
        predict = res.predict(xtest)
        trains.append(res.score(xtrain,ytrain)) #训练集准确率
        predicts.append(predict) #测试集预测值
        tests.append(accuracy_score(ytest,predict,normalize=True,)) #测试集准确率
        estimators.append(res) #保存模型训练结果
    return estimators,trains,tests,predicts

def TopK_all_AUC(estimators,data,label,f_names,test_index,type,title=title):
        AUCs = dict()
        for i in range(len(estimators[0])):  # 50
            AUC = []
            fprs = dict()
            tprs = dict()
            if type == 'binary':
                for c in range(len(estimators)):  # 6
                    xtest = data[f_names[c][i]].iloc[test_index[i]]
                    ytest = label[test_index[i]]
                    proba = estimators[c][i].predict_proba(xtest)[:, 1]
                    fprs[c], tprs[c], threshold = roc_curve(ytest, proba)
                    Auc = auc(fprs[c], tprs[c])
                    AUC.append(Auc)
            elif type == 'multiple':
                for c in range(len(estimators)):  # 6
                    xtest = data[f_names[c][i]].iloc[test_index[i]]
                    ytest = label[test_index[i]]
                    proba = estimators[c][i].predict_proba(xtest)
                    Auc = roc_auc_score(ytest, proba, multi_class="ovr", average="macro")
                    AUC.append(Auc)
            AUCs[i] = AUC
        df_AUCs = pd.DataFrame(AUCs, index=title).T
        df_AUCs_describe = df_AUCs.describe().loc[("mean", 'min', 'max', 'std'), :]
        return df_AUCs, df_AUCs_describe


def FSS_BSS_all_AUC(estimators,data,label,test_index,f_names,type,title=title):
    AUCs = dict()
    for i in range(len(estimators[0])): #50
        AUC = []
        fprs,tprs = {},{}
        if type == 'binary':
            for c in range(len(estimators)):  # 6
                xtest = data[f_names[c]].iloc[test_index[i]]
                ytest = label[test_index[i]]
                if c == 1:
                    proba = estimators[c][i].decision_function(xtest)
                else:
                    proba = estimators[c][i].predict_proba(xtest)[:, 1]
                fprs[c], tprs[c], threshold = roc_curve(ytest, proba)
                Auc = auc(fprs[c], tprs[c])
                AUC.append(Auc)
            AUCs[i] = AUC
        elif type == 'multiple':
            for c in range(len(estimators)): #6
                xtest = data[f_names[c]].iloc[test_index[i]]
                ytest = label[test_index[i]]
                if c==1 :
                    proba = estimators[c][i].decision_function(xtest)
                    y = label_binarize(ytest, classes=np.unique(ytest))
                    fpr,tpr,Auc = macro_roc(estimators[c][i],xtest,y,proba,len(np.unique(label)))
                else:
                    proba = estimators[c][i].predict_proba(xtest)
                    Auc = roc_auc_score(ytest,proba,multi_class="ovr",average="macro")
                AUC.append(Auc)
        AUCs[i] = AUC
    df_AUCs = pd.DataFrame(AUCs,index=title).T
    df_AUCs_describe = df_AUCs.describe().loc[("mean",'min','max','std'),:]
    return df_AUCs,df_AUCs_describe

def classif_report(predict,label,test_index,evaluation_index):
    reports = dict()
    final_reports = list()
    for t in range(len(predict[0])):  # 50
        for i in range(len(predict)):  # 6
            ytest = label[test_index[t]]
            report = classification_report(ytest, predict[i][t], output_dict=True)
            reports[i] = pd.DataFrame.from_dict(report).iloc[:-1, 1]
        final_report = pd.DataFrame.from_dict(reports).T.loc[:, evaluation_index]
        final_reports.append(final_report)
    final_reports = pd.concat(final_reports, axis=1).T
    final_reports.columns = title
    f_describe = final_reports.describe().loc[("mean", 'min', 'max', 'std'), :]
    return final_reports, f_describe

def multi_classif_report(predict,label,test_index,evaluation_index):
    reports = {}
    final_reports = []
    for t in range(len(predict[0])):  #50
        for i in range(len(predict)): #6
            ytest = label[test_index[t]]
            report = classification_report(ytest,predict[i][t],output_dict=True)
            reports[i] = pd.DataFrame.from_dict(report).iloc[:-1,:]['macro avg']
        final_report = pd.DataFrame.from_dict(reports).T.loc[:,evaluation_index]
        final_reports.append(final_report)
    final_reports = pd.concat(final_reports,axis=1).T
    final_reports.columns = title
    f_describe = final_reports.describe().loc[("mean",'min','max','std'),:]
    return final_reports,f_describe

def get_TopK_ROC_info(estimators, data, label, f_names, test_index, df_AUC, title=title):
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    auc_mean_std = pd.DataFrame()
    for j in range(len(estimators)):  # 6分类器
        tprs = []
        for i in range(len(estimators[0])):  # 50重复次数
            xtest = data[f_names[j][i]].iloc[test_index[i]]
            ytest = label[test_index[i]]
            proba = estimators[j][i].predict_proba(xtest)[:, 1]
            fpr, tpr, threshold = roc_curve(ytest, proba)
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    #对曲线进行插值，因为每个曲线的样本不一样，所以获取到的fpr和tpr也不一样长度，所以需要进行插值
    #插值原理，获取所有fpr的值，然后将每个交叉验证的roc都插值成和fpr的值一样多的长度。并不会改变每个roc曲线的形状
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_tpr)
        std_auc = np.std(df_AUC[title[j]])
        mean_TPR_df[title[j]] = mean_tpr
        auc_mean_std[title[j]] = [mean_auc, std_auc]
    auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, auc_mean_std

def get_FSS_BSS_ROC_info(estimators, data, label, f_names, test_index, df_AUC, title=title):
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    auc_mean_std = pd.DataFrame()
    for j in range(len(estimators)):  # 6分类器
        tprs = []
        for i in range(len(estimators[0])):  # 50重复次数
            xtest = data[f_names[j]].iloc[test_index[i]]
            ytest = label[test_index[i]]
            if j==1 :
                proba = estimators[j][i].decision_function(xtest)
            else:
                proba = estimators[j][i].predict_proba(xtest)[:,1]
            fpr, tpr, threshold = roc_curve(ytest, proba)
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    #对曲线进行插值，因为每个曲线的样本不一样，所以获取到的fpr和tpr也不一样长度，所以需要进行插值
    #插值原理，获取所有fpr的值，然后将每个交叉验证的roc都插值成和fpr的值一样多的长度。并不会改变每个roc曲线的形状
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_tpr)
        std_auc = np.std(df_AUC[title[j]])
        mean_TPR_df[title[j]] = mean_tpr
        auc_mean_std[title[j]] = [mean_auc, std_auc]
    auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, auc_mean_std

def macro_roc(estimator,xtest,ytest,proba,n_classes):
    # proba = estimator.predict_proba(xtest)
    mean_fpr = np.linspace(0, 1, 100)
    fpr,tpr,roc_auc = {},{},{}
    #macro_fpr,macro_tpr,macro_roc_auc = [],[],[]
    #n_classes = len(np.unique(ml4label2))
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
    return macro_fpr,macro_tpr,macro_roc_auc

def multi_label_get_TopK_ROC_info(estimators, data, label, f_names, test_index, df_AUC, df_AUCs_describe, title=title):
    mean_tpr, mean_auc = {}, {}
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    auc_mean_std = pd.DataFrame()

    for j in range(len(estimators)):  # 6分类器
        tprs = []
        for i in range(len(estimators[0])):  # 50重复次数
            xtest = data[f_names[j][i]].iloc[test_index[i]]
            ytest = label[test_index[i]]
            y = label_binarize(ytest, classes=np.unique(ytest))
            fpr, tpr, roc_auc = macro_roc(estimators[j][i], xtest, y, len(np.unique(label)))
            #         macro求均值（插值法）
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    #对曲线进行插值，因为每个曲线的样本不一样，所以获取到的fpr和tpr也不一样长度，所以需要进行插值
    #插值原理，获取所有fpr的值，然后将每个交叉验证的roc都插值成和fpr的值一样多的长度。并不会改变每个roc曲线的形状
        mean_tpr[j] = np.mean(tprs, axis=0)
        mean_tpr[j][-1] = 1.0
        mean_auc[j] = df_AUCs_describe.iloc[0, j]
        std_auc = np.std(df_AUC[title[j]])
        mean_TPR_df[title[j]] = mean_tpr[j]
        auc_mean_std[title[j]] = [mean_auc[j], std_auc]
    auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, auc_mean_std

def multi_label_get_FSS_BSS_ROC_info(estimators,data,label,test_index,df_AUCs,f_names,df_AUCs_describe,title=title):
    mean_tpr, mean_auc = {}, {}
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR_df = pd.DataFrame()
    auc_mean_std = pd.DataFrame()

    for j in range(len(estimators)):
        fprs, tprs, roc_aucs = [], [], []
        for i in range(len(estimators[0])):
            xtest = data[f_names[j]].iloc[test_index[i]]
            ytest = label[test_index[i]]
            y = label_binarize(ytest, classes=np.unique(ytest))
            if j == 1:
                proba = estimators[j][i].decision_function(xtest)
            else:
                proba = estimators[j][i].predict_proba(xtest)
            fpr, tpr, roc_auc = macro_roc(estimators[j][i], xtest, y, proba, len(np.unique(label)))
            #         macro求均值（插值法）
            interp_tpr = np.interp(mean_FPR, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr[j] = np.mean(tprs, axis=0)
        mean_tpr[j][-1] = 1.0
        mean_auc[j] = df_AUCs_describe.iloc[0, j]
        std_auc = np.std(df_AUCs[title[j]])
        mean_TPR_df[title[j]] = mean_tpr[j]
        auc_mean_std[title[j]] = [mean_auc[j], std_auc]
    auc_mean_std.index = ['mean_auc', 'std_auc']
    return mean_FPR, mean_TPR_df, auc_mean_std
#FSS方法
def FSS_fun(feature_names,clf,cv,data,label):
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

def BSS_fun(feature_names,clf,cv,data,label):
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
        print(i)
    selected_feature.reverse() #反向排序
    max_scores.reverse()
    return selected_feature,max_scores

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

def df2bp(df):
    data = []
    for col in df.columns:
        trace = {
            'type': 'box',
            'name': col,
            'y': df[col].to_list()
        }
        data.append(trace)
    return data

def mkroc(mean_FPR, mean_TPR_df, auc_mean_std, title=title):
    data = []
    chance = {
        'line': {
            'dash': 'dash',
            'color': 'red'
        },
        'name': 'Chance',
        'mode': 'lines',
        'type': 'scatter',
        'x': [0, 1],
        'y': [0, 1],
        'showlegend': False
    }
    data.append(chance)
    if auc_mean_std.shape != ():
        for t in title:
            trace = {
                'mode': 'lines',
                'name': 'Mean ROC of {}(AUC=%0.2f ± %0.3f)'.format(t) %(auc_mean_std[t]['mean_auc'], auc_mean_std[t]['std_auc']),
                'type': 'scatter',
                'x': list(mean_FPR),
                'y': list(mean_TPR_df[t])
            }
            data.append(trace)
    else:
        for t in title:
            trace = {
                'mode': 'lines',
                'name': r"ROC of {:}(AUC={:})".format(t,np.round(auc_mean_std,3)),
                'type': 'scatter',
                'x': list(mean_FPR),
                'y': list(mean_TPR_df[t])
            }
            data.append(trace)
    return data


def label_data_split(label, data):
    blind_label = label.loc[pd.isna(label.iloc[:,0]),:]
    train_label = label.loc[~ pd.isna(label.iloc[:,0]),:].loc[pd.isna(label.iloc[:,1]),:]
    validation_label = label.loc[~ pd.isna(label.iloc[:,0]),:].loc[label.iloc[:,1].str.lower()=='validation',:]
    blind_data = data.loc[list(blind_label.index),:]
    train_data = data.loc[list(train_label.index),:]
    validation_data = data.loc[list(validation_label.index),:]
    return blind_label, train_label, validation_label, blind_data, train_data, validation_data

def mkradar(mean_method_model):
    names = list(mean_method_model.columns)
    if 'reports' in names:
        names.remove('reports')
    data = []
    for n in names:
        trace = {
            'fill': 'tonext',
            'name': n,
            'r': mean_method_model[n].tolist(),
            'type': 'scatterpolar',
            'theta': ['Test Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
        }
        data.append(trace)
    return data