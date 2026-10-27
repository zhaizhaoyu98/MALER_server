"""Regression tests for demonstrated reviewer-audit gaps."""
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from django.template.loader import render_to_string
from django.test import Client, SimpleTestCase, override_settings
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

from .cv_policy import make_splits
from .safe_ml import (FoldImputer, classification_prediction, classification_scorer,
                      load_signed_model_bundle, nested_cv_classification,
                      nested_cv_regression, build_classification_pipeline)
from .safe_survival import nested_cv_survival
from .source_version import source_identity
from .validated_analysis import run_validated_analysis


def synthetic_project(directory, task='classification', grouped=False, multiclass=False):
    rng = np.random.RandomState(19)
    n = 76
    X = rng.normal(size=(n, 6))
    y = np.repeat(np.arange(38) % (3 if multiclass else 2), 2)
    X[:, 0] += y
    frame = pd.DataFrame(X.T, index=['g%d' % i for i in range(6)],
                         columns=['sample%02d' % i for i in range(n)])
    if task == 'survival':
        frame.loc['status'] = y
        frame.loc['time'] = np.exp(2 + .2*X[:, 1] + rng.normal(scale=.3, size=n))
        meta = ['status', 'time', 'set']
    else:
        frame.loc['label'] = (1.3*X[:, 0] - .4*X[:, 1] if task == 'regression'
                              else ['case' if v else 'control' for v in y] if not multiclass
                              else ['class%d' % v for v in y])
        meta = ['label', 'set']
    frame.loc['set'] = ['training']*72 + ['testing']*4
    if grouped:
        frame.loc['subject_id'] = np.repeat(['person%d' % i for i in range(38)], 2)
        meta.append('subject_id')
    frame.loc[meta+['g%d' % i for i in range(6)]].to_csv(os.path.join(directory, 'data.csv'))


@override_settings(MALER_WEB_OUTER_SPLITS=2, MALER_WEB_OUTER_REPEATS=1,
                   MALER_WEB_INNER_SPLITS=2, MALER_WEB_N_JOBS=1,
                   MALER_MODEL_SIGNING_KEY='synthetic-test-key',
                   MALER_WEB_MODELS={'classification':['logistic_regression'],
                                     'regression':['ridge'], 'survival':['survival_tree']})
class SubmissionGapTests(SimpleTestCase):
    def test_prediction_preview_histogram_ignores_missing_cells_without_imputing(self):
        from .views.preview_views import data_hist
        frame=pd.DataFrame({'set':['blind','blind'],'g1':[np.nan,2.],'g2':[1.,np.nan]})
        before=frame.copy(deep=True)
        density,edges,centers=data_hist(frame)
        self.assertTrue(np.isfinite(density).all())
        self.assertEqual((edges[0],edges[-1]),(1.,2.))
        pd.testing.assert_frame_equal(frame,before)

    def test_grouped_sequential_paths_are_rejected_before_fitting_all_tasks(self):
        X = pd.DataFrame(np.random.RandomState(1).normal(size=(24,4)))
        y = np.repeat(np.arange(12) % 2, 2)
        groups = np.repeat(np.arange(12), 2)
        for method in ['fss','bss']:
            for task in ['classification','multiclass','regression','survival']:
                with self.subTest(task=task, method=method), tempfile.TemporaryDirectory() as directory:
                    actual_task = 'classification' if task == 'multiclass' else task
                    synthetic_project(directory, actual_task, grouped=True, multiclass=task=='multiclass')
                    prefix = {'classification':'BCO','multiclass':'MCO','regression':'RO','survival':'SO'}[task]
                    with self.assertRaisesRegex(ValueError, 'Grouped analyses do not support'):
                        run_validated_analysis(directory, prefix+'-test',
                                               {'feature_select_method':method,'positive_class':'case'})
        with self.assertRaisesRegex(ValueError, 'not support'):
            nested_cv_classification(X,y,LogisticRegression(),groups=groups,feature_method='fss')
        with self.assertRaisesRegex(ValueError, 'not support'):
            nested_cv_regression(X,y,Ridge(),groups=groups,feature_method='bss')
        with self.assertRaisesRegex(ValueError, 'not support'):
            nested_cv_survival(X,y,None,groups=groups,feature_method='fss')

    def test_supported_grouped_pipeline_completes_four_tasks(self):
        for task in ['classification','multiclass','regression','survival']:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                actual = 'classification' if task == 'multiclass' else task
                synthetic_project(directory,actual,grouped=True,multiclass=task=='multiclass')
                prefix = {'classification':'BCO','multiclass':'MCO','regression':'RO','survival':'SO'}[task]
                result = run_validated_analysis(directory,prefix+'-test',
                           {'feature_select_method':'TopK','positive_class':'case','feature_norm':'Z'})
                self.assertEqual(result['dataset']['split_level'],'subject')
                self.assertEqual(result['selection']['validation_budget']['actual_outer_evaluations'],2)
                self.assertIsNotNone(result['models'][0]['heldout_test'])

    def test_group_boundaries_all_resampling_levels_four_tasks(self):
        X = np.zeros((72,2))
        groups = np.repeat(np.arange(36),2)
        for task in ['binary','multiclass','regression','survival']:
            labels = np.repeat(np.arange(36) % (3 if task=='multiclass' else 2),2)
            actual = 'classification' if task in ['binary','multiclass'] else task
            y = np.array(list(zip(labels.astype(bool),np.arange(72)+1)),
                         dtype=[('event',bool),('time',float)]) if task=='survival' else labels
            outer,_ = make_splits(X,y,actual,2,2,groups=groups)
            for train,test in outer:
                self.assertFalse(set(groups[train]) & set(groups[test]))
                inner,_ = make_splits(X[train],y[train],actual,2,groups=groups[train],inner=True)
                for fit,valid in inner:
                    self.assertFalse(set(groups[train][fit]) & set(groups[train][valid]))

    def test_fixed_cutoff_scorer_matches_decisions_for_both_positive_labels(self):
        rng=np.random.RandomState(6)
        X=rng.normal(size=(50,3))
        y=np.where(X[:,0]+.4*X[:,1]>0,'case','control')
        model=LogisticRegression().fit(X,y)
        for positive in ['case','control']:
            for cutoff in [0,.5,1]:
                with self.subTest(positive=positive,cutoff=cutoff):
                    scorer=classification_scorer(positive,'explicit_probability_threshold',cutoff)
                    decision,_,_=classification_prediction(model,X,positive,'explicit_probability_threshold',cutoff)
                    self.assertAlmostEqual(scorer(model,X,y),balanced_accuracy_score(y,decision))

    def test_zero_threshold_survives_full_lock_export_and_render(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_project(directory)
            result=run_validated_analysis(directory,'BCO-test',
                    {'positive_class':'case','threshold_policy':'explicit_probability_threshold',
                     'threshold_value':0,'feature_select_method':'TopK','feature_norm':'Z'})
            self.assertEqual(result['selection']['threshold_value'],0)
            model=result['models'][0]
            loaded,manifest=load_signed_model_bundle(os.path.join(directory,model['model_bundle']),
                                                     'synthetic-test-key')
            self.assertEqual(manifest['metadata']['threshold_value'],0)
            html=render_to_string('validated_result.html',{'result':result})
            self.assertIn('explicit_probability_threshold (0.0)',html)
            prediction=pd.read_csv(os.path.join(directory,model['prediction_files']['testing']))
            self.assertTrue(prediction['predicted'].eq('case').all())

    def test_invalid_cutoffs_fail_before_search(self):
        for cutoff in [-.1,1.1,float('nan'),float('inf')]:
            with self.subTest(cutoff=cutoff), tempfile.TemporaryDirectory() as directory:
                synthetic_project(directory)
                with self.assertRaisesRegex(ValueError,'finite value'):
                    run_validated_analysis(directory,'BCO-test',
                        {'positive_class':'case','threshold_policy':'explicit_probability_threshold',
                         'threshold_value':cutoff})

    def test_complete_lock_uses_actual_file_hash_and_only_selected_final_refit(self):
        from .validated_analysis import fit_final_model as actual_fit
        with tempfile.TemporaryDirectory() as directory:
            synthetic_project(directory,'regression')
            with override_settings(MALER_WEB_MODELS={'regression':['ridge','linear_regression']}), \
                 patch('mlserver.validated_analysis.fit_final_model',wraps=actual_fit) as final_fit:
                result=run_validated_analysis(directory,'RO-test',{'feature_select_method':'TopK'})
            self.assertEqual(final_fit.call_count,1)
            lock_bytes=open(os.path.join(directory,'selection_lock.json'),'rb').read()
            lock=json.loads(lock_bytes.decode())
            chosen=next(x for x in result['models'] if x['key']==result['selected_family'])
            self.assertEqual(lock['best_params'],chosen['best_params'])
            self.assertEqual(lock['selected_features'],chosen['selected_features'])
            self.assertEqual(lock['candidate_counts'][chosen['key']],len(ParameterGrid(chosen['candidate_grid'])))
            self.assertEqual(lock['fitted_model_sha256'],hashlib.sha256(open(os.path.join(directory,'selected_pipeline.joblib'),'rb').read()).hexdigest())
            _,manifest=load_signed_model_bundle(os.path.join(directory,chosen['model_bundle']),'synthetic-test-key')
            self.assertEqual(manifest['metadata']['selection_lock_sha256'],hashlib.sha256(lock_bytes).hexdigest())
            self.assertLessEqual(lock['family_selected_at'],lock['selected_at'])
            self.assertLessEqual(lock['selected_at'],lock['heldout_unlocked_at'])
            before=open(os.path.join(directory,'selected_pipeline.joblib'),'rb').read()
            with self.assertRaisesRegex(ValueError,'already has a selection lock'):
                run_validated_analysis(directory,'RO-test',{})
            self.assertEqual(before,open(os.path.join(directory,'selected_pipeline.joblib'),'rb').read())

    def test_concurrent_analysis_lease_prevents_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory,'.analysis_running'),'w').close()
            with self.assertRaisesRegex(ValueError,'already running'):
                run_validated_analysis(directory,'RO-test',{})

    def test_missing_entire_training_fold_column_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError,'training fold has all-missing'):
            FoldImputer(strategy='median').fit([[np.nan,1],[np.nan,2]])
        imputer=FoldImputer(strategy='median').fit([[1,1],[3,2],[np.nan,3]])
        np.testing.assert_allclose(imputer.transform([[np.nan,4]]),[[2,4]])

    def test_native_and_explicit_policy_are_distinct_grid_scorers(self):
        X=pd.DataFrame(np.random.RandomState(4).normal(size=(30,3)))
        y=np.where(X[0]>0,'case','control')
        pipe=build_classification_pipeline(LogisticRegression(),k=2)
        scorer=classification_scorer('case','explicit_probability_threshold',.8)
        search=GridSearchCV(pipe,{'model__C':[.1,1]},scoring=scorer,cv=3).fit(X,y)
        decisions,_,_=classification_prediction(search,X,'case','explicit_probability_threshold',.8)
        self.assertAlmostEqual(search.score(X,y),balanced_accuracy_score(y,decisions))

    def test_public_state_policy_and_controls_match_effective_settings(self):
        for enabled in [False, True]:
            with override_settings(MALER_PUBLIC_UPLOADS_ENABLED=enabled,
                                   MALER_CACHE_RETENTION_HOURS=12,
                                   MALER_MAX_DATA_UPLOAD_BYTES=8*1024*1024):
                for route in ['/maler/home','/maler/analysis','/maler/predict','/maler/help']:
                    response=Client().get(route)
                    self.assertEqual(response.status_code,200)
                    html=response.content.decode()
                    self.assertIn('Public research uploads are available' if enabled
                                  else 'Public uploads are temporarily paused',html)
                    if enabled:
                        self.assertNotIn('User uploads remain paused',html)
                        self.assertNotIn('Public uploads are temporarily paused',html)
                    if route in ['/maler/analysis','/maler/predict']:
                        self.assertIn('12 hours',html)
                        self.assertIn('Data upload limit: 8 MiB' if route=='/maler/analysis'
                                      else 'Data/model limits: 8/100 MiB',html)
                        self.assertEqual('disabled aria-disabled="true"' in html, not enabled)
                    if route in ['/maler/analysis','/maler/help']:
                        self.assertIn('5 outer folds × 2 repeats',html)
                        self.assertIn('5 outer folds × 10 repeats',html)
                self.assertIn('Data Handling and Retention',html)
                self.assertIn('retention remains unverified',html)

    def test_declared_version_alone_is_not_release_verification(self):
        with override_settings(MALER_SOFTWARE_COMMIT='incorrect',MALER_REQUIRE_VERIFIED_RELEASE=True):
            with self.assertRaisesRegex(ValueError,'clean, verifiable source'):
                source_identity()
