"""Actual signed-Predict regressions discovered in the comprehensive R2 audit."""
import os
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import Client, SimpleTestCase, override_settings
from sklearn.linear_model import LogisticRegression, Ridge
from sksurv.tree import SurvivalTree

from .safe_ml import (DataValidationError, FoldImputer, align_prediction_frame,
                      build_classification_pipeline, build_regression_pipeline,
                      save_signed_model_bundle)
from .safe_survival import build_survival_pipeline, validate_survival_target
from .task_access import issue_task_token
from .views.predict_views import _bundle_as_legacy_dict, pred_val_split


class SignedPredictAuditTests(SimpleTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy = override_settings(MALER_TASK_ROOT=self.tmp.name,
                                        MALER_MODEL_SIGNING_KEY='audit-synthetic-key')
        self.policy.enable()
        rng = np.random.RandomState(412)
        self.X = pd.DataFrame(rng.normal(size=(40, 4)), columns=['g0', 'g1', 'g2', 'g3'])
        self.X.index = ['sample%d' % n for n in range(40)]

    def tearDown(self):
        self.policy.disable()
        self.tmp.cleanup()

    def project(self, model, method, data, positive=None, cutoff=None, sep=','):
        task = {'model_bclass': 'classification', 'model_mclass': 'classification',
                'model_reg': 'regression', 'model_sur': 'survival'}[method]
        project = 'PRED-%s-audittest' % method.replace('model_', '').upper()
        folder = os.path.join(self.tmp.name, project)
        os.mkdir(folder)
        metadata = {'method': method, 'model_name': 'audit_model',
                    'positive_class': None if positive is None else str(positive),
                    'threshold_policy': ('estimator_native_predict' if cutoff is None
                                         else 'explicit_probability_threshold'),
                    'threshold_value': cutoff}
        bundle = os.path.join(folder, 'model.maler')
        save_signed_model_bundle(bundle, model, task, self.X.columns,
                                 'audit-synthetic-key', metadata=metadata)
        data.T.to_csv(os.path.join(folder, 'data.csv'), sep=sep)
        token = issue_task_token(project)
        with patch.object(FoldImputer, 'fit', side_effect=AssertionError('Predict refitted')):
            response = Client().get('/maler/predict_result/%s?token=%s' % (project, token))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Delete this prediction project')
        self.assertContains(response, 'name="access_token" value="%s"' % token)
        with open(os.path.join(folder, 'predict_pickle.pkl'), 'rb') as stream:
            result = pickle.load(stream)
        self.assertNotIn('access_token', result)
        return result, response, bundle

    def test_zero_cutoff_is_preserved_through_actual_signed_predict(self):
        y = np.where(self.X.g0 > 0, 'case', 'control')
        model = build_classification_pipeline(LogisticRegression(), k=3).fit(self.X, y)
        result, response, bundle = self.project(model, 'model_bclass', self.X,
                                                positive='case', cutoff=0)
        self.assertEqual(_bundle_as_legacy_dict(bundle)['threshold_value'], 0)
        self.assertEqual(result['threshold_value'], 0)
        self.assertTrue(all(r['Label'] == 'case' for r in result['predict_reports_dict']))

    def test_numeric_outcome_and_missing_first_label_are_recognized(self):
        frame = self.X.copy()
        frame.insert(0, 'label', np.where(self.X.g0 > 0, 1, 0).astype(float))
        frame.iloc[0, 0] = np.nan
        blind, validation = pred_val_split(frame, datatype='model_bclass')
        self.assertEqual(len(blind), 1)
        self.assertEqual(len(validation), 39)
        self.assertNotIn('label', blind.columns)

    def test_nonnumeric_prediction_cells_are_rejected_not_imputed(self):
        frame = self.X.copy()
        frame.loc['sample0', 'g0'] = 'invalid numeric cell'
        with self.assertRaisesRegex(DataValidationError, 'Non-numeric'):
            align_prediction_frame(frame, self.X.columns)

    def test_semicolon_matrix_is_reused_through_actual_signed_predict(self):
        model = build_regression_pipeline(Ridge(), k=3).fit(self.X, self.X.g0 * 2)
        result, _, _ = self.project(model, 'model_reg', self.X, sep=';')
        np.testing.assert_allclose([r['Label'] for r in result['predict_reports_dict']],
                                   model.predict(self.X))

    def test_survival_signed_pipeline_blind_predictions_and_curves(self):
        y = validate_survival_target(np.tile([0, 1], 20),
                                     np.exp(2 - self.X.g0.to_numpy()))
        model = build_survival_pipeline(SurvivalTree(min_samples_leaf=4), k=3).fit(self.X, y)
        result, _, _ = self.project(model, 'model_sur', self.X)
        np.testing.assert_allclose([r['Label'] for r in result['predict_reports_dict']],
                                   model.predict(self.X))
        self.assertTrue(result['surv_plot'])

    def test_survival_validation_keeps_rows_and_reports_censor_reference_limit(self):
        y = validate_survival_target(np.tile([0, 1], 20),
                                     np.exp(2 - self.X.g0.to_numpy()))
        model = build_survival_pipeline(SurvivalTree(min_samples_leaf=4), k=3).fit(self.X, y)
        frame = self.X.copy()
        frame.insert(0, 'time', y['time'])
        frame.insert(0, 'status', y['event'].astype(int))
        result, response, _ = self.project(model, 'model_sur', frame)
        self.assertEqual(len(result['validation_reports_dict']), 40)
        self.assertIn('c_index', result['signed_validation_metrics'])
        self.assertContains(response, 'censoring reference')

    def test_multiclass_signed_validation_reports_macro_metrics_without_refit(self):
        y = np.tile(['class0', 'class1', 'class2', 'class1'], 10)
        model = build_classification_pipeline(LogisticRegression(), k=3).fit(self.X, y)
        frame = self.X.copy()
        frame.insert(0, 'label', y)
        result, _, _ = self.project(model, 'model_mclass', frame)
        self.assertEqual(len(result['validation_reports_dict']), 40)
        self.assertIn('roc_auc_ovr_macro', result['signed_validation_metrics'])
        self.assertIn('pr_auc_macro', result['signed_validation_metrics'])

    def test_regression_signed_validation_reports_full_metrics_without_refit(self):
        y = self.X.g0.to_numpy() * 2
        model = build_regression_pipeline(Ridge(), k=3).fit(self.X, y)
        frame = self.X.copy()
        frame.insert(0, 'label', y)
        result, _, _ = self.project(model, 'model_reg', frame)
        self.assertEqual(len(result['validation_reports_dict']), 40)
        self.assertEqual(set(result['signed_validation_metrics']), {'r2', 'mae', 'mse', 'rmse'})

    def test_numeric_signed_binary_validation_and_blind_rows(self):
        y = np.where(self.X.g0 > 0, '1', '0')
        model = build_classification_pipeline(LogisticRegression(), k=3).fit(self.X, y)
        frame = self.X.copy()
        frame.insert(0, 'label', y.astype(float))
        frame.iloc[0, 0] = np.nan
        result, _, _ = self.project(model, 'model_bclass', frame, positive='1', cutoff=0.65)
        self.assertEqual(len(result['predict_reports_dict']), 1)
        self.assertEqual(len(result['validation_reports_dict']), 39)
        self.assertEqual(result['positive_class'], '1')
