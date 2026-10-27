"""XGBoost label, probability and signed-pipeline regression checks."""
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from .model_registry import build_estimator
from .safe_ml import (
    FoldImputer, build_classification_pipeline,
    build_cross_platform_classification_pipeline, classification_metrics,
    classification_prediction, load_signed_model_bundle,
    nested_cv_classification, save_signed_model_bundle,
)


class XGBoostLabelTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.RandomState(73)
        self.X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list('abcd'))
        self.binary = np.where(self.X['a'] > 0, 'case', 'control')
        self.multiclass = np.asarray(['alpha', 'beta', 'gamma'])[np.arange(60) % 3]

    def estimator(self):
        return XGBClassifier(n_estimators=6, max_depth=2, n_jobs=1,
                             random_state=10, eval_metric='logloss')

    def test_binary_labels_and_positive_first_probability_match_native_encoding(self):
        model = build_classification_pipeline(self.estimator(), feature_method='none', scaler='none')
        model.fit(self.X, self.binary)
        reference = self.estimator().fit(self.X, (self.binary == 'control').astype(int))
        np.testing.assert_array_equal(model.classes_, ['case', 'control'])
        np.testing.assert_array_equal(model.predict(self.X), np.asarray(['case', 'control'])[reference.predict(self.X)])
        np.testing.assert_allclose(model.predict_proba(self.X), reference.predict_proba(self.X), rtol=0, atol=0)
        predicted, scores, kind = classification_prediction(model, self.X, 'case', 'explicit_probability_threshold', .4)
        np.testing.assert_array_equal(predicted, np.where(scores[:, 0] >= .4, 'case', 'control'))
        metrics = classification_metrics(self.binary, predicted, scores, model.classes_, 'case', kind)
        self.assertAlmostEqual(metrics['roc_auc'], roc_auc_score(self.binary == 'case', scores[:, 0]))

    def test_registry_multiclass_text_and_noncontiguous_integer_grid_search(self):
        for labels in [self.multiclass, np.asarray([2, 5, 9])[np.arange(60) % 3]]:
            with self.subTest(labels=str(labels.dtype)):
                estimator = build_estimator('classification', 'xgboost')[1]
                estimator.set_params(n_estimators=4, n_jobs=1, max_depth=2, eval_metric='mlogloss')
                model = GridSearchCV(build_classification_pipeline(estimator, feature_method='none'),
                                     {'model__max_depth': [1, 2]}, cv=2, scoring='balanced_accuracy', error_score='raise').fit(self.X, labels)
                np.testing.assert_array_equal(model.classes_, np.unique(labels))
                self.assertTrue(set(model.predict(self.X)).issubset(set(labels)))
                self.assertEqual(model.predict_proba(self.X).shape, (60, 3))
                np.testing.assert_allclose(model.predict_proba(self.X).sum(axis=1), 1, atol=1e-6)
                self.assertEqual(clone(model.best_estimator_).get_params()['model__n_estimators'], 4)

    def test_signed_bundle_preserves_mapping_and_predict_never_refits(self):
        X = self.X.copy(); X.iloc[0, 0] = np.nan
        model = build_classification_pipeline(self.estimator(), feature_method='none').fit(X, self.binary)
        expected = model.predict(X); probabilities = model.predict_proba(X)
        statistics = model.named_steps['imputer'].statistics_.copy()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'synthetic.maler')
            save_signed_model_bundle(path, model, 'classification', X.columns, 'test-key',
                                     metadata={'class_order': list(model.classes_)})
            restored, manifest = load_signed_model_bundle(path, 'test-key')
            with patch.object(FoldImputer, 'fit', side_effect=AssertionError('Prediction must not fit')):
                np.testing.assert_array_equal(restored.predict(X), expected)
                np.testing.assert_array_equal(restored.predict_proba(X), probabilities)
            np.testing.assert_array_equal(restored.named_steps['imputer'].statistics_, statistics)
            self.assertEqual(manifest['metadata']['class_order'], ['case', 'control'])

    def test_nested_cv_accepts_text_labels_with_explicit_positive_threshold(self):
        result = nested_cv_classification(self.X, self.binary, self.estimator(),
            param_grid={'model__max_depth': [1, 2]}, k_values=(2,), outer_splits=3,
            outer_repeats=1, inner_splits=2, positive_class='case',
            threshold_policy='explicit_probability_threshold', threshold=.4)
        self.assertEqual(len(result['folds']), 3)
        self.assertTrue(all(np.isfinite(row['balanced_accuracy']) for row in result['folds']))

    def test_feature_selection_and_cross_platform_use_same_label_contract(self):
        for direction in ['fss', 'bss']:
            model = build_classification_pipeline(self.estimator(), k=2, feature_method=direction)
            model.fit(self.X, self.binary)
            np.testing.assert_array_equal(model.classes_, ['case', 'control'])
            self.assertEqual(model.named_steps['selector'].get_support().sum(), 2)
        model = build_cross_platform_classification_pipeline(self.estimator(), k=2).fit(self.X, self.binary)
        self.assertTrue(set(model.predict(self.X)).issubset({'case', 'control'}))

    def test_eval_set_does_not_add_unknown_classes_and_refit_replaces_mapping(self):
        model = build_estimator('classification', 'xgboost')[1]
        model.set_params(n_estimators=4, n_jobs=1, max_depth=2, eval_metric='logloss')
        model.fit(self.X, self.binary, eval_set=[(self.X, self.binary)], verbose=False)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.binary, eval_set=[(self.X.iloc[:1], ['unseen'])], verbose=False)
        other = np.where(self.binary == 'case', 'yes', 'no')
        model.fit(self.X, other)
        np.testing.assert_array_equal(model.classes_, ['no', 'yes'])
        self.assertTrue(set(model.predict(self.X)).issubset({'no', 'yes'}))

    def test_output_margin_is_not_decoded_as_class_labels(self):
        model = build_estimator('classification', 'xgboost')[1]
        model.set_params(n_estimators=4, max_depth=2, n_jobs=1, eval_metric='logloss')
        model.fit(self.X, self.binary)
        reference = XGBClassifier(**model.get_params()).fit(self.X, (self.binary == 'control').astype(int))
        np.testing.assert_array_equal(model.predict(self.X, output_margin=True), reference.predict(self.X, output_margin=True))
