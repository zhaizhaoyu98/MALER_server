from __future__ import absolute_import

import os
import shutil
import json
import pickle
import tempfile
import time
import unittest
import zipfile
from io import StringIO

import numpy as np
import pandas as pd
from django.http import Http404
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import SuspiciousOperation
from django.core.management import call_command
from django.test import Client, override_settings
from sklearn.linear_model import LogisticRegression, Ridge
from sksurv.tree import SurvivalTree

from .safe_ml import (
    DataValidationError,
    ModelBundleError,
    align_prediction_frame,
    alignment_report,
    build_classification_pipeline,
    build_cross_platform_classification_pipeline,
    build_regression_pipeline,
    classification_metrics,
    load_signed_model_bundle,
    nested_cv_classification,
    selected_feature_names,
    regression_metrics,
    save_signed_model_bundle,
    validate_feature_matrix,
)
from .safe_survival import (
    build_survival_pipeline,
    nested_cv_survival,
    validate_survival_target,
)
from .model_registry import FEATURE_REDUCTION, MODEL_REGISTRY, build_estimator
from .views.download_views import download_model, download_sample_data
from .security import validated_project_id
from .task_access import issue_task_token
from .views import predict_result as predict_result_view


class SafeMLValidationTests(unittest.TestCase):
    def test_rejects_duplicate_features(self):
        frame = pd.DataFrame([[1.0, 2.0], [2.0, 3.0]], columns=["A", "A"])
        with self.assertRaises(DataValidationError):
            validate_feature_matrix(frame)

    def test_rejects_non_numeric_values(self):
        frame = pd.DataFrame({"A": [1.0, "not-a-number"], "B": [2.0, 3.0]})
        with self.assertRaises(DataValidationError):
            validate_feature_matrix(frame)

    def test_prediction_features_are_matched_by_identifier(self):
        frame = pd.DataFrame({"B": [2.0], "extra": [9.0], "A": [1.0]})
        aligned, unexpected = align_prediction_frame(frame, ["A", "B"])
        self.assertEqual(aligned.columns.tolist(), ["A", "B"])
        self.assertEqual(unexpected, ["extra"])

    def test_alignment_report_is_silent_when_order_matches(self):
        frame = pd.DataFrame({"A": [1.0], "B": [2.0]})
        self.assertIsNone(alignment_report(frame, ["A", "B"]))

    def test_alignment_report_detects_reordered_features(self):
        frame = pd.DataFrame({"B": [2.0], "A": [1.0]})
        report = alignment_report(frame, ["A", "B"], cohort="blind")
        self.assertEqual(report["reordered"], ["B", "A"])
        self.assertEqual(report["unexpected"], [])
        self.assertEqual(report["n_reordered"], 2)
        self.assertEqual(report["cohort"], "blind")

    def test_alignment_report_detects_unexpected_features(self):
        frame = pd.DataFrame({"A": [1.0], "B": [2.0], "extra": [9.0]})
        report = alignment_report(frame, ["A", "B"])
        self.assertEqual(report["unexpected"], ["extra"])
        self.assertEqual(report["reordered"], [])
        self.assertEqual(report["n_unexpected"], 1)

    def test_alignment_report_combines_reordering_and_unexpected_features(self):
        frame = pd.DataFrame({"B": [2.0], "extra": [9.0], "A": [1.0]})
        report = alignment_report(frame, ["A", "B"])
        self.assertEqual(report["reordered"], ["B", "A"])
        self.assertEqual(report["unexpected"], ["extra"])

    def test_alignment_report_accepts_numeric_column_names(self):
        frame = pd.DataFrame({1: [1.0], 2: [2.0]})
        self.assertIsNone(alignment_report(frame, ["1", "2"]))

    def test_alignment_report_ignores_duplicate_columns(self):
        frame = pd.DataFrame([[1.0, 2.0]], columns=["A", "A"])
        self.assertIsNone(alignment_report(frame, ["A"]))

    def test_alignment_report_defers_missing_features_to_alignment(self):
        frame = pd.DataFrame({"A": [1.0]})
        self.assertIsNone(alignment_report(frame, ["A", "B"]))
        with self.assertRaises(DataValidationError):
            align_prediction_frame(frame, ["A", "B"])


class SafeMLPipelineTests(unittest.TestCase):
    def test_imputer_and_scaler_are_fit_inside_pipeline(self):
        X_train = pd.DataFrame({"A": [0.0, 1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0, 4.0]})
        y_train = np.array([0, 0, 1, 1])
        pipeline = build_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=10), k=1, scaler="standard"
        )
        pipeline.fit(X_train, y_train)
        self.assertAlmostEqual(pipeline.named_steps["imputer"].statistics_[0], 1.0)
        self.assertTrue(hasattr(pipeline.named_steps["scaler"], "mean_"))

    def test_nested_classification_reports_required_metrics(self):
        rng = np.random.RandomState(10)
        X = pd.DataFrame(rng.normal(size=(40, 8)), columns=["g%d" % i for i in range(8)])
        y = np.array([0] * 20 + [1] * 20)
        X.loc[y == 1, "g0"] += 3.0
        result = nested_cv_classification(
            X,
            y,
            LogisticRegression(solver="liblinear", random_state=10),
            param_grid={"model__C": [0.1, 1.0]},
            k_values=(2, 4),
            outer_splits=4,
            outer_repeats=1,
            inner_splits=2,
        )
        self.assertEqual(len(result["folds"]), 4)
        for key in ("accuracy", "balanced_accuracy", "specificity", "f1", "mcc", "roc_auc", "pr_auc"):
            self.assertIn(key, result["summary"])

    def test_regression_pipeline_and_metrics(self):
        rng = np.random.RandomState(3)
        X = pd.DataFrame(rng.normal(size=(30, 6)))
        y = 2.0 * X.iloc[:, 0].to_numpy() + rng.normal(scale=0.1, size=30)
        pipeline = build_regression_pipeline(Ridge(alpha=1.0), k=3)
        pipeline.fit(X, y)
        metrics = regression_metrics(y, pipeline.predict(X))
        self.assertGreater(metrics["r2"], 0.8)
        self.assertGreaterEqual(metrics["rmse"], 0.0)

    def test_forward_selection_is_fitted_as_a_pipeline_step(self):
        rng = np.random.RandomState(41)
        X = pd.DataFrame(rng.normal(size=(30, 6)), columns=["g%d" % i for i in range(6)])
        y = np.array([0, 1] * 15)
        X.loc[y == 1, "g0"] += 2.0
        pipeline = build_classification_pipeline(
            LogisticRegression(solver="liblinear"), k=2, feature_method="fss")
        pipeline.fit(X, y)
        self.assertEqual(len(pipeline.named_steps["selector"].get_support(indices=True)), 2)

    def test_mrmr_is_fitted_as_a_pipeline_step(self):
        rng = np.random.RandomState(43)
        X = pd.DataFrame(rng.normal(size=(30, 6)), columns=["g%d" % i for i in range(6)])
        y = np.array([0, 1] * 15)
        X.loc[y == 1, "g0"] += 2.0
        pipeline = build_classification_pipeline(
            LogisticRegression(solver="liblinear"), k=2, feature_method="mrmr")
        pipeline.fit(X, y)
        self.assertEqual(len(pipeline.named_steps["selector"].get_support(indices=True)), 2)

    def test_pca_is_fitted_inside_exportable_pipeline(self):
        rng = np.random.RandomState(17)
        X = pd.DataFrame(rng.normal(size=(30, 6)), columns=["g%d" % i for i in range(6)])
        y = np.array([0] * 15 + [1] * 15)
        X.loc[y == 1, "g0"] += 2.0
        pipeline = build_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=10),
            k=3,
            feature_method="pca",
        )
        pipeline.fit(X, y)
        self.assertEqual(pipeline.named_steps["selector"].n_components_, 3)
        self.assertEqual(selected_feature_names(pipeline, X.columns), ["PC1", "PC2", "PC3"])
        self.assertEqual(len(pipeline.predict(X)), len(X))

    def test_nested_pca_classification_runs_without_pre_fit_reduction(self):
        rng = np.random.RandomState(23)
        X = pd.DataFrame(rng.normal(size=(40, 10)), columns=["g%d" % i for i in range(10)])
        y = np.array([0] * 20 + [1] * 20)
        X.loc[y == 1, "g0"] += 2.5
        result = nested_cv_classification(
            X,
            y,
            LogisticRegression(solver="liblinear", random_state=10),
            param_grid={"model__C": [1.0]},
            k_values=(2, 4),
            outer_splits=4,
            outer_repeats=1,
            inner_splits=2,
            feature_method="pca",
        )
        self.assertEqual(len(result["folds"]), 4)
        self.assertTrue(all(
            fold["best_params"]["selector__n_components"] in (2, 4)
            for fold in result["folds"]
        ))

    def test_binary_specificity(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            scores=np.array([0.1, 0.6, 0.8, 0.9]),
            classes=np.array([0, 1]),
        )
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["recall_sensitivity"], 1.0)

    def test_cross_platform_pipeline_uses_samplewise_ranks(self):
        X = pd.DataFrame({"A": [1.0, 40.0, 2.0, 30.0], "B": [2.0, 20.0, 3.0, 10.0]})
        y = np.array([0, 0, 1, 1])
        pipeline = build_cross_platform_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=10), k=2)
        pipeline.fit(X, y)
        ranked = pipeline.named_steps["sample_rank"].transform(X)
        np.testing.assert_allclose(ranked[0], [0.5, 1.0])
        np.testing.assert_allclose(ranked[1], [1.0, 0.5])


class SignedModelBundleTests(unittest.TestCase):
    def _model(self):
        X = pd.DataFrame({"A": [0.0, 0.2, 1.0, 1.2], "B": [1.0, 0.9, 0.1, 0.0]})
        y = np.array([0, 0, 1, 1])
        model = build_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=10), k=1
        )
        model.fit(X, y)
        return X, model

    def test_signed_bundle_round_trip(self):
        X, model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.maler")
            save_signed_model_bundle(path, model, "binary", X.columns, "test-secret")
            loaded, manifest = load_signed_model_bundle(path, "test-secret")
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
            self.assertEqual(manifest["feature_names"], ["A", "B"])

    def test_tampered_bundle_is_rejected_before_deserialization(self):
        X, model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            original = os.path.join(directory, "model.maler")
            tampered = os.path.join(directory, "tampered.maler")
            save_signed_model_bundle(original, model, "binary", X.columns, "test-secret")
            with zipfile.ZipFile(original, "r") as source, zipfile.ZipFile(tampered, "w") as target:
                for name in source.namelist():
                    content = source.read(name)
                    if name == "model.joblib":
                        content += b"tampered"
                    target.writestr(name, content)
            with self.assertRaises(ModelBundleError):
                load_signed_model_bundle(tampered, "test-secret")

    def test_model_download_converts_internal_cache_to_signed_bundle(self):
        X, model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = os.path.join(directory, "cache", "BC-test")
            os.makedirs(cache_dir)
            legacy_path = os.path.join(cache_dir, "Demo.pkl")
            with open(legacy_path, "wb") as handle:
                pickle.dump({
                    "method": "model_bclass",
                    "name": "Demo",
                    "model": model,
                    "classes": {"control": 0, "case": 1},
                    "feature_names": list(X.columns),
                }, handle)
            with override_settings(STATIC_ROOT=directory, MALER_MODEL_SIGNING_KEY="test-secret"):
                response = download_model(None, "BC-test_Demo")
                self.assertIn(".maler", response["Content-Disposition"])
                response.close()
                bundle_path = os.path.join(cache_dir, "Demo.maler")
                loaded, manifest = load_signed_model_bundle(bundle_path, "test-secret")
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
            self.assertEqual(manifest["metadata"]["pipeline_complete"], False)

    def test_download_rejects_path_traversal(self):
        with self.assertRaises(Http404):
            download_sample_data(None, "../settings.py")


class SafeSurvivalTests(unittest.TestCase):
    def test_survival_status_encoding(self):
        target = validate_survival_target(
            ["dead", "alive", "1.0", "0.0"], [5.0, 8.0, 11.0, 13.0]
        )
        self.assertEqual(target.dtype.names, ("event", "time"))
        self.assertEqual(target["event"].tolist(), [True, False, True, False])

    def test_survival_rejects_non_positive_time(self):
        with self.assertRaises(DataValidationError):
            validate_survival_target([1, 0, 1], [5.0, 0.0, 2.0])

    def test_survival_preprocessing_is_fold_local_and_metrics_run(self):
        rng = np.random.RandomState(5)
        X = pd.DataFrame(rng.normal(size=(36, 6)), columns=["g%d" % i for i in range(6)])
        time = np.exp(2.0 - X["g0"].to_numpy() + rng.normal(scale=0.2, size=36))
        status = np.array(([1, 1, 0] * 12), dtype=int)
        y = validate_survival_target(status, time)
        result = nested_cv_survival(
            X,
            y,
            SurvivalTree(random_state=10, min_samples_leaf=3),
            param_grid={"model__max_depth": [2, 3]},
            k_values=(2, 3),
            outer_splits=3,
            outer_repeats=1,
            inner_splits=2,
        )
        self.assertEqual(len(result["folds"]), 3)
        self.assertIn("c_index", result["summary"])

    def test_survival_pca_runs_inside_nested_pipeline(self):
        rng = np.random.RandomState(29)
        X = pd.DataFrame(rng.normal(size=(42, 8)), columns=["g%d" % i for i in range(8)])
        time_values = np.exp(2.0 - X["g0"].to_numpy() + rng.normal(scale=0.2, size=42))
        status = np.array(([1, 1, 0] * 14), dtype=int)
        y = validate_survival_target(status, time_values)
        result = nested_cv_survival(
            X,
            y,
            SurvivalTree(random_state=10, min_samples_leaf=3),
            param_grid={"model__max_depth": [2]},
            k_values=(2, 4),
            outer_splits=3,
            outer_repeats=1,
            inner_splits=2,
            feature_method="pca",
        )
        self.assertEqual(len(result["folds"]), 3)
        self.assertTrue(all(
            fold["best_params"]["selector__n_components"] in (2, 4)
            for fold in result["folds"]
        ))

    def test_survival_forward_selection_uses_cox_prescreen(self):
        rng = np.random.RandomState(31)
        X = pd.DataFrame(rng.normal(size=(30, 6)), columns=["g%d" % i for i in range(6)])
        time_values = np.exp(2.0 - X["g0"].to_numpy() + rng.normal(scale=0.2, size=30))
        y = validate_survival_target(np.array([1, 1, 0] * 10), time_values)
        pipeline = build_survival_pipeline(
            SurvivalTree(random_state=10, min_samples_leaf=3), k=2, feature_method="fss")
        pipeline.fit(X, y)
        self.assertEqual(len(pipeline.named_steps["selector"].get_support(indices=True)), 2)


class CacheRetentionTests(unittest.TestCase):
    def test_cleanup_is_dry_run_by_default_and_apply_is_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = os.path.join(directory, "cache")
            old_path = os.path.join(cache_root, "old-project")
            fresh_path = os.path.join(cache_root, "fresh-project")
            os.makedirs(old_path)
            os.makedirs(fresh_path)
            old_time = time.time() - 48 * 3600
            os.utime(old_path, (old_time, old_time))
            with override_settings(
                    STATIC_ROOT=directory, MALER_CACHE_RETENTION_HOURS=24):
                output = StringIO()
                call_command("cleanup_maler_cache", stdout=output)
                self.assertTrue(os.path.isdir(old_path))
                self.assertIn("expired", output.getvalue())
                call_command("cleanup_maler_cache", apply_changes=True, stdout=StringIO())
            self.assertFalse(os.path.exists(old_path))
            self.assertTrue(os.path.isdir(fresh_path))


class ModelRegistryTests(unittest.TestCase):
    def test_every_supported_task_has_reproducible_model_metadata(self):
        for task in ("classification", "regression", "survival"):
            self.assertIn(task, MODEL_REGISTRY)
            self.assertTrue(MODEL_REGISTRY[task]["primary_metric"])
            for model in MODEL_REGISTRY[task]["models"].values():
                self.assertIn(".", model["class"])
                self.assertIsInstance(model["requires_scaling"], bool)
                self.assertTrue(model["grid"])

    def test_pca_is_declared_as_non_gene_feature_reduction(self):
        self.assertIn("pca", FEATURE_REDUCTION)
        self.assertIn("not gene signatures", FEATURE_REDUCTION["pca"]["description"])

    def test_registry_builds_trusted_estimators_and_aliases(self):
        key, estimator = build_estimator("classification", "randomforest")
        self.assertEqual(key, "random_forest")
        self.assertEqual(estimator.__class__.__name__, "RandomForestClassifier")


class SecurityConfigurationTests(unittest.TestCase):
    def test_project_cache_identifier_rejects_path_traversal(self):
        self.assertEqual(validated_project_id("BCO-AN-a1b2c3-TopK"), "BCO-AN-a1b2c3-TopK")
        for value in ("../secret", "..\\secret", "a/b", "", None):
            with self.assertRaises(SuspiciousOperation):
                validated_project_id(value)
        response = Client().get("/maler/predict_result/..")
        self.assertEqual(response.status_code, 400)

    def test_strict_security_audit_passes_for_production_like_settings(self):
        with override_settings(
            DEBUG=False,
            SECRET_KEY="s" * 64,
            MALER_MODEL_SIGNING_KEY="m" * 64,
            ALLOWED_HOSTS=["maler.example.org"],
            SECURE_SSL_REDIRECT=True,
            SESSION_COOKIE_SECURE=True,
            CSRF_COOKIE_SECURE=True,
            SECURE_HSTS_SECONDS=31536000,
            MALER_ALLOW_LEGACY_MODEL_UPLOAD=False,
            MALER_CACHE_RETENTION_HOURS=24,
        ):
            output = StringIO()
            call_command("audit_security_configuration", strict=True, stdout=output)
        self.assertIn("10/10 checks passed", output.getvalue())


class PredictionAlignmentNoticeTests(unittest.TestCase):
    """The predict route must report feature-order and extra-column deviations."""

    def setUp(self):
        self.client = Client()
        self.directory = tempfile.mkdtemp()
        self._original_root = predict_result_view.STATIC_ROOT
        predict_result_view.STATIC_ROOT = self.directory

    def tearDown(self):
        predict_result_view.STATIC_ROOT = self._original_root
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write_project(self, project_id, matrix):
        directory = os.path.join(self.directory, "cache", project_id)
        os.makedirs(directory)
        with open(os.path.join(directory, "data.csv"), "w") as handle:
            handle.write(matrix)
        rng = np.random.RandomState(0)
        features = pd.DataFrame(rng.normal(size=(40, 3)), columns=["A", "B", "C"])
        outcome = np.array([0, 1] * 20)
        features.loc[outcome == 1, "A"] += 1.5
        model = build_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=10), k=3, scaler="standard"
        )
        model.fit(features, outcome)
        payload = {
            "method": "Classification",
            "name": "logistic_regression",
            "model": model,
            "feature_names": ["A", "B", "C"],
            "classes": {0: "class0", 1: "class1"},
        }
        with open(os.path.join(directory, "pickle.pkl"), "wb") as handle:
            pickle.dump(payload, handle)

    def test_reordered_matrix_reports_order_difference(self):
        project_id = "PRED-BCLASS-aaaaaa-bbbbbb"
        self._write_project(project_id, ",s1,s2\nC,0.30,0.32\nA,0.10,0.12\nB,0.20,0.22\n")
        response = self.client.get("/maler/predict_result/%s" % project_id)
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("Feature alignment notice", body)
        self.assertIn("Order differed for", body)

    def test_unexpected_columns_are_reported(self):
        project_id = "PRED-BCLASS-cccccc-dddddd"
        self._write_project(
            project_id, ",s1,s2\nA,0.10,0.12\nB,0.20,0.22\nC,0.30,0.32\nD,0.40,0.42\n"
        )
        response = self.client.get("/maler/predict_result/%s" % project_id)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Unexpected columns ignored", response.content.decode("utf-8"))

    def test_correct_matrix_is_silent(self):
        project_id = "PRED-BCLASS-eeeeee-ffffff"
        self._write_project(project_id, ",s1,s2\nA,0.10,0.12\nB,0.20,0.22\nC,0.30,0.32\n")
        response = self.client.get("/maler/predict_result/%s" % project_id)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Feature alignment notice", response.content.decode("utf-8"))


class AutomatedUsabilityTests(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_core_task_pages_render_and_expose_expected_controls(self):
        expectations = {
            "/maler/home": b"MALER",
            "/maler/analysis": b"feature_select_method",
            "/maler/predict": b".maler",
            "/maler/help": b"Help",
        }
        for path, marker in expectations.items():
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(marker.lower(), response.content.lower(), path)

    def test_predict_page_warns_about_privacy_and_retention(self):
        response = self.client.get("/maler/predict")
        content = response.content.lower()
        self.assertIn(b"privacy", content)
        self.assertIn(b"direct identifiers", content)
        self.assertIn(b"24 hours", content)
        self.assertNotIn(b'accept=".pkl', content)
        self.assertNotIn(b'accept=".pickle', content)

    def test_analysis_page_explains_validated_options_and_exposes_registry(self):
        response = self.client.get("/maler/analysis")
        content = response.content.lower()
        for marker in (b"fold", b"mrmr", b"fss", b"bss", b"pca", b"no reduction",
                       b"directly identifying", b"method registry"):
            self.assertIn(marker, content)
        registry = self.client.get("/maler/analysis/methods.json")
        self.assertEqual(registry.status_code, 200)
        self.assertIn("random_forest", registry.json()["models"]["classification"]["models"])

    def test_missing_model_upload_returns_actionable_error(self):
        response = self.client.post("/maler/predict_preview", {
            "select_model": "model_bclass",
            "file_upload_type": "upload_data",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"only signed .maler", response.content.lower())


class ValidatedWebsiteWorkflowTests(unittest.TestCase):
    def _write_regression_project(self, root, projectid):
        directory = os.path.join(root, "cache", projectid)
        os.makedirs(directory)
        rng = np.random.RandomState(101)
        sample_names = ["sample_%02d" % index for index in range(18)]
        features = rng.normal(size=(4, len(sample_names)))
        outcome = 2.5 * features[0] - 0.7 * features[1] + rng.normal(scale=0.1, size=len(sample_names))
        sets = ["training"] * 14 + ["testing"] * 4
        matrix = pd.DataFrame(
            np.vstack([outcome, sets, features]),
            index=["label", "set", "g0", "g1", "g2", "g3"],
            columns=sample_names,
        )
        matrix.to_csv(os.path.join(directory, "data.csv"))
        return directory

    def _write_classification_project(self, root, projectid):
        directory = os.path.join(root, "cache", projectid)
        os.makedirs(directory)
        rng = np.random.RandomState(103)
        sample_names = ["sample_%02d" % index for index in range(30)]
        outcome = np.array(["control", "case"] * 15)
        features = rng.normal(size=(5, len(sample_names)))
        features[0, outcome == "case"] += 2.0
        sets = ["training"] * 24 + ["testing"] * 6
        pd.DataFrame(np.vstack([outcome, sets, features]),
                     index=["label", "set", "g0", "g1", "g2", "g3", "g4"],
                     columns=sample_names).to_csv(os.path.join(directory, "data.csv"))
        return directory

    def _write_survival_project(self, root, projectid):
        directory = os.path.join(root, "cache", projectid)
        os.makedirs(directory)
        rng = np.random.RandomState(107)
        sample_names = ["sample_%02d" % index for index in range(30)]
        features = rng.normal(size=(5, len(sample_names)))
        status = np.array([1, 1, 0] * 10)
        follow_up = np.exp(2.0 - features[0] + rng.normal(scale=0.2, size=30))
        sets = ["training"] * 24 + ["testing"] * 6
        pd.DataFrame(np.vstack([status, follow_up, sets, features]),
                     index=["status", "time", "set", "g0", "g1", "g2", "g3", "g4"],
                     columns=sample_names).to_csv(os.path.join(directory, "data.csv"))
        return directory

    def test_result_requires_capability_token_and_project_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as root:
            projectid = "RO-AZ-test01-TopK-1234abcd"
            directory = self._write_regression_project(root, projectid)
            with override_settings(STATIC_ROOT=root, MALER_CACHE_RETENTION_HOURS=24):
                token = issue_task_token(projectid)
                client = Client()
                self.assertEqual(client.get(
                    "/maler/regression_oc_result/%s" % projectid).status_code, 403)
                response = client.get(
                    "/maler/regression_oc_result/%s?token=%s" % (projectid, token))
                self.assertEqual(response.status_code, 200)
                deletion = client.post("/maler/task/%s/delete" % projectid,
                                       {"access_token": token})
                self.assertEqual(deletion.status_code, 200)
                self.assertFalse(os.path.exists(directory))

    def test_preview_does_not_prefit_normalization_and_issues_private_access(self):
        with tempfile.TemporaryDirectory() as root:
            sample_names = ["s%d" % index for index in range(12)]
            values = np.arange(36, dtype=float).reshape(3, 12)
            outcome = np.linspace(1.0, 3.0, 12)
            sets = ["training"] * 8 + ["testing"] * 4
            matrix = pd.DataFrame(np.vstack([outcome, sets, values]),
                                  index=["label", "set", "g0", "g1", "g2"],
                                  columns=sample_names)
            upload = SimpleUploadedFile(
                "input.csv", matrix.to_csv().encode("utf-8"), content_type="text/csv")
            with override_settings(STATIC_ROOT=root):
                response = Client().post("/maler/preview", {
                    "projectid": "RO-AZ-upload-TopK",
                    "feature_select_method": "TopK",
                    "fsm": "A",
                    "file_upload_type": "user_data",
                    "select_model": "model_reg",
                    "strategy": "O",
                    "to_mail": "",
                    "feature_norm": "Z",
                    "data_consent": "confirmed",
                    "upload_file": upload,
                })
                self.assertEqual(response.status_code, 200, response.content.decode("utf-8"))
                projectid = response.context["projectid"]
                directory = os.path.join(root, "cache", projectid)
                self.assertTrue(response.context["access_token"])
                self.assertTrue(os.path.exists(os.path.join(directory, "access.json")))
                self.assertFalse(os.path.exists(os.path.join(directory, "normalization_data.pkl")))

    def test_regression_post_runs_nested_validation_and_exports_json(self):
        with tempfile.TemporaryDirectory() as root:
            projectid = "RO-AZ-test02-TopK-5678abcd"
            self._write_regression_project(root, projectid)
            with override_settings(
                STATIC_ROOT=root,
                MALER_CACHE_RETENTION_HOURS=24,
                MALER_WEB_MODELS={"regression": ["ridge"]},
                MALER_WEB_OUTER_SPLITS=2,
                MALER_WEB_OUTER_REPEATS=1,
                MALER_WEB_INNER_SPLITS=2,
                MALER_WEB_N_JOBS=1,
                MALER_MODEL_SIGNING_KEY="test-signing-key",
            ):
                token = issue_task_token(projectid)
                client = Client()
                response = client.post("/maler/regression_oc_result/%s" % projectid, {
                    "access_token": token,
                    "feature_select_method": "TopK",
                    "fsm": "A",
                    "feature_norm": "Z",
                })
                self.assertEqual(response.status_code, 200, response.content.decode("utf-8"))
                exported = client.get(
                    "/maler/task/%s/result.json?token=%s" % (projectid, token))
                self.assertEqual(exported.status_code, 200)
                payload = exported.json()
                self.assertEqual(payload["validation"]["outer_splits"], 2)
                self.assertEqual(payload["models"][0]["key"], "ridge")
                self.assertIn("heldout_test", payload["models"][0])
                self.assertTrue(os.path.exists(os.path.join(
                    root, "cache", projectid, "ridge.maler")))

    def test_classification_and_survival_public_routes_use_validated_engine(self):
        cases = [
            ("BCO-AZ-class01-TopK-abcd1234", self._write_classification_project,
             "classification", "logistic_regression", "/maler/classification_oc_result/"),
            ("SO-CZ-surv01-TopK-abcd5678", self._write_survival_project,
             "survival", "survival_tree", "/maler/survival_oc_result/"),
        ]
        for projectid, writer, task, model, route in cases:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as root:
                writer(root, projectid)
                with override_settings(
                    STATIC_ROOT=root,
                    MALER_CACHE_RETENTION_HOURS=24,
                    MALER_WEB_MODELS={task: [model]},
                    MALER_WEB_OUTER_SPLITS=2,
                    MALER_WEB_OUTER_REPEATS=1,
                    MALER_WEB_INNER_SPLITS=2,
                    MALER_WEB_N_JOBS=1,
                    MALER_MODEL_SIGNING_KEY="test-signing-key",
                ):
                    token = issue_task_token(projectid)
                    response = Client().post(route + projectid, {
                        "access_token": token,
                        "feature_select_method": "TopK",
                        "fsm": "C" if task == "survival" else "A",
                        "feature_norm": "Z",
                    })
                    self.assertEqual(response.status_code, 200, response.content.decode("utf-8"))
                    with open(os.path.join(root, "cache", projectid, "validated_result.json"),
                              "r", encoding="utf-8") as handle:
                        result = json.load(handle)
                    self.assertEqual(result["task"], task)
                    self.assertEqual(result["models"][0]["key"], model)

    def test_custom_model_route_uses_server_created_estimator_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            projectid = "RC-AZ-custom01-token-abcd9012"
            directory = self._write_regression_project(root, projectid)
            model_md5 = "trusted-config"
            with open(os.path.join(directory, "model_pickle.pkl"), "wb") as handle:
                pickle.dump({model_md5: {
                    "model": Ridge(alpha=1.0),
                    "model_name": "ridge_custom",
                    "gridsearch_para": {"alpha": [0.1, 1.0]},
                }}, handle)
            with override_settings(
                STATIC_ROOT=root,
                MALER_CACHE_RETENTION_HOURS=24,
                MALER_WEB_OUTER_SPLITS=2,
                MALER_WEB_OUTER_REPEATS=1,
                MALER_WEB_INNER_SPLITS=2,
                MALER_WEB_N_JOBS=1,
                MALER_MODEL_SIGNING_KEY="test-signing-key",
            ):
                token = issue_task_token(projectid)
                response = Client().post("/maler/regression_cp_result/" + projectid, {
                    "access_token": token,
                    "feature_select_method": "TopK",
                    "fsm": "A",
                    "feature_norm": "Z",
                    "model_md5": model_md5,
                })
                self.assertEqual(response.status_code, 200, response.content.decode("utf-8"))
                with open(os.path.join(directory, "validated_result.json"),
                          "r", encoding="utf-8") as handle:
                    result = json.load(handle)
                self.assertEqual(result["strategy"], "custom")
                self.assertEqual(result["models"][0]["key"], "ridge_custom")


if __name__ == "__main__":
    unittest.main()
