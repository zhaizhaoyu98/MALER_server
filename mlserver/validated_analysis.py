"""Unified leakage-safe analysis service used by every public result route."""

from __future__ import absolute_import

import csv
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .model_registry import (
    MODEL_REGISTRY, MODEL_REGISTRY_VERSION, build_estimator, get_model_spec,
    normalize_model_key, pipeline_param_grid,
)
from .safe_ml import (
    build_classification_pipeline, build_regression_pipeline, classification_metrics,
    fit_final_model, nested_cv_classification, nested_cv_regression, regression_metrics,
    save_signed_model_bundle, selected_feature_names, validate_feature_matrix, validate_target,
)
from .safe_survival import (
    build_survival_pipeline, nested_cv_survival, survival_metrics, validate_survival_target,
)
from .patient_partitions import patient_partition_audit


SCHEMA_VERSION = 1


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def validate_sample_header(header_line):
    # pandas silently mangles duplicate CSV header names (for example S1 -> S1.1).
    # Check the original sample identifiers before pandas can rename them.
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=",\t;|/")
        header = next(csv.reader([header_line], dialect=dialect))
    except (csv.Error, StopIteration):
        raise ValueError("The uploaded matrix has an invalid CSV header.")
    if len(header) < 2:
        raise ValueError("The uploaded matrix requires sample identifiers in its header.")
    sample_identifiers = header[1:]
    if len(sample_identifiers) != len(set(sample_identifiers)):
        raise ValueError("Duplicate sample identifiers are not allowed.")


def read_uploaded_matrix(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        validate_sample_header(handle.readline())
    data = pd.read_csv(path, header=0, index_col=0, sep=None, engine="python").T
    data.index = data.index.map(str)
    data.columns = data.columns.map(str)
    if data.index.has_duplicates:
        raise ValueError("Duplicate sample identifiers are not allowed.")
    if data.columns.has_duplicates:
        raise ValueError("Duplicate row/feature identifiers are not allowed.")
    return data


def infer_task(projectid):
    prefix = str(projectid).split("-", 1)[0].upper()
    if prefix.startswith("B") or prefix.startswith("M"):
        return "classification"
    if prefix.startswith("R"):
        return "regression"
    if prefix.startswith("S"):
        return "survival"
    raise ValueError("The project identifier does not encode a supported analysis type.")


def infer_strategy(projectid):
    prefix = str(projectid).split("-", 1)[0].upper()
    return "custom" if prefix.endswith("C") else "one_click"


def _partition_masks(values):
    normalized = pd.Series(values).astype(str).str.strip().str.lower()
    training = normalized.isin(["training", "train", "development", "developmental"])
    testing = normalized.isin(["testing", "test", "validation", "external"])
    blind = normalized.isin(["blind", "prediction", "predict"])
    unknown = ~(training | testing | blind)
    if unknown.any():
        examples = sorted(normalized[unknown].unique().tolist())[:10]
        raise ValueError("Unsupported set labels: %s" % ", ".join(examples))
    if not training.any():
        raise ValueError("At least one training/development sample is required.")
    return training.to_numpy(), testing.to_numpy(), blind.to_numpy()


def parse_dataset(path, task):
    raw = read_uploaded_matrix(path)
    outcome_columns = 2 if task == "survival" else 1
    required = outcome_columns + 1
    if raw.shape[1] <= required:
        raise ValueError("The uploaded file contains no usable feature columns.")
    training, testing, blind = _partition_masks(raw.iloc[:, outcome_columns])
    analysis_rows = np.flatnonzero(training | testing)
    analysis_split = np.where(training[analysis_rows], "train", "test")
    retained, patient_audit = patient_partition_audit(raw.index[analysis_rows], analysis_split)
    if len(retained) != len(analysis_rows):
        keep_rows = np.sort(np.r_[analysis_rows[retained], np.flatnonzero(blind)])
        raw = raw.iloc[keep_rows].copy()
        training, testing, blind = _partition_masks(raw.iloc[:, outcome_columns])
    raw.attrs["patient_level_audit"] = patient_audit
    features = raw.iloc[:, required:]
    train_features = validate_feature_matrix(features.loc[training])
    feature_sets = {
        "train": train_features,
        "test": validate_feature_matrix(features.loc[testing], reject_zero_variance=False)
        if testing.any() else pd.DataFrame(columns=train_features.columns),
        "blind": validate_feature_matrix(features.loc[blind], reject_zero_variance=False)
        if blind.any() else pd.DataFrame(columns=train_features.columns),
    }
    if task == "survival":
        targets = {
            "train": validate_survival_target(raw.loc[training].iloc[:, 0], raw.loc[training].iloc[:, 1]),
            "test": validate_survival_target(raw.loc[testing].iloc[:, 0], raw.loc[testing].iloc[:, 1],
                                               require_event_mix=False)
            if testing.any() else None,
        }
    else:
        target_task = "classification" if task == "classification" else "regression"
        targets = {
            "train": validate_target(raw.loc[training].iloc[:, 0], target_task),
            "test": validate_target(raw.loc[testing].iloc[:, 0], target_task,
                                    require_class_counts=False)
            if testing.any() else None,
        }
    return raw, feature_sets, targets


def normalize_feature_method(feature_select_method, fsm="A"):
    subset = str(feature_select_method or "TopK").strip().lower()
    ranking = str(fsm or "A").strip().lower()
    if subset in ("none", "off", "no"):
        return "none"
    if subset in ("pca", "principal-components"):
        return "pca"
    if subset in ("fss", "forward", "forward-selection"):
        return "fss"
    if subset in ("bss", "backward", "backward-selection"):
        return "bss"
    return "mrmr" if ranking in ("m", "mrmr") else "select"


def normalize_scaler(value, requires_scaling=True):
    if not requires_scaling:
        return "none"
    return {"Z": "standard", "MM": "minmax", "MA": "maxabs", "N": "none"}.get(
        str(value or "Z").upper(), "standard")


def class_imbalance_warning(labels):
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() == 0 or counts.max() / counts.min() < 3:
        return None
    summary = ", ".join("%s=%d" % (name, count) for name, count in counts.items())
    return ("Class imbalance in the development set (%s). Review per-class metrics and "
            "balanced accuracy; MALER does not automatically reweight classes." % summary)


def _bounded_grid(grid, maximum_candidates=24):
    bounded = {key: list(values) for key, values in grid.items()}
    product = 1
    for values in bounded.values():
        product *= max(1, len(values))
    while product > maximum_candidates:
        longest = max(bounded, key=lambda key: len(bounded[key]))
        if len(bounded[longest]) <= 1:
            break
        bounded[longest] = bounded[longest][:-1]
        product = 1
        for values in bounded.values():
            product *= max(1, len(values))
    return bounded


def _custom_model(project_dir, model_md5):
    if not model_md5:
        raise ValueError("A custom analysis requires a selected model configuration.")
    with open(os.path.join(project_dir, "model_pickle.pkl"), "rb") as handle:
        model_set = pickle.load(handle)
    if model_md5 not in model_set:
        raise ValueError("The requested custom model configuration was not found.")
    record = model_set[model_md5]
    estimator = clone(record["model"])
    valid = estimator.get_params(deep=True)
    grid = {}
    for raw_name, values in dict(record.get("gridsearch_para") or {}).items():
        name = str(raw_name).replace("estimator__", "")
        if name in valid:
            grid["model__" + name] = values if isinstance(values, (list, tuple)) else [values]
    key = str(record.get("model_name") or estimator.__class__.__name__)
    return key, estimator, grid, True


def _model_records(task, strategy, project_dir, model_md5):
    if strategy == "custom":
        return [_custom_model(project_dir, model_md5)]
    configured = getattr(settings, "MALER_WEB_MODELS", {}).get(task) if isinstance(
        getattr(settings, "MALER_WEB_MODELS", {}), dict) else None
    keys = configured or list(MODEL_REGISTRY[task]["models"].keys())
    records = []
    for key in keys:
        canonical, estimator = build_estimator(task, key)
        _, grid = pipeline_param_grid(task, canonical)
        _, spec = get_model_spec(task, canonical)
        records.append((canonical, estimator, grid, bool(spec.get("requires_scaling"))))
    return records


def _cv_sizes(task, y, requested_outer, requested_inner):
    if task == "classification":
        minimum = int(pd.Series(y).value_counts().min())
    elif task == "survival":
        minimum = int(min(np.sum(y["event"]), np.sum(~y["event"])))
    else:
        minimum = len(y)
    outer = min(int(requested_outer), minimum)
    if outer < 2:
        raise ValueError("The training set is too small for cross-validation.")
    approximate_train = int(np.floor(len(y) * (outer - 1.0) / outer))
    if task == "classification":
        approximate_minimum = max(2, int(np.floor(minimum * (outer - 1.0) / outer)))
        inner = min(int(requested_inner), approximate_minimum)
    elif task == "survival":
        approximate_minimum = max(2, int(np.floor(minimum * (outer - 1.0) / outer)))
        inner = min(int(requested_inner), approximate_minimum)
    else:
        inner = min(int(requested_inner), approximate_train)
    return outer, max(2, inner)


def _feature_counts(method, X):
    maximum = min(X.shape[1], 50)
    if method == "pca":
        # Safe for both outer- and inner-training folds in the default design.
        maximum = min(maximum, max(1, X.shape[0] // 2))
    if method in ("fss", "bss"):
        candidates = [min(5, maximum), min(10, maximum), min(20, maximum)]
    else:
        candidates = [min(10, maximum), min(20, maximum), min(50, maximum)]
    return sorted(set(value for value in candidates if value >= 1))


def _write_prediction_table(project_dir, model_key, cohort, frame):
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_key)).strip("_.") or "model"
    filename = "%s_%s_predictions.csv" % (safe_key, cohort)
    temporary = os.path.join(project_dir, filename + ".tmp")
    frame.to_csv(temporary, index=True, index_label="sample_id")
    os.replace(temporary, os.path.join(project_dir, filename))
    return filename


def run_validated_analysis(project_dir, projectid, options):
    task = infer_task(projectid)
    strategy = infer_strategy(projectid)
    raw, X, y = parse_dataset(os.path.join(project_dir, "data.csv"), task)
    method = normalize_feature_method(options.get("feature_select_method"), options.get("fsm"))
    requested_outer = int(getattr(settings, "MALER_WEB_OUTER_SPLITS", 5))
    repeats = int(getattr(settings, "MALER_WEB_OUTER_REPEATS", 2))
    requested_inner = int(getattr(settings, "MALER_WEB_INNER_SPLITS", 3))
    n_jobs = int(getattr(settings, "MALER_WEB_N_JOBS", 1))
    outer, inner = _cv_sizes(task, y["train"], requested_outer, requested_inner)
    result = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": MODEL_REGISTRY_VERSION,
        "project_id": projectid,
        "task": task,
        "strategy": strategy,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": {"training_samples": int(len(X["train"])), "testing_samples": int(len(X["test"])),
                    "blind_samples": int(len(X["blind"])), "features": int(X["train"].shape[1]),
                    "patient_level_audit": raw.attrs.get("patient_level_audit")},
        "validation": {"outer_splits": outer, "outer_repeats": repeats,
                       "inner_splits": inner, "random_state": 10,
                       "preprocessing_scope": "fit independently within every training fold"},
        "feature_reduction": method,
        "models": [],
        "failures": [],
        "warnings": [],
    }
    signing_key = getattr(settings, "MALER_MODEL_SIGNING_KEY", "")
    if not signing_key:
        result["warnings"].append(
            "Signed model export is disabled until MALER_MODEL_SIGNING_KEY is configured.")
    patient_audit = raw.attrs.get("patient_level_audit") or {}
    removed = (len(patient_audit.get("excluded_cross_partition_patients", []))
               + len(patient_audit.get("collapsed_within_partition_patients", [])))
    if removed:
        result["warnings"].append(
            "Patient-level audit excluded cross-partition specimens and collapsed within-partition aliquots; see dataset.patient_level_audit for exact IDs and retained samples.")
    if task == "classification":
        warning = class_imbalance_warning(y["train"])
        if warning:
            result["warnings"].append(warning)
    for key, estimator, raw_grid, requires_scaling in _model_records(
            task, strategy, project_dir, options.get("model_md5")):
        try:
            scaler = normalize_scaler(options.get("feature_norm"), requires_scaling)
            grid = _bounded_grid(raw_grid)
            k_values = _feature_counts(method, X["train"])
            if task == "classification":
                nested = nested_cv_classification(
                    X["train"], y["train"], estimator, grid, k_values, scaler=scaler,
                    outer_splits=outer, outer_repeats=repeats, inner_splits=inner,
                    n_jobs=n_jobs, feature_method=method)
                pipeline = build_classification_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=method)
                final_grid = dict(grid)
                if method in ("select", "mrmr", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif method == "pca":
                    final_grid["selector__n_components"] = k_values
                final = fit_final_model(X["train"], y["train"], pipeline, final_grid,
                                        "classification", inner_splits=inner, n_jobs=n_jobs)
                prediction_files = {}
                heldout = None
                if len(X["test"]):
                    test_prediction = final.predict(X["test"])
                    test_scores = (final.predict_proba(X["test"]) if hasattr(final, "predict_proba") else
                                   final.decision_function(X["test"]) if hasattr(final, "decision_function") else None)
                    heldout = classification_metrics(
                        y["test"], test_prediction, scores=test_scores, classes=final.classes_)
                    table = pd.DataFrame({"observed": y["test"], "predicted": test_prediction},
                                         index=X["test"].index)
                    if test_scores is not None:
                        score_array = np.asarray(test_scores)
                        if score_array.ndim == 2:
                            for index, class_name in enumerate(final.classes_):
                                table["score_%s" % class_name] = score_array[:, index]
                        else:
                            table["decision_score"] = score_array
                    prediction_files["testing"] = _write_prediction_table(
                        project_dir, key, "testing", table)
                if len(X["blind"]):
                    prediction_files["blind"] = _write_prediction_table(
                        project_dir, key, "blind",
                        pd.DataFrame({"predicted": final.predict(X["blind"])}, index=X["blind"].index))
            elif task == "regression":
                nested = nested_cv_regression(
                    X["train"], y["train"], estimator, grid, k_values, scaler=scaler,
                    outer_splits=outer, outer_repeats=repeats, inner_splits=inner,
                    n_jobs=n_jobs, feature_method=method)
                pipeline = build_regression_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=method)
                final_grid = dict(grid)
                if method in ("select", "mrmr", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif method == "pca":
                    final_grid["selector__n_components"] = k_values
                final = fit_final_model(X["train"], y["train"], pipeline, final_grid,
                                        "regression", inner_splits=inner, n_jobs=n_jobs)
                prediction_files = {}
                heldout = None
                if len(X["test"]):
                    test_prediction = final.predict(X["test"])
                    heldout = regression_metrics(y["test"], test_prediction)
                    prediction_files["testing"] = _write_prediction_table(
                        project_dir, key, "testing",
                        pd.DataFrame({"observed": y["test"], "predicted": test_prediction},
                                     index=X["test"].index))
                if len(X["blind"]):
                    prediction_files["blind"] = _write_prediction_table(
                        project_dir, key, "blind",
                        pd.DataFrame({"predicted": final.predict(X["blind"])}, index=X["blind"].index))
            else:
                survival_method = "select" if method == "mrmr" else method
                nested = nested_cv_survival(
                    X["train"], y["train"], estimator, grid, k_values, scaler=scaler,
                    outer_splits=outer, outer_repeats=repeats, inner_splits=inner,
                    n_jobs=n_jobs, feature_method=survival_method)
                pipeline = build_survival_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=survival_method)
                final_grid = dict(grid)
                if survival_method in ("select", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif survival_method == "pca":
                    final_grid["selector__n_components"] = k_values
                inner_cv = StratifiedKFold(n_splits=inner, shuffle=True, random_state=10)
                final = GridSearchCV(pipeline, final_grid, cv=list(inner_cv.split(
                    X["train"], y["train"]["event"].astype(int))), n_jobs=n_jobs, error_score="raise")
                final.fit(X["train"], y["train"])
                prediction_files = {}
                heldout = None
                if len(X["test"]):
                    heldout = survival_metrics(final.best_estimator_, X["train"], y["train"],
                                               X["test"], y["test"])
                    prediction_files["testing"] = _write_prediction_table(
                        project_dir, key, "testing", pd.DataFrame({
                            "event": y["test"]["event"], "time": y["test"]["time"],
                            "risk_score": final.predict(X["test"]),
                        }, index=X["test"].index))
                if len(X["blind"]):
                    prediction_files["blind"] = _write_prediction_table(
                        project_dir, key, "blind",
                        pd.DataFrame({"risk_score": final.predict(X["blind"])}, index=X["blind"].index))
            feature_names = selected_feature_names(final, X["train"].columns)
            model_result = {"key": key, "estimator": estimator.__class__.__module__ + "." + estimator.__class__.__name__,
                            "scaler": scaler, "nested_cv": nested, "best_params": final.best_params_,
                            "selected_features": feature_names, "heldout_test": heldout,
                            "prediction_files": prediction_files}
            if signing_key:
                bundle_name = "%s.maler" % key.replace(" ", "_")
                save_signed_model_bundle(os.path.join(project_dir, bundle_name), final.best_estimator_, task,
                                         X["train"].columns, signing_key,
                                         metadata={"project_id": projectid, "model_key": key})
                model_result["model_bundle"] = bundle_name
            result["models"].append(model_result)
        except Exception as exc:
            result["failures"].append({"key": key, "error": str(exc)})
    if not result["models"]:
        raise ValueError("No requested model completed successfully: %s" % result["failures"])
    temporary = os.path.join(project_dir, "validated_result.json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=_json_default, indent=2, sort_keys=True)
    os.replace(temporary, os.path.join(project_dir, "validated_result.json"))
    return result
