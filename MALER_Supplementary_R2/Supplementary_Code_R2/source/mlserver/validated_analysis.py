"""Unified leakage-safe analysis service used by every public result route."""

from __future__ import absolute_import

import csv
import hashlib
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid

from .model_registry import (
    MODEL_REGISTRY, MODEL_REGISTRY_VERSION, build_estimator, get_model_spec,
    normalize_model_key, pipeline_param_grid,
)
from .safe_ml import (
    build_classification_pipeline, build_regression_pipeline, classification_metrics,
    classification_prediction,
    fit_final_model, nested_cv_classification, nested_cv_regression, regression_metrics,
    save_signed_model_bundle, selected_feature_names, validate_feature_matrix, validate_target,
    file_sha256,
)
from .safe_survival import (
    build_survival_pipeline, nested_cv_survival, survival_metrics, validate_survival_target,
)
from .patient_partitions import patient_partition_audit
from .cv_policy import make_splits, validate_group_feature_method
from .source_version import source_identity


SCHEMA_VERSION = 3


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
    metadata_names = {"subject_id", "cohort_provenance"}
    metadata = {str(name).lower(): name for name in raw.columns[required:]
                if str(name).lower() in metadata_names}
    if len(metadata) != len([name for name in raw.columns[required:]
                             if str(name).lower() in metadata_names]):
        raise ValueError("Duplicate reserved metadata row names are not allowed.")
    groups = None
    if "subject_id" in metadata:
        group_values = raw[metadata["subject_id"]]
        if group_values.isna().any() or group_values.astype(str).str.strip().eq("").any():
            raise ValueError("subject_id must be present for every sample when supplied.")
        groups = group_values.astype(str).str.strip()
        shared = set(groups.loc[training]).intersection(groups.loc[testing])
        if shared:
            raise ValueError("%d subject IDs cross development and heldout partitions." % len(shared))
    raw.attrs["groups"] = groups
    raw.attrs["cohort_provenance"] = (
        raw[metadata["cohort_provenance"]].astype(str).str.strip().to_dict()
        if "cohort_provenance" in metadata else None)
    features = raw.drop(columns=list(raw.columns[:required]) + list(metadata.values()))
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
    """Serialize task execution so no concurrent refit can replace a locked model."""
    if os.path.exists(os.path.join(project_dir, "selection_lock.json")):
        raise ValueError("This project already has a selection lock; the heldout cohort cannot be evaluated again.")
    lease = os.path.join(project_dir, ".analysis_running")
    try:
        descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("An analysis is already running for this project.")
    os.close(descriptor)
    try:
        return _run_validated_analysis(project_dir, projectid, options)
    finally:
        os.unlink(lease)


def _run_validated_analysis(project_dir, projectid, options):
    identity = source_identity()
    task = infer_task(projectid)
    strategy = infer_strategy(projectid)
    raw, X, y = parse_dataset(os.path.join(project_dir, "data.csv"), task)
    method = normalize_feature_method(options.get("feature_select_method"), options.get("fsm"))
    requested_outer = int(getattr(settings, "MALER_WEB_OUTER_SPLITS", 5))
    repeats = int(getattr(settings, "MALER_WEB_OUTER_REPEATS", 2))
    requested_inner = int(getattr(settings, "MALER_WEB_INNER_SPLITS", 3))
    n_jobs = int(getattr(settings, "MALER_WEB_N_JOBS", 1))
    outer, inner = _cv_sizes(task, y["train"], requested_outer, requested_inner)
    group_series = raw.attrs.get("groups")
    groups = None if group_series is None else group_series.loc[X["train"].index].to_numpy()
    validate_group_feature_method(groups, method)
    if groups is not None:
        outer = min(outer, len(np.unique(groups)))
        inner = min(inner, len(np.unique(groups)) - 1)
        if inner < 2:
            raise ValueError("At least three distinct development subject IDs are needed for nested grouped CV.")
    positive_class = options.get("positive_class")
    if task == "classification" and len(np.unique(y["train"])) == 2:
        available = list(np.unique(y["train"]))
        matching = [label for label in available if str(label) == str(positive_class)]
        if len(matching) != 1:
            raise ValueError("Choose the binary positive class from: %s" %
                             ", ".join(map(str, available)))
        positive_class = matching[0]
    else:
        positive_class = None
    threshold_policy = options.get("threshold_policy") or "estimator_native_predict"
    raw_threshold = options.get("threshold_value")
    threshold_value = float(0.5 if raw_threshold is None or raw_threshold == "" else raw_threshold)
    if threshold_policy not in ("estimator_native_predict", "explicit_probability_threshold"):
        raise ValueError("Unsupported classification threshold policy.")
    if threshold_policy == "explicit_probability_threshold":
        if positive_class is None or not np.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
            raise ValueError("An explicit probability cutoff requires binary data and a finite value in [0,1].")
        if method in ("fss", "bss"):
            raise ValueError("FSS/BSS require the estimator-native threshold policy in this release.")
    patient_audit = raw.attrs.get("patient_level_audit") or {}
    patient_counts = {
        "source_samples": patient_audit.get("source_samples", 0),
        "retained_patients": patient_audit.get("retained_patients", 0),
        "excluded_cross_partition_patients": len(patient_audit.get("excluded_cross_partition_patients", [])),
        "collapsed_within_partition_patients": len(patient_audit.get("collapsed_within_partition_patients", [])),
    }
    if patient_audit:
        private_path = os.path.join(project_dir, "patient_audit_private.json")
        with open(private_path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(patient_audit, handle, indent=2)
        os.replace(private_path + ".tmp", private_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": MODEL_REGISTRY_VERSION,
        "software": identity,
        "project_id": projectid,
        "task": task,
        "strategy": strategy,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": {"training_samples": int(len(X["train"])), "testing_samples": int(len(X["test"])),
                    "blind_samples": int(len(X["blind"])), "features": int(X["train"].shape[1]),
                     "patient_level_audit": patient_counts,
                     "split_level": "subject" if groups is not None else "sample",
                     "development_subjects": int(len(np.unique(groups))) if groups is not None else None,
                     # Provenance labels can identify a clinical site or study.
                     # Keep them in the private input, not in shareable results.
                     "cohort_provenance_supplied": bool(raw.attrs.get("cohort_provenance")),
                     "independent_external_cohort_verified": False},
        "validation": {"outer_splits": outer, "outer_repeats": repeats,
                        "inner_splits": inner, "random_state": 10,
                        "preprocessing_scope": "fit within inner and outer training folds; FSS/BSS internal scoring uses upstream training-fold preprocessing",
                        "family_scope": "fixed family per outer CV; family selected from development summaries",
                        "selection_metric": {"classification": "balanced_accuracy", "regression": "r2",
                                             "survival": "c_index"}[task],
                        "threshold_policy": threshold_policy if positive_class is not None else "not applicable",
                        "threshold_value": threshold_value if threshold_policy == "explicit_probability_threshold" else None},
        "positive_class": positive_class,
        "feature_reduction": method,
        "models": [],
        "failures": [],
        "warnings": [],
    }
    signing_key = getattr(settings, "MALER_MODEL_SIGNING_KEY", "")
    if not signing_key:
        result["warnings"].append(
            "Signed model export is disabled until MALER_MODEL_SIGNING_KEY is configured.")
    removed = (patient_counts["excluded_cross_partition_patients"]
               + patient_counts["collapsed_within_partition_patients"])
    if removed:
        result["warnings"].append(
            "Patient-level audit excluded cross-partition specimens and collapsed within-partition aliquots; see the private project audit for IDs.")
    if task == "classification":
        warning = class_imbalance_warning(y["train"])
        if warning:
            result["warnings"].append(warning)
    candidates = {}
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
                    n_jobs=n_jobs, feature_method=method, groups=groups,
                    positive_class=positive_class, threshold_policy=threshold_policy,
                    threshold=threshold_value)
                pipeline = build_classification_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=method)
                final_grid = dict(grid)
                if method in ("select", "mrmr", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif method == "pca":
                    final_grid["selector__n_components"] = k_values
                prediction_files = {}
                heldout = None
            elif task == "regression":
                nested = nested_cv_regression(
                    X["train"], y["train"], estimator, grid, k_values, scaler=scaler,
                    outer_splits=outer, outer_repeats=repeats, inner_splits=inner,
                    n_jobs=n_jobs, feature_method=method, groups=groups)
                pipeline = build_regression_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=method)
                final_grid = dict(grid)
                if method in ("select", "mrmr", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif method == "pca":
                    final_grid["selector__n_components"] = k_values
                prediction_files = {}
                heldout = None
            else:
                survival_method = "select" if method == "mrmr" else method
                nested = nested_cv_survival(
                    X["train"], y["train"], estimator, grid, k_values, scaler=scaler,
                    outer_splits=outer, outer_repeats=repeats, inner_splits=inner,
                    n_jobs=n_jobs, feature_method=survival_method, groups=groups)
                pipeline = build_survival_pipeline(
                    estimator, k=max(k_values), scaler=scaler, feature_method=survival_method)
                final_grid = dict(grid)
                if survival_method in ("select", "fss", "bss"):
                    final_grid["selector__k"] = k_values
                elif survival_method == "pca":
                    final_grid["selector__n_components"] = k_values
                prediction_files = {}
                heldout = None
            actual_estimator = pipeline.named_steps['model']
            model_result = {"key": key, "estimator": actual_estimator.__class__.__module__ + "." + actual_estimator.__class__.__name__,
                            "scaler": scaler, "nested_cv": nested, "best_params": None,
                            "selected_features": None, "heldout_test": heldout,
                            "candidate_grid": final_grid,
                            "candidate_count": len(ParameterGrid(final_grid)),
                            "hyperparameter_candidate_count": len(ParameterGrid(grid)),
                            "effective_feature_method": survival_method if task == "survival" else method,
                            "prediction_files": prediction_files}
            candidates[key] = (pipeline, final_grid)
            result["models"].append(model_result)
        except Exception as exc:
            result["failures"].append({"key": key, "error": str(exc)})
    if not result["models"]:
        raise ValueError("No requested model completed successfully: %s" % result["failures"])
    metric_name = result["validation"]["selection_metric"]
    ranked = sorted(
        result["models"],
        key=lambda model: (-float(model["nested_cv"]["summary"][metric_name]["mean"]),
                           str(model["key"])))
    chosen = ranked[0]
    selected_key = chosen["key"]
    family_selected_at = datetime.utcnow().isoformat() + "Z"
    pipeline, final_grid = candidates[selected_key]
    if task == "survival":
        inner_cv, _ = make_splits(X["train"], y["train"], "survival", inner,
                                  seed=10, groups=groups, inner=True)
        final = GridSearchCV(pipeline, final_grid, cv=inner_cv,
                             n_jobs=n_jobs, error_score="raise")
        final.fit(X["train"], y["train"])
    else:
        final = fit_final_model(
            X["train"], y["train"], pipeline, final_grid, task,
            inner_splits=inner, n_jobs=n_jobs, groups=groups,
            positive_class=positive_class, threshold_policy=threshold_policy,
            threshold=threshold_value)
    chosen["best_params"] = final.best_params_
    chosen["selected_features"] = selected_feature_names(final, X["train"].columns)
    fitted_path = os.path.join(project_dir, "selected_pipeline.joblib")
    joblib.dump(final.best_estimator_, fitted_path, compress=3)
    fitted_hash = file_sha256(fitted_path)
    selected_at = datetime.utcnow().isoformat() + "Z"
    with open(os.path.join(project_dir, "data.csv"), "rb") as handle:
        data_hash = hashlib.sha256(handle.read()).hexdigest()
    lock = {
        "schema_version": SCHEMA_VERSION,
        "project_id": projectid,
        "input_sha256": data_hash,
        "selected_family": selected_key,
        "selection_metric": metric_name,
        "selection_rule": "highest development per-family outer-CV mean; lexicographic tie-break",
        "candidate_families": [model["key"] for model in result["models"]],
        "candidate_grids": {model["key"]: model["candidate_grid"] for model in result["models"]},
        "candidate_counts": {model["key"]: model["candidate_count"] for model in result["models"]},
        "estimator": chosen["estimator"],
        "imputer_strategy": "median",
        "scaler": chosen["scaler"],
        "feature_method": chosen["effective_feature_method"],
        "best_params": chosen["best_params"],
        "input_features": list(map(str, X["train"].columns)),
        "selected_features": chosen["selected_features"],
        "fitted_model_sha256": fitted_hash,
        "class_order": list(map(str, final.classes_)) if task == "classification" else None,
        "family_selected_at": family_selected_at,
        "validation_budget": {"requested_outer_splits": requested_outer,
                              "requested_outer_repeats": repeats,
                              "requested_inner_splits": requested_inner,
                              "actual_outer_splits": outer,
                              "actual_outer_repeats": chosen["nested_cv"]["actual_repeats"],
                              "actual_outer_evaluations": len(chosen["nested_cv"]["folds"]),
                              "actual_inner_splits": inner},
        "outer_split_ids_hash": chosen["nested_cv"]["split_ids_hash"],
        "inner_split_policy": "fold-local task-aware CV; grouped when subject_id supplied",
        "random_seed": 10,
        "positive_class": str(positive_class) if positive_class is not None else None,
        "threshold_policy": threshold_policy if positive_class is not None else None,
        "threshold_value": threshold_value if threshold_policy == "explicit_probability_threshold" else None,
        "selected_at": selected_at,
        "software_commit": identity["commit"],
        "software_identity": identity,
        # This is written once, immediately before any heldout prediction.
        "heldout_unlocked_at": datetime.utcnow().isoformat() + "Z"
        if len(X["test"]) else None,
    }
    lock_path = os.path.join(project_dir, "selection_lock.json")
    try:
        with os.fdopen(os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise ValueError("This project already has a selection lock; the heldout cohort cannot be evaluated again.")
    result["selection"] = dict(lock)
    lock_hash = file_sha256(lock_path)
    result["selection_lock_sha256"] = lock_hash
    result["validation"]["requested_budget"] = {
        "outer_splits": requested_outer, "outer_repeats": repeats, "inner_splits": requested_inner}
    result["selected_family"] = selected_key
    prediction_files = chosen["prediction_files"]
    if len(X["test"]):
        if task == "classification":
            prediction, scores, score_type = classification_prediction(
                final, X["test"], positive_class, threshold_policy, threshold_value)
            chosen["heldout_test"] = classification_metrics(
                y["test"], prediction, scores=scores, classes=final.classes_,
                positive_class=positive_class, score_type=score_type)
            table = pd.DataFrame({"observed": y["test"], "predicted": prediction},
                                 index=X["test"].index)
            if scores is not None:
                score_array = np.asarray(scores)
                if score_array.ndim == 2:
                    for index, class_name in enumerate(final.classes_):
                        table["score_%s" % class_name] = score_array[:, index]
                else:
                    table["decision_score"] = score_array
        elif task == "regression":
            prediction = final.predict(X["test"])
            chosen["heldout_test"] = regression_metrics(y["test"], prediction)
            table = pd.DataFrame({"observed": y["test"], "predicted": prediction},
                                 index=X["test"].index)
        else:
            chosen["heldout_test"] = survival_metrics(
                final.best_estimator_, X["train"], y["train"], X["test"], y["test"])
            table = pd.DataFrame({"event": y["test"]["event"], "time": y["test"]["time"],
                                  "risk_score": final.predict(X["test"])}, index=X["test"].index)
        prediction_files["testing"] = _write_prediction_table(
            project_dir, selected_key, "testing", table)
    if len(X["blind"]):
        if task == "classification":
            prediction, scores, score_type = classification_prediction(
                final, X["blind"], positive_class, threshold_policy, threshold_value)
            table = pd.DataFrame({"predicted": prediction}, index=X["blind"].index)
            if scores is not None and np.asarray(scores).ndim == 2:
                for index, class_name in enumerate(final.classes_):
                    table["score_%s" % class_name] = np.asarray(scores)[:, index]
        else:
            column = "risk_score" if task == "survival" else "predicted"
            table = pd.DataFrame({column: final.predict(X["blind"])}, index=X["blind"].index)
        prediction_files["blind"] = _write_prediction_table(
            project_dir, selected_key, "blind", table)
    if signing_key:
        bundle_name = "%s.maler" % re.sub(r"[^A-Za-z0-9_.-]+", "_", str(selected_key))
        metadata = {"project_id": projectid, "model_key": selected_key,
                    "model_name": selected_key,
                    "method": {"classification": "model_bclass" if positive_class is not None else "model_mclass",
                               "regression": "model_reg", "survival": "model_sur"}[task],
                    "positive_class": str(positive_class) if positive_class is not None else None,
                    "class_order": list(map(str, final.classes_)) if task == "classification" else None,
                    "threshold_policy": threshold_policy if positive_class is not None else None,
                    "threshold_value": threshold_value if threshold_policy == "explicit_probability_threshold" else None,
                    "software_commit": lock["software_commit"],
                    "configuration": {name: lock[name] for name in (
                        "estimator", "imputer_strategy", "scaler", "feature_method",
                        "best_params", "selected_features", "validation_budget")},
                    "fitted_model_sha256": fitted_hash,
                    "selection_lock_sha256": lock_hash}
        save_signed_model_bundle(os.path.join(project_dir, bundle_name), final.best_estimator_, task,
                                 X["train"].columns, signing_key, metadata=metadata)
        chosen["model_bundle"] = bundle_name
    result["warnings"].append(
        "The development outer-CV scores describe fixed estimator families; the winning family was selected using those scores. Only the selected family has a heldout result.")
    if "external" in set(raw.iloc[:, 1 if task != "survival" else 2].astype(str).str.lower()):
        result["warnings"].append(
            "An input set was named external; independence was not established by that label.")
    temporary = os.path.join(project_dir, "validated_result.json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=_json_default, indent=2, sort_keys=True)
    os.replace(temporary, os.path.join(project_dir, "validated_result.json"))
    return result
