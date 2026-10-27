"""Independent refit and metric parity for four locked MALER held-out models.

The patient exclusions, selected hyperparameters and input hash come from the
archived result metadata. No MALER pipeline or metric helper is imported here.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sksurv.metrics import concordance_index_censored
from sksurv.tree import SurvivalTree


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("MALER_VALIDATION_OUTPUT_DIR") or ROOT / "validation" / "results")
FIGURES = Path(os.environ.get("MALER_FIGURE_OUTPUT_DIR") or ROOT / "validation" / "figures")
EXAMPLES = ROOT / "mlserver" / "static" / "cache" / "example"


def _logistic_parameters(result):
    if "lbfgs" in result.get("protocol", {}).get("estimator", ""):
        return {"solver": "lbfgs"}
    return {"solver": "liblinear", "multi_class": "ovr"}


def _matrix(filename, metadata, target):
    path = EXAMPLES / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["source_sha256"]:
        raise AssertionError("Input checksum changed: " + filename)
    raw = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
    excluded = set()
    audit = metadata["patient_level_audit"]
    for row in audit["excluded_cross_partition_patients"]:
        excluded.update(row["train_samples"])
        excluded.update(row["test_samples"])
    for row in audit["collapsed_within_partition_patients"]:
        excluded.update(set(row["samples"]) - {row["selected"]})
    raw = raw.drop(columns=list(excluded))
    labels = raw.loc[target]
    split = raw.loc["set"].str.strip().str.lower()
    features = raw.drop(index=[target, "set"]).T.apply(pd.to_numeric)
    train = split.isin(["train", "training"])
    test = split.isin(["test", "testing"])
    if (int(train.sum()), int(test.sum())) != (metadata["n_train"], metadata["n_test"]):
        raise AssertionError("Patient-level split count mismatch")
    return features.loc[train], labels.loc[train], features.loc[test], labels.loc[test]


def _check(name, observed, expected, tolerance=1e-10):
    if not np.isclose(observed, expected, rtol=0, atol=tolerance):
        raise AssertionError("%s: %.15g != %.15g" % (name, observed, expected))
    return float(observed)


def binary_parity():
    result = json.loads((RESULTS / "binary_classification.json").read_text(encoding="utf-8"))
    X_train, y_train, X_test, y_test = _matrix(
        "binary_classification_example.csv", result["dataset"], "label")
    parameters = result["final_best_params"]
    reference = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=parameters["selector__k"])),
        ("model", LogisticRegression(
            C=parameters["model__C"], class_weight="balanced",
            max_iter=500, random_state=10, **_logistic_parameters(result))),
    ]).fit(X_train, y_train)
    selected = X_train.columns[reference.named_steps["selector"].get_support()].tolist()
    if selected != result["final_selected_features"]:
        raise AssertionError("Binary selected features differ")
    pred = reference.predict(X_test)
    positive = reference.classes_[-1]
    truth = (y_test == positive).astype(int)
    score = reference.predict_proba(X_test)[:, -1]
    stored = pd.read_csv(FIGURES /
                         "Figure_3_heldout_predictions.csv", index_col=0).loc[X_test.index]
    if not np.array_equal(pred, stored["predicted_class"].to_numpy()):
        raise AssertionError("Binary patient predictions differ")
    if not np.allclose(score, stored["LUSC_score"].to_numpy(), rtol=0, atol=1e-10):
        raise AssertionError("Binary probabilities differ")
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(truth, score),
        "pr_auc": average_precision_score(truth, score),
    }
    return {name: _check("binary " + name, value, result["test_metrics"][name])
            for name, value in metrics.items()}


def regression_parity():
    result = json.loads((RESULTS / "regression.json").read_text(encoding="utf-8"))
    X_train, y_train, X_test, y_test = _matrix(
        "regression_example.csv", result["dataset"], "PURITY")
    y_train, y_test = y_train.astype(float), y_test.astype(float)
    parameters = result["final_best_params"]
    reference = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_regression, k=parameters["selector__k"])),
        ("model", Ridge(alpha=parameters["model__alpha"])),
    ]).fit(X_train, y_train)
    selected = X_train.columns[reference.named_steps["selector"].get_support()].tolist()
    if selected != result["final_selected_features"]:
        raise AssertionError("Regression selected features differ")
    pred = reference.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }
    return {name: _check("regression " + name, value, result["test_metrics"][name])
            for name, value in metrics.items()}


def multiclass_parity():
    result = json.loads((RESULTS / "multiclass_classification.json").read_text(encoding="utf-8"))
    X_train, y_train, X_test, y_test = _matrix(
        "multiclass_classification_example.csv", result["dataset"], "label")
    parameters = result["final_best_params"]
    reference = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=parameters["selector__k"])),
        ("model", LogisticRegression(
            C=parameters["model__C"], class_weight="balanced",
            max_iter=500, random_state=10, **_logistic_parameters(result))),
    ]).fit(X_train, y_train)
    selected = X_train.columns[reference.named_steps["selector"].get_support()].tolist()
    if selected != result["final_selected_features"]:
        raise AssertionError("Multiclass selected features differ")
    classes = reference.classes_
    prediction = reference.predict(X_test)
    probabilities = reference.predict_proba(X_test)
    truth = label_binarize(y_test, classes=classes)
    metrics = {
        "accuracy": float(np.mean(prediction == y_test.to_numpy())),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "roc_auc_ovr_macro": float(np.mean([
            roc_auc_score(truth[:, i], probabilities[:, i]) for i in range(len(classes))])),
        "pr_auc_macro": float(np.mean([
            average_precision_score(truth[:, i], probabilities[:, i])
            for i in range(len(classes))])),
    }
    observed_cm = confusion_matrix(y_test, prediction, labels=classes)
    evidence = {name: _check("multiclass " + name, value, result["test_metrics"][name])
                for name, value in metrics.items()}
    evidence["class_order"] = list(map(str, classes))
    evidence["confusion_matrix"] = observed_cm.tolist()
    return evidence


def _survival_input(metadata):
    path = EXAMPLES / "survival_example.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise AssertionError("Survival input checksum changed")
    raw = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
    audit = metadata["patient_level_audit"]
    excluded = set()
    for row in audit["excluded_cross_partition_patients"]:
        excluded.update(row["train_samples"] + row["test_samples"])
    for row in audit["collapsed_within_partition_patients"]:
        excluded.update(set(row["samples"]) - {row["selected"]})
    raw = raw.drop(columns=list(excluded))
    split = raw.loc["set"].str.strip().str.lower()
    features = raw.drop(index=["Status", "time", "set"]).T.apply(pd.to_numeric)
    outcome = np.empty(len(raw.columns), dtype=[("event", "?"), ("time", "<f8")])
    outcome["event"] = raw.loc["Status"].astype(float).to_numpy() == 1
    outcome["time"] = raw.loc["time"].astype(float).to_numpy()
    train, test = split.eq("training").to_numpy(), split.eq("testing").to_numpy()
    if (int(train.sum()), int(test.sum())) != (metadata["n_train"], metadata["n_test"]):
        raise AssertionError("Survival split count mismatch")
    return features.iloc[train], outcome[train], features.iloc[test], outcome[test]


def _cox_score_ranking(scaled, outcome, k):
    """Direct Cox score test at beta=0, coded independently of MALER helpers."""
    order = np.argsort(outcome["time"], kind="mergesort")[::-1]
    values = np.asarray(scaled)[order]
    times = outcome["time"][order]
    events = outcome["event"][order]
    numerator = np.zeros(values.shape[1])
    variance = np.zeros(values.shape[1])
    for time in np.unique(times):
        risk = values[times >= time]
        event_values = values[(times == time) & events]
        if len(event_values):
            numerator += np.sum(event_values - np.mean(risk, axis=0), axis=0)
            variance += len(event_values) * np.var(risk, axis=0)
    score = np.abs(numerator) / np.sqrt(variance + 0.1)
    return np.sort(np.argsort(score)[::-1][:int(k)])


def survival_parity():
    result = json.loads((RESULTS / "survival.json").read_text(encoding="utf-8"))
    X_train, y_train, X_test, y_test = _survival_input(result["dataset"])
    params = result["final_best_params"]
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    train_scaled = scaler.transform(imputer.transform(X_train))
    test_scaled = scaler.transform(imputer.transform(X_test))
    selected_index = _cox_score_ranking(train_scaled, y_train, params["selector__k"])
    selected = X_train.columns[selected_index].tolist()
    if selected != result["final_selected_features"]:
        raise AssertionError("Survival feature ranking differs")
    reference = SurvivalTree(
        random_state=10, min_samples_leaf=10,
        max_depth=params["model__max_depth"]).fit(train_scaled[:, selected_index], y_train)
    risk = reference.predict(test_scaled[:, selected_index])
    observed = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    return {"c_index": _check("survival c_index", observed,
                               result["test_metrics"]["c_index"]),
            "event_encoding": "1=event, 0=censored", "risk_direction": "higher=worse",
            "n_test": len(y_test)}


def main():
    evidence = {
        "protocol": "Independent direct scikit-learn refit using archived patient exclusions, best parameters, input hashes, and held-out partitions; no MALER pipeline/metric helpers imported.",
        "binary_classification": binary_parity(),
        "multiclass_classification": multiclass_parity(),
        "regression": regression_parity(),
        "survival": survival_parity(),
        "tolerance": 1e-10,
    }
    path = RESULTS / "reference_parity_check.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
