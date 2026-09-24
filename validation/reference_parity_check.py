"""Independent scikit-learn refit of two locked MALER held-out models.

The patient exclusions, selected hyperparameters and input hash come from the
archived result metadata. No MALER pipeline or metric helper is imported here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
EXAMPLES = ROOT / "mlserver" / "static" / "cache" / "example"


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
            C=parameters["model__C"], solver="liblinear", class_weight="balanced",
            max_iter=500, random_state=10, multi_class="ovr")),
    ]).fit(X_train, y_train)
    selected = X_train.columns[reference.named_steps["selector"].get_support()].tolist()
    if selected != result["final_selected_features"]:
        raise AssertionError("Binary selected features differ")
    pred = reference.predict(X_test)
    positive = reference.classes_[-1]
    truth = (y_test == positive).astype(int)
    score = reference.predict_proba(X_test)[:, -1]
    stored = pd.read_csv(ROOT / "validation" / "figures" /
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


def main():
    evidence = {
        "protocol": "Independent direct scikit-learn refit using archived patient exclusions, best parameters, input hashes, and held-out partitions; no MALER pipeline/metric helpers imported.",
        "binary_classification": binary_parity(),
        "regression": regression_parity(),
        "tolerance": 1e-10,
    }
    path = RESULTS / "reference_parity_check.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
