"""Reproducible Windows capacity benchmarks for the legacy MALER runtime.

Each case runs in a fresh subprocess so Windows PeakWorkingSetSize records an
isolated peak.  The benchmark reports observed performance, not a universal
maximum supported dimension.
"""

from __future__ import absolute_import, print_function

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sksurv.tree import SurvivalTree


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import (  # noqa: E402
    build_classification_pipeline,
    nested_cv_classification,
    nested_cv_regression,
)
from mlserver.safe_survival import nested_cv_survival, validate_survival_target  # noqa: E402


RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "validation", "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "capacity_benchmark.json")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "capacity_benchmark.csv")
CHECKPOINT_JSON = OUTPUT_JSON + ".partial"
RANDOM_STATE = 10


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_counters():
    if os.name != "nt":
        return {}
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb)
    if not ok:
        return {}
    divisor = 1024.0 * 1024.0
    return {
        "peak_working_set_mb": counters.PeakWorkingSetSize / divisor,
        "working_set_mb": counters.WorkingSetSize / divisor,
        "peak_pagefile_mb": counters.PeakPagefileUsage / divisor,
        "private_mb": counters.PrivateUsage / divisor,
    }


def _classification_data(n_samples, n_features):
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    signal = X[:, : min(10, n_features)].sum(axis=1) + rng.normal(scale=1.0, size=n_samples)
    y = (signal > np.median(signal)).astype(int)
    return pd.DataFrame(X, columns=["g%d" % index for index in range(n_features)]), y


def _regression_data(n_samples, n_features):
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    coefficients = np.linspace(1.0, 0.2, min(10, n_features))
    y = X[:, :len(coefficients)].dot(coefficients) + rng.normal(scale=0.5, size=n_samples)
    return pd.DataFrame(X, columns=["g%d" % index for index in range(n_features)]), y


def _survival_data(n_samples, n_features):
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    linear = X[:, : min(5, n_features)].sum(axis=1) / max(1.0, np.sqrt(min(5, n_features)))
    event_time = np.exp(5.0 - 0.5 * linear + rng.normal(scale=0.4, size=n_samples))
    censor_time = rng.exponential(scale=np.median(event_time) * 2.0, size=n_samples)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    y = validate_survival_target(event, observed)
    return pd.DataFrame(X, columns=["g%d" % index for index in range(n_features)]), y


CASES = {
    "classification_small": ("classification", 200, 100),
    "classification_medium": ("classification", 500, 1000),
    "classification_large": ("classification", 1000, 5000),
    "regression_medium": ("regression", 500, 2000),
    "survival_medium": ("survival", 300, 1000),
    "prediction_throughput": ("prediction", 10000, 100),
}


def run_case(case_name):
    task, n_samples, n_features = CASES[case_name]
    started = time.perf_counter()
    result = {
        "case": case_name,
        "task": task,
        "n_samples": n_samples,
        "n_features": n_features,
        "input_matrix_mb": n_samples * n_features * 8.0 / (1024.0 * 1024.0),
    }
    if task == "classification":
        X, y = _classification_data(n_samples, n_features)
        validation_started = time.perf_counter()
        cv_result = nested_cv_classification(
            X, y, LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
            param_grid={"model__C": [1.0]}, k_values=(min(20, n_features),),
            outer_splits=3, outer_repeats=1, inner_splits=2, random_state=RANDOM_STATE,
        )
        result["analysis_seconds"] = time.perf_counter() - validation_started
        result["primary_metric"] = cv_result["summary"]["balanced_accuracy"]["mean"]
    elif task == "regression":
        X, y = _regression_data(n_samples, n_features)
        validation_started = time.perf_counter()
        cv_result = nested_cv_regression(
            X, y, Ridge(alpha=1.0), param_grid={"model__alpha": [1.0]},
            k_values=(min(20, n_features),), outer_splits=3, outer_repeats=1,
            inner_splits=2, random_state=RANDOM_STATE,
        )
        result["analysis_seconds"] = time.perf_counter() - validation_started
        result["primary_metric"] = cv_result["summary"]["r2"]["mean"]
    elif task == "survival":
        X, y = _survival_data(n_samples, n_features)
        validation_started = time.perf_counter()
        cv_result = nested_cv_survival(
            X, y, SurvivalTree(random_state=RANDOM_STATE, min_samples_leaf=10),
            param_grid={"model__max_depth": [3]}, k_values=(min(20, n_features),),
            outer_splits=3, outer_repeats=1, inner_splits=2, random_state=RANDOM_STATE,
        )
        result["analysis_seconds"] = time.perf_counter() - validation_started
        result["primary_metric"] = cv_result["summary"]["c_index"]["mean"]
    else:
        X_train, y_train = _classification_data(500, n_features)
        X_predict, _ = _classification_data(n_samples, n_features)
        pipeline = build_classification_pipeline(
            LogisticRegression(solver="liblinear", random_state=RANDOM_STATE), k=20)
        pipeline.fit(X_train, y_train)
        model_path = os.path.join(tempfile.gettempdir(), "maler_capacity_model.joblib")
        joblib.dump(pipeline, model_path)
        result["model_file_mb"] = os.path.getsize(model_path) / (1024.0 * 1024.0)
        predict_started = time.perf_counter()
        prediction = pipeline.predict(X_predict)
        result["analysis_seconds"] = time.perf_counter() - predict_started
        result["predictions_per_second"] = len(prediction) / result["analysis_seconds"]

    if task != "prediction":
        csv_started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            csv_path = handle.name
        try:
            X.to_csv(csv_path, index=False)
            result["csv_size_mb"] = os.path.getsize(csv_path) / (1024.0 * 1024.0)
            read_started = time.perf_counter()
            parsed = pd.read_csv(csv_path)
            result["csv_parse_seconds"] = time.perf_counter() - read_started
            result["parsed_shape"] = list(parsed.shape)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
        result["csv_write_seconds"] = read_started - csv_started

    result["total_seconds"] = time.perf_counter() - started
    result.update(memory_counters())
    return result


def orchestrate():
    rows = []
    if os.path.isfile(CHECKPOINT_JSON) and not os.environ.get("MALER_VALIDATION_FORCE"):
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    completed_names = set(row["case"] for row in rows)
    for case_name in CASES:
        if case_name in completed_names:
            print("Loaded benchmark checkpoint", case_name, flush=True)
            continue
        print("Benchmark", case_name, flush=True)
        command = [sys.executable, os.path.abspath(__file__), "--case", case_name]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(CHECKPOINT_JSON, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "protocol": "fresh subprocess per case; 3-fold outer and 2-fold inner for training cases",
        "interpretation": "Observed single-process Windows performance; not a universal maximum.",
        "cases": rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print("Wrote", OUTPUT_JSON)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    arguments = parser.parse_args()
    if arguments.case:
        print(json.dumps(run_case(arguments.case), sort_keys=True))
    else:
        orchestrate()


if __name__ == "__main__":
    main()
