"""External audit of the application-controlled legacy survival example model.

The pickle is loaded only because it is shipped inside the trusted repository;
the public prediction endpoint continues to reject user-provided pickle files.
Historical feature-selection provenance is not reconstructed, so this result is
supporting evidence rather than the primary leakage-controlled benchmark.
"""

from __future__ import absolute_import, print_function

import json
import os
import sys

import joblib
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import file_sha256  # noqa: E402
from validation.run_reviewer_validation import load_survival, split_declared  # noqa: E402


EXAMPLE_DIR = os.path.join(REPOSITORY_ROOT, "mlserver", "static", "cache", "example")
MODEL_PATH = os.path.join(EXAMPLE_DIR, "Survival_ExtraSurvivalTrees.pkl")
OUTPUT_PATH = os.path.join(REPOSITORY_ROOT, "validation", "results", "legacy_survival_external_audit.json")
RANDOM_STATE = 10


def bootstrap_c_index(y, risk, repetitions=2000):
    rng = np.random.RandomState(RANDOM_STATE)
    estimates = []
    for _ in range(repetitions):
        index = rng.randint(0, len(y), len(y))
        try:
            estimates.append(float(concordance_index_censored(
                y["event"][index], y["time"][index], risk[index])[0]))
        except ValueError:
            continue
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def main():
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or artifact.get("method") != "model_sur":
        raise ValueError("Unexpected trusted example model structure.")
    model = artifact["model"]
    features = list(map(str, artifact["feature_names"]))
    X, y, split, source_path = load_survival("survival_example.csv")
    X_train, y_train, X_external, y_external = split_declared(X, y, split)
    missing = sorted(set(features) - set(X_external.columns))
    if missing:
        raise ValueError("Legacy model features are missing: %s" % ", ".join(missing))
    risk = np.asarray(model.predict(X_external.loc[:, features]), dtype=float)
    train_risk = np.asarray(model.predict(X_train.loc[:, features]), dtype=float)
    c_index = float(concordance_index_censored(
        y_external["event"], y_external["time"], risk)[0])
    lower = max(float(np.min(y_train["time"])), float(np.min(y_external["time"])))
    upper = min(float(np.max(y_train["time"])), float(np.max(y_external["time"])))
    event_times = y_external["time"][
        y_external["event"] & (y_external["time"] > lower) & (y_external["time"] < upper)
    ]
    times = np.unique(np.quantile(event_times, [0.25, 0.50, 0.75]))
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_external, risk, times)
    survival_functions = model.predict_survival_function(X_external.loc[:, features])
    probabilities = np.asarray([[fn(time) for time in times] for fn in survival_functions])
    ibs_mask = y_external["time"] < float(np.max(y_train["time"]))
    ibs = float(integrated_brier_score(
        y_train, y_external[ibs_mask], probabilities[ibs_mask], times))
    result = {
        "status": "supporting external audit; historical selection provenance not fully reconstructable",
        "trusted_repository_artifact": True,
        "user_pickle_policy": "user-provided pickle remains rejected",
        "model_path": MODEL_PATH,
        "model_sha256": file_sha256(MODEL_PATH),
        "data_path": source_path,
        "data_sha256": file_sha256(source_path),
        "model_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "features": features,
        "n_features": len(features),
        "training_samples": len(X_train),
        "external_samples": len(X_external),
        "external_events": int(y_external["event"].sum()),
        "training_c_index_descriptive": float(concordance_index_censored(
            y_train["event"], y_train["time"], train_risk)[0]),
        "external_metrics": {
            "c_index": c_index,
            "c_index_bootstrap_95_ci": bootstrap_c_index(y_external, risk),
            "time_auc_mean": float(mean_auc),
            "time_auc_times": [float(value) for value in times],
            "time_auc_values": [float(value) for value in auc],
            "integrated_brier_score": ibs,
            "integrated_brier_score_samples": int(ibs_mask.sum()),
        },
        "interpretation": (
            "The pre-existing model transports better than the de-novo single-tree reference, "
            "but its historical feature-selection provenance prevents use as the primary unbiased estimate."
        ),
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("Wrote", OUTPUT_PATH)
    print(json.dumps(result["external_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
