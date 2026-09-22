"""Create the revised main Figure 3 from the locked MALER validation output.

The script intentionally uses the model bundle and result JSON already created by
``run_reviewer_validation.py``.  It verifies the bundle checksum and recomputed
held-out metrics before drawing anything, so the figure cannot silently drift
from the reported validation results.
"""

from __future__ import absolute_import, print_function

import hashlib
import io
import json
import os
import sys
import zipfile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(VALIDATION_DIR)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import _decision_scores, classification_metrics  # noqa: E402
from run_reviewer_validation import load_classification, split_declared  # noqa: E402


RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
FIGURE_DIR = os.path.join(VALIDATION_DIR, "figures")

BLUE = "#2369A8"
ORANGE = "#D55E00"
TEAL = "#15847B"
PURPLE = "#6F4C9B"
INK = "#24313D"
GREY = "#68757F"
LIGHT_GREY = "#D9E0E5"
PALE_BLUE = "#E8F1F8"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_locked_model(bundle_path, expected_sha256):
    observed_sha256 = _sha256(bundle_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Model bundle checksum mismatch: expected %s, observed %s"
            % (expected_sha256, observed_sha256)
        )
    with zipfile.ZipFile(bundle_path, "r") as archive:
        expected_members = {"manifest.json", "model.joblib", "signature.sha256"}
        if set(archive.namelist()) != expected_members:
            raise RuntimeError("Unexpected model bundle contents.")
        model_bytes = archive.read("model.joblib")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    # This is a trusted, checksum-locked local artifact produced by our own run.
    return joblib.load(io.BytesIO(model_bytes)), manifest, observed_sha256


def _assert_close(recomputed, stored):
    checked = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    failures = []
    for metric in checked:
        if not np.isclose(recomputed[metric], stored[metric], rtol=0.0, atol=1e-12):
            failures.append(
                "%s: recomputed %.15f != stored %.15f"
                % (metric, recomputed[metric], stored[metric])
            )
    if failures:
        raise RuntimeError("Held-out metric verification failed:\n" + "\n".join(failures))


def _style_axis(axis, grid_axis="both"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GREY)
    axis.spines["bottom"].set_color(GREY)
    axis.tick_params(colors=INK, labelsize=9)
    if grid_axis:
        axis.grid(axis=grid_axis, color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _panel_title(axis, letter, title, subtitle=None):
    axis.set_title(
        "%s  %s" % (letter, title), loc="left", color=INK,
        fontsize=11.5, fontweight="bold", pad=15,
    )
    if subtitle:
        axis.text(
            0.0, 1.02, subtitle, transform=axis.transAxes,
            fontsize=8.8, color=GREY, ha="left", va="bottom",
        )


def build_figure():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, "binary_classification.json")
    result = _load_json(result_path)
    bundle_path = os.path.join(RESULTS_DIR, "binary_classification_pipeline.maler")
    model, manifest, bundle_sha256 = _load_locked_model(
        bundle_path, result["model_bundle"]["sha256"]
    )

    X, y, split, source_path = load_classification("binary_classification_example.csv")
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    source_sha256 = _sha256(source_path)
    if source_sha256 != result["dataset"]["source_sha256"]:
        raise RuntimeError("Example dataset checksum does not match the locked result JSON.")
    if X_train.shape[0] != result["dataset"]["n_train"]:
        raise RuntimeError("Training sample count does not match the locked result JSON.")
    if X_test.shape[0] != result["dataset"]["n_test"]:
        raise RuntimeError("Test sample count does not match the locked result JSON.")

    prediction = model.predict(X_test)
    scores = _decision_scores(model, X_test)
    recomputed = classification_metrics(
        y_test, prediction, scores=scores, classes=model.classes_
    )
    _assert_close(recomputed, result["test_metrics"])

    classes = np.asarray(model.classes_)
    positive_class = classes[-1]
    positive_scores = scores[:, -1] if np.asarray(scores).ndim == 2 else scores
    binary_truth = (np.asarray(y_test) == positive_class).astype(int)
    fpr, tpr, _ = roc_curve(binary_truth, positive_scores)
    precision, recall, _ = precision_recall_curve(binary_truth, positive_scores)
    matrix = confusion_matrix(y_test, prediction, labels=classes)

    folds = pd.DataFrame(result["nested_cv"]["folds"])
    cv_metrics = [
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("roc_auc", "ROC–AUC"),
        ("pr_auc", "PR–AUC"),
        ("mcc", "MCC"),
    ]

    figure = plt.figure(figsize=(12.4, 8.4), facecolor="white")
    grid = figure.add_gridspec(
        2, 3, width_ratios=[1.18, 1.05, 1.05], height_ratios=[1.0, 1.0],
        wspace=0.36, hspace=0.42,
    )
    axis_a = figure.add_subplot(grid[0, :2])
    axis_b = figure.add_subplot(grid[0, 2])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1:])

    positions = np.arange(1, len(cv_metrics) + 1)
    data = [folds[key].to_numpy(dtype=float) for key, _ in cv_metrics]
    boxes = axis_a.boxplot(
        data, positions=positions, widths=0.52, patch_artist=True,
        showfliers=False, medianprops={"color": INK, "linewidth": 1.4},
        boxprops={"facecolor": PALE_BLUE, "edgecolor": BLUE, "linewidth": 1.1},
        whiskerprops={"color": BLUE, "linewidth": 1.0},
        capprops={"color": BLUE, "linewidth": 1.0},
    )
    del boxes
    rng = np.random.RandomState(10)
    for index, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        axis_a.scatter(
            np.full(len(values), index) + jitter, values,
            s=13, color=BLUE, alpha=0.48, edgecolors="none", zorder=2,
        )
        heldout = result["test_metrics"][cv_metrics[index - 1][0]]
        axis_a.scatter(
            [index], [heldout], marker="D", s=48, color=ORANGE,
            edgecolor="white", linewidth=0.7, zorder=4,
        )
    axis_a.scatter([], [], marker="o", s=25, color=BLUE, alpha=0.55,
                   label="Outer-fold estimate")
    axis_a.scatter([], [], marker="D", s=45, color=ORANGE,
                   label="Locked held-out test")
    axis_a.set_xticks(positions)
    axis_a.set_xticklabels([label for _, label in cv_metrics])
    axis_a.set_ylim(0.75, 1.015)
    axis_a.set_ylabel("Metric value", color=INK)
    axis_a.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    _panel_title(
        axis_a, "A", "Repeated nested cross-validation",
        "Development set n=312; 5 folds × 10 repeats; model selection confined to training folds",
    )
    _style_axis(axis_a, grid_axis="y")

    axis_b.plot(
        fpr, tpr, color=BLUE, linewidth=2.1,
        label="ROC–AUC %.3f" % result["test_metrics"]["roc_auc"],
    )
    axis_b.plot(
        recall, precision, color=ORANGE, linewidth=2.1,
        label="PR–AUC %.3f" % result["test_metrics"]["pr_auc"],
    )
    axis_b.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1.0)
    prevalence = float(binary_truth.mean())
    axis_b.axhline(prevalence, color=GREY, linestyle="--", linewidth=1.0)
    axis_b.set_xlim(0, 1)
    axis_b.set_ylim(0, 1.02)
    axis_b.set_xlabel("FPR (ROC) or recall (PR)", color=INK)
    axis_b.set_ylabel("TPR (ROC) or precision (PR)", color=INK)
    axis_b.legend(frameon=False, fontsize=8.2, loc="lower right")
    _panel_title(axis_b, "B", "Held-out discrimination", "Independent test n=302")
    _style_axis(axis_b, grid_axis="both")

    image = axis_c.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    del image
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > matrix.max() * 0.55 else INK
            axis_c.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                fontsize=16, fontweight="bold", color=color,
            )
    axis_c.set_xticks(range(len(classes)))
    axis_c.set_xticklabels(classes)
    axis_c.set_yticks(range(len(classes)))
    axis_c.set_yticklabels(classes)
    axis_c.set_xlabel("Predicted class", color=INK)
    axis_c.set_ylabel("Observed class", color=INK)
    axis_c.tick_params(colors=INK, labelsize=9)
    _panel_title(
        axis_c, "C", "Held-out confusion matrix",
        "LUAD n=153; LUSC n=149",
    )

    ci_metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC–AUC"),
        ("pr_auc", "PR–AUC"),
        ("recall_sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("mcc", "MCC"),
    ]
    y_positions = np.arange(len(ci_metrics))[::-1]
    estimates = []
    lower = []
    upper = []
    labels = []
    for key, label in ci_metrics:
        entry = result["test_bootstrap"][key]
        estimates.append(entry["estimate"])
        lower.append(entry["bootstrap_95_ci"][0])
        upper.append(entry["bootstrap_95_ci"][1])
        labels.append(label)
    estimates = np.asarray(estimates)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    axis_d.errorbar(
        estimates, y_positions,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o", color=TEAL, ecolor=TEAL, capsize=3.5,
        markersize=6, linewidth=1.4,
    )
    for estimate, high, y_position in zip(estimates, upper, y_positions):
        axis_d.text(
            min(high + 0.012, 1.004), y_position,
            "%.3f" % estimate, va="center", ha="left", fontsize=8.4, color=INK,
        )
    axis_d.set_yticks(y_positions)
    axis_d.set_yticklabels(labels)
    axis_d.set_xlim(0.82, 1.02)
    axis_d.set_xlabel("Estimate with 95% bootstrap confidence interval", color=INK)
    _panel_title(
        axis_d, "D", "Held-out performance uncertainty",
        "1,000 bootstrap resamples; estimates were not used for model selection",
    )
    _style_axis(axis_d, grid_axis="x")

    figure.text(
        0.015, 0.012,
        "Model: class-weighted logistic regression; fold-local median imputation, z-score scaling, "
        "and univariate feature selection. Positive class for ROC/PR: LUSC.",
        fontsize=8.2, color=GREY, ha="left", va="bottom",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.085)

    png_path = os.path.join(FIGURE_DIR, "Figure_3_results.png")
    pdf_path = os.path.join(FIGURE_DIR, "Figure_3_results.pdf")
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)

    prediction_frame = pd.DataFrame(
        {
            "sample_id": X_test.index.astype(str),
            "observed_class": y_test,
            "predicted_class": prediction,
            "lusq_score": positive_scores,
        }
    )
    prediction_csv = os.path.join(FIGURE_DIR, "Figure_3_heldout_predictions.csv")
    # Keep the public class label in the column name without changing old files.
    prediction_frame = prediction_frame.rename(columns={"lusq_score": "LUSC_score"})
    prediction_frame.to_csv(prediction_csv, index=False)

    audit = {
        "bundle_sha256": bundle_sha256,
        "dataset_sha256": source_sha256,
        "manifest": manifest,
        "recomputed_test_metrics": recomputed,
        "stored_test_metrics": result["test_metrics"],
        "metric_tolerance": 1e-12,
        "figure_png": os.path.relpath(png_path, REPOSITORY_ROOT).replace(os.sep, "/"),
        "figure_pdf": os.path.relpath(pdf_path, REPOSITORY_ROOT).replace(os.sep, "/"),
        "prediction_csv": os.path.relpath(prediction_csv, REPOSITORY_ROOT).replace(os.sep, "/"),
    }
    audit_path = os.path.join(FIGURE_DIR, "Figure_3_generation_audit.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print(json.dumps({"png": png_path, "pdf": pdf_path, "audit": audit_path}, indent=2))


if __name__ == "__main__":
    build_figure()
