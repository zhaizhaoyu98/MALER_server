"""Create publication-ready static figures from validated MALER results."""

from __future__ import absolute_import

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve


VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(VALIDATION_DIR)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import build_cross_platform_classification_pipeline
from run_reviewer_validation import load_classification, split_declared


RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
FIGURE_DIR = os.path.join(
    os.path.dirname(REPOSITORY_ROOT), "article-review", "revision", "figures")
BLUE = "#276FBF"
ORANGE = "#E07A3F"
GOLD = "#C49A2C"
INK = "#24313D"
GREY = "#7A8793"
LIGHT_GREY = "#D8DEE4"


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GREY)
    axis.spines["bottom"].set_color(GREY)
    axis.tick_params(colors=INK, labelsize=9)
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def internal_performance_figure(all_results, equal_budget, pca_result):
    tasks = [
        ("binary_classification", "Binary classification", "balanced_accuracy"),
        ("multiclass_classification", "Multiclass classification", "balanced_accuracy"),
        ("regression", "Regression", "r2"),
        ("survival", "Survival", "c_index"),
    ]
    labels = [item[1] for item in tasks]
    y = np.arange(len(tasks))[::-1]
    nested_mean = []
    nested_sd = []
    baseline_mean = []
    pca_mean = []
    pca_sd = []
    heldout = []
    stability = []
    seconds = []
    baseline_keys = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "regression": "regression",
        "survival": "survival",
    }
    for key, _, metric in tasks:
        task = all_results["tasks"][key]
        nested_mean.append(task["nested_cv"]["summary"][metric]["mean"])
        nested_sd.append(task["nested_cv"]["summary"][metric]["std"])
        baseline_mean.append(
            equal_budget[baseline_keys[key]]["nested_cv"]["summary"][metric]["mean"])
        if key == "survival":
            pca_mean.append(np.nan)
            pca_sd.append(np.nan)
        else:
            pca_task = pca_result[baseline_keys[key]]
            pca_mean.append(pca_task["nested_cv"]["summary"][metric]["mean"])
            pca_sd.append(pca_task["nested_cv"]["summary"][metric]["std"])
        heldout.append(task["test_metrics"][metric])
        stability.append(task["feature_stability"]["mean_pairwise_jaccard"])
        seconds.append(task["performance"]["nested_cv_seconds"])

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), gridspec_kw={"width_ratios": [1.65, 1, 1]})
    axis = axes[0]
    axis.errorbar(
        nested_mean, y + 0.14, xerr=nested_sd, fmt="o", color=BLUE,
        ecolor=BLUE, capsize=3, markersize=6, label="Nested CV mean ± SD")
    axis.scatter(baseline_mean, y - 0.14, marker="s", s=38, facecolors="white",
                 edgecolors=GREY, linewidths=1.2, label="Matched-resampling no selection")
    pca_positions = np.asarray(y, dtype=float) - 0.28
    valid_pca = np.isfinite(pca_mean)
    axis.errorbar(
        np.asarray(pca_mean)[valid_pca], pca_positions[valid_pca],
        xerr=np.asarray(pca_sd)[valid_pca], fmt="^", color=GOLD,
        ecolor=GOLD, capsize=3, markersize=6, label="Fold-local PCA mean ± SD")
    axis.scatter(heldout, y, marker="D", s=32, color=ORANGE, label="Held-out test")
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlim(0.35, 1.02)
    axis.set_xlabel("Primary metric (balanced accuracy, R², or C-index)", color=INK)
    axis.set_title("A  Internal predictive performance", loc="left", color=INK,
                   fontweight="bold", y=1.10)
    axis.text(0.0, 1.025, "Matched resampling (not compute matched); PCA used 5 × 5",
              transform=axis.transAxes, fontsize=9, color=GREY)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(axis)

    axis = axes[1]
    axis.barh(y, stability, color=BLUE, edgecolor=INK, linewidth=0.5)
    for value, position in zip(stability, y):
        axis.text(value + 0.015, position, "%.2f" % value, va="center", fontsize=8, color=INK)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Mean pairwise Jaccard", color=INK)
    axis.set_title("B  Feature-set stability", loc="left", color=INK,
                   fontweight="bold", y=1.10)
    axis.text(0.0, 1.025, "Across 50 outer folds", transform=axis.transAxes,
              fontsize=9, color=GREY)
    style_axis(axis)

    axis = axes[2]
    axis.barh(y, seconds, color=GOLD, edgecolor=INK, linewidth=0.5)
    for value, position in zip(seconds, y):
        axis.text(value * 1.06, position, "%.1f min" % (value / 60.0),
                  va="center", fontsize=8, color=INK)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xscale("log")
    axis.set_xlabel("Elapsed seconds (log scale)", color=INK)
    axis.set_title("C  Computation time", loc="left", color=INK,
                   fontweight="bold", y=1.10)
    axis.text(0.0, 1.025, "Single process, Windows test environment",
              transform=axis.transAxes, fontsize=9, color=GREY)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S1_internal_validation.png"), dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S1_internal_validation.pdf"), bbox_inches="tight")
    plt.close(figure)


def _confusion_axis(axis, matrix, title, classes):
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > matrix.max() * 0.55 else INK
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color=color, fontsize=12, fontweight="bold")
    axis.set_xticks(range(len(classes)))
    axis.set_xticklabels(classes)
    axis.set_yticks(range(len(classes)))
    axis.set_yticklabels(classes)
    axis.set_xlabel("Predicted", color=INK)
    axis.set_ylabel("Observed", color=INK)
    axis.text(0.0, 1.12, title, transform=axis.transAxes, fontsize=12,
              color=INK, fontweight="bold", ha="left", va="bottom")
    axis.tick_params(colors=INK, labelsize=9)
    return image


def external_validation_figure(external_result):
    external_X = pd.read_csv(
        os.path.join(RESULTS_DIR, "gse37745_selected_expression.csv"), index_col=0)
    metadata = pd.read_csv(os.path.join(RESULTS_DIR, "gse37745_samples.csv"), index_col=0)
    y = metadata.loc[external_X.index, "label"].to_numpy()
    X, internal_y, split, _ = load_classification("binary_classification_example.csv")
    X_train, y_train, _, _ = split_declared(X, internal_y, split)
    mapped = external_result["feature_mapping"]["mapped_features"]
    estimator = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=500,
        random_state=10, C=external_result["best_params"]["model__C"])
    model = build_cross_platform_classification_pipeline(
        estimator, k=external_result["best_params"]["selector__k"])
    model.fit(X_train[mapped], y_train)
    classes = model.classes_
    primary_scores = model.predict_proba(external_X[mapped])[:, -1]
    primary_prediction = model.predict(external_X[mapped])

    selected = external_result["feature_mapping"]["final_model_selected_features"]
    train_z = (X_train[selected] - X_train[selected].mean(axis=0)) / X_train[selected].std(axis=0, ddof=0)
    external_selected = external_X[selected]
    external_z = (external_selected - external_selected.mean(axis=0)) / external_selected.std(axis=0, ddof=0)
    z_model = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=500,
        random_state=10, C=external_result["best_params"]["model__C"])
    z_model.fit(train_z, y_train)
    z_scores = z_model.predict_proba(external_z)[:, -1]
    z_prediction = z_model.predict(external_z)
    truth = (y == classes[-1]).astype(int)
    primary_fpr, primary_tpr, _ = roc_curve(truth, primary_scores)
    z_fpr, z_tpr, _ = roc_curve(truth, z_scores)
    primary_auc = external_result["external_primary_metrics"]["roc_auc"]
    primary_ci = external_result["external_primary_bootstrap"]["roc_auc"]["bootstrap_95_ci"]
    z_auc = external_result["sensitivity_cohort_zscore"]["external_metrics"]["roc_auc"]
    z_ci = external_result["sensitivity_cohort_zscore"]["external_bootstrap"]["roc_auc"]["bootstrap_95_ci"]

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.25), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    axis = axes[0]
    axis.plot(primary_fpr, primary_tpr, color=BLUE, linewidth=2.2,
              label="Sample rank: %.3f (95%% CI %.3f–%.3f)" % (
                  primary_auc, primary_ci[0], primary_ci[1]))
    axis.plot(z_fpr, z_tpr, color=ORANGE, linewidth=2.0, linestyle="--",
              label="Cohort z-score: %.3f (95%% CI %.3f–%.3f)" % (
                  z_auc, z_ci[0], z_ci[1]))
    axis.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate", color=INK)
    axis.set_ylabel("True-positive rate", color=INK)
    axis.set_title("A  External discrimination", loc="left", color=INK,
                   fontweight="bold", y=1.10)
    axis.text(0.0, 1.025, "GSE37745: LUAD n=106; LUSC n=66",
              transform=axis.transAxes, fontsize=9, color=GREY)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(axis)
    _confusion_axis(
        axes[1], confusion_matrix(y, primary_prediction, labels=classes),
        "B  Sample-rank threshold", classes)
    axes[1].text(0.0, 1.045, "Independent per-sample transform",
                 transform=axes[1].transAxes, fontsize=9, color=GREY)
    _confusion_axis(
        axes[2], confusion_matrix(y, z_prediction, labels=classes),
        "C  Z-score sensitivity", classes)
    axes[2].text(0.0, 1.045, "Unlabeled cohort distribution used",
                 transform=axes[2].transAxes, fontsize=9, color=GREY)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S2_external_GSE37745.png"), dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S2_external_GSE37745.pdf"), bbox_inches="tight")
    plt.close(figure)


def external_stability_figure(external_result):
    frequency = external_result["feature_stability"]["top_selection_frequency"][:20]
    frame = pd.DataFrame(frequency).sort_values("fraction")
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    y = np.arange(len(frame))
    axis.barh(y, frame["fraction"], color=BLUE, edgecolor=INK, linewidth=0.4)
    axis.set_yticks(y)
    axis.set_yticklabels(frame["feature"])
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Selection fraction across 50 outer folds", color=INK)
    axis.set_title("Cross-platform feature selection stability", loc="left", color=INK,
                   fontweight="bold", y=1.085)
    axis.text(0.0, 1.02, "GPL570-measurable feature universe; training outcomes only",
              transform=axis.transAxes, fontsize=9, color=GREY)
    for position, value in zip(y, frame["fraction"]):
        axis.text(min(value + 0.015, 0.96), position, "%.0f%%" % (value * 100),
                  va="center", fontsize=8, color=INK)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S3_external_feature_stability.png"), dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S3_external_feature_stability.pdf"), bbox_inches="tight")
    plt.close(figure)


def survival_expanded_figure(result):
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharex=True)
    panels = [
        (axes[0], result["gbsg2"], "all_heldout_audit_results",
         "A  GBSG2 development → declared holdout",
         "514 development; 172 held-out samples"),
        (axes[1], result["tcga_cgga"], "all_external_audit_results",
         "B  TCGA development → CGGA audit",
         "160 unique TCGA patients; 133 CGGA samples"),
    ]
    for axis, section, audit_key, title, subtitle in panels:
        nested = {row["name"]: row for row in section["candidate_results"]}
        audit = {row["name"]: row for row in section[audit_key]}
        names = sorted(nested, key=lambda name: nested[name]["summary"]["c_index"]["mean"])
        positions = np.arange(len(names))
        labels = [name.replace("_", " ") + (" *" if name == section["winner"] else "")
                  for name in names]
        means = [nested[name]["summary"]["c_index"]["mean"] for name in names]
        errors = [nested[name]["summary"]["c_index"]["std"] for name in names]
        heldout = [audit[name]["metrics"]["c_index"] for name in names]
        axis.errorbar(
            means, positions + 0.10, xerr=errors, fmt="o", color=BLUE,
            ecolor=BLUE, capsize=3, markersize=5,
            label="Nested CV mean ± SD")
        axis.scatter(
            heldout, positions - 0.10, marker="D", s=34, color=ORANGE,
            label="External/held-out audit")
        axis.axvline(0.5, color=GREY, linestyle=":", linewidth=1.1)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.set_xlim(0.42, 0.77)
        axis.set_xlabel("Harrell C-index", color=INK)
        axis.set_title(title, loc="left", color=INK, fontweight="bold", y=1.10)
        axis.text(0.0, 1.025, subtitle + "; * training-selected winner",
                  transform=axis.transAxes, fontsize=9, color=GREY)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=8, ncol=2,
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S4_survival_expanded.png"),
                   dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S4_survival_expanded.pdf"),
                   bbox_inches="tight")
    plt.close(figure)


def external_calibration_figure(gse37745_result, gse50081_result):
    external_X = pd.read_csv(
        os.path.join(RESULTS_DIR, "gse50081_selected_expression.csv"), index_col=0)
    metadata = pd.read_csv(os.path.join(RESULTS_DIR, "gse50081_samples.csv"), index_col=0)
    y = metadata.loc[external_X.index, "label"].to_numpy()
    X, internal_y, split, _ = load_classification("binary_classification_example.csv")
    X_train, y_train, _, _ = split_declared(X, internal_y, split)
    mapped = gse37745_result["feature_mapping"]["mapped_features"]
    estimator = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=500,
        random_state=10, C=gse37745_result["best_params"]["model__C"])
    model = build_cross_platform_classification_pipeline(
        estimator, k=gse37745_result["best_params"]["selector__k"])
    model.fit(X_train[mapped], y_train)
    classes = model.classes_
    probability = model.predict_proba(external_X[mapped])
    scores = probability[:, -1]
    default_prediction = model.predict(external_X[mapped])
    threshold = gse50081_result["gse37745_calibration"]["selected_threshold"]
    calibrated_prediction = np.where(scores >= threshold, classes[-1], classes[0])
    truth = (y == classes[-1]).astype(int)
    fpr, tpr, _ = roc_curve(truth, scores)
    auc = gse50081_result["gse50081_default_threshold"]["metrics"]["roc_auc"]
    auc_ci = gse50081_result["gse50081_default_threshold"]["bootstrap"]["roc_auc"]["bootstrap_95_ci"]

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.25),
                               gridspec_kw={"width_ratios": [1.35, 1, 1]})
    axis = axes[0]
    axis.plot(fpr, tpr, color=BLUE, linewidth=2.2,
              label="ROC-AUC %.3f (95%% CI %.3f–%.3f)" % (auc, auc_ci[0], auc_ci[1]))
    axis.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate", color=INK)
    axis.set_ylabel("True-positive rate", color=INK)
    axis.set_title("A  GSE50081 discrimination", loc="left", color=INK,
                   fontweight="bold", y=1.10)
    axis.text(0.0, 1.025, "GSE50081: %s n=%d; %s n=%d" % (
                  classes[0], int(np.sum(y == classes[0])),
                  classes[-1], int(np.sum(y == classes[-1]))),
              transform=axis.transAxes, fontsize=9, color=GREY)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(axis)
    _confusion_axis(
        axes[1], confusion_matrix(y, calibrated_prediction, labels=classes),
        "B  GSE37745-calibrated threshold", classes)
    axes[1].text(0.0, 1.045, "Threshold %.3f; balanced accuracy %.3f" % (
                     threshold, gse50081_result["gse50081_gse37745_calibrated_threshold"]["metrics"]["balanced_accuracy"]),
                 transform=axes[1].transAxes, fontsize=9, color=GREY)
    _confusion_axis(
        axes[2], confusion_matrix(y, default_prediction, labels=classes),
        "C  Default threshold 0.5", classes)
    axes[2].text(0.0, 1.045, "Balanced accuracy %.3f; specificity %.3f" % (
                     gse50081_result["gse50081_default_threshold"]["metrics"]["balanced_accuracy"],
                     gse50081_result["gse50081_default_threshold"]["metrics"]["specificity"]),
                 transform=axes[2].transAxes, fontsize=9, color=GREY)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S5_external_GSE50081.png"),
                   dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S5_external_GSE50081.pdf"),
                   bbox_inches="tight")
    plt.close(figure)


def capacity_figure(result):
    frame = pd.DataFrame(result["cases"])
    training = frame[frame["task"] != "prediction"].copy()
    training["label"] = training["case"].str.replace("_", " ")
    training = training.sort_values("peak_working_set_mb")
    y = np.arange(len(training))
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].barh(y, training["peak_working_set_mb"], color=BLUE,
                 edgecolor=INK, linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(training["label"])
    axes[0].set_xlim(0, max(training["peak_working_set_mb"]) * 1.18)
    axes[0].set_xlabel("Peak working set (MB)", color=INK)
    axes[0].set_title("A  Observed peak memory", loc="left", color=INK,
                      fontweight="bold", y=1.10)
    axes[0].text(0.0, 1.025, "Fresh Windows subprocess per case",
                 transform=axes[0].transAxes, fontsize=9, color=GREY)
    for value, position in zip(training["peak_working_set_mb"], y):
        axes[0].text(value + 5, position, "%.1f" % value, va="center", fontsize=8, color=INK)
    style_axis(axes[0])

    elapsed = training.sort_values("total_seconds")
    y2 = np.arange(len(elapsed))
    axes[1].barh(y2, elapsed["total_seconds"], color=GOLD,
                 edgecolor=INK, linewidth=0.5)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(elapsed["label"])
    axes[1].set_xlim(0, max(elapsed["total_seconds"]) * 1.22)
    axes[1].set_xlabel("Total elapsed time (seconds)", color=INK)
    axes[1].set_title("B  Fresh-process script elapsed time", loc="left", color=INK,
                      fontweight="bold", y=1.10)
    prediction = frame[frame["task"] == "prediction"].iloc[0]
    axes[1].text(0.0, 1.025, "In-process prediction: %s rows/s" %
                 format(int(round(prediction["predictions_per_second"])), ","),
                 transform=axes[1].transAxes, fontsize=9, color=GREY)
    for value, position in zip(elapsed["total_seconds"], y2):
        axes[1].text(value + 0.12, position, "%.2f" % value, va="center", fontsize=8, color=INK)
    style_axis(axes[1])
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S6_capacity.png"),
                   dpi=300, bbox_inches="tight")
    figure.savefig(os.path.join(FIGURE_DIR, "Figure_S6_capacity.pdf"),
                   bbox_inches="tight")
    plt.close(figure)


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    internal = load_json("validation_results.json")
    external = load_json("external_gse37745.json")
    equal_budget = load_json("equal_budget_no_selection_baselines.json")
    pca_result = load_json("pca_sensitivity.json")
    survival_result = load_json("survival_expanded_validation.json")
    gse50081 = load_json("external_gse50081.json")
    capacity = load_json("capacity_benchmark.json")
    internal_performance_figure(internal, equal_budget, pca_result)
    external_validation_figure(external)
    external_stability_figure(external)
    survival_expanded_figure(survival_result)
    external_calibration_figure(external, gse50081)
    capacity_figure(capacity)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
