"""Versioned estimator and search-space registry for MALER.

The legacy UI historically duplicated defaults across templates and large view
modules.  This data-only registry is the canonical reproducibility record for
the revision and is intentionally importable without loading optional model
packages.
"""

from __future__ import absolute_import

import importlib


MODEL_REGISTRY_VERSION = 1
RANDOM_STATE = 10


MODEL_REGISTRY = {
    "classification": {
        "primary_metric": "balanced_accuracy",
        "splitter": "RepeatedStratifiedKFold",
        "models": {
            "naive_bayes": {
                "class": "sklearn.naive_bayes.GaussianNB",
                "requires_scaling": False,
                "defaults": {"var_smoothing": 1e-9},
                "grid": {"var_smoothing": [1e-11, 1e-9, 1e-7]},
            },
            "svm": {
                "class": "sklearn.svm.SVC",
                "requires_scaling": True,
                "defaults": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True},
                "grid": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
            },
            "logistic_regression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "requires_scaling": True,
                "defaults": {"C": 1.0, "solver": "liblinear", "max_iter": 4000, "random_state": RANDOM_STATE},
                "grid": {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
            },
            "knn": {
                "class": "sklearn.neighbors.KNeighborsClassifier",
                "requires_scaling": True,
                "defaults": {"n_neighbors": 5, "weights": "uniform", "p": 2},
                "grid": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"], "p": [1, 2]},
            },
            "decision_tree": {
                "class": "sklearn.tree.DecisionTreeClassifier",
                "requires_scaling": False,
                "defaults": {"random_state": RANDOM_STATE},
                "grid": {"max_depth": [None, 3, 5, 10], "min_samples_leaf": [1, 3, 5, 10]},
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "requires_scaling": False,
                "defaults": {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]},
            },
            "adaboost": {
                "class": "sklearn.ensemble.AdaBoostClassifier",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE},
                "grid": {"n_estimators": [50, 100, 250], "learning_rate": [0.03, 0.1, 0.3]},
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingClassifier",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1], "max_depth": [1, 2, 3]},
            },
            "xgboost": {
                "class": "xgboost.XGBClassifier",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.3, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1, 0.3], "max_depth": [3, 6]},
            },
            "lightgbm": {
                "class": "lightgbm.LGBMClassifier",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63]},
            },
        },
    },
    "regression": {
        "primary_metric": "r2",
        "splitter": "RepeatedKFold",
        "models": {
            "linear_regression": {
                "class": "sklearn.linear_model.LinearRegression",
                "requires_scaling": False,
                "defaults": {},
                "grid": {"fit_intercept": [True, False]},
            },
            "svr": {
                "class": "sklearn.svm.SVR",
                "requires_scaling": True,
                "defaults": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                "grid": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"], "epsilon": [0.01, 0.1]},
            },
            "ridge": {
                "class": "sklearn.linear_model.Ridge",
                "requires_scaling": True,
                "defaults": {"alpha": 1.0, "random_state": RANDOM_STATE},
                "grid": {"alpha": [0.1, 1.0, 10.0, 100.0]},
            },
            "lasso": {
                "class": "sklearn.linear_model.Lasso",
                "requires_scaling": True,
                "defaults": {"alpha": 1.0, "max_iter": 5000, "random_state": RANDOM_STATE},
                "grid": {"alpha": [0.001, 0.01, 0.1, 1.0]},
            },
            "decision_tree": {
                "class": "sklearn.tree.DecisionTreeRegressor",
                "requires_scaling": False,
                "defaults": {"random_state": RANDOM_STATE},
                "grid": {"max_depth": [None, 3, 5, 10], "min_samples_leaf": [1, 3, 5, 10]},
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "requires_scaling": False,
                "defaults": {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5, 1.0]},
            },
            "adaboost": {
                "class": "sklearn.ensemble.AdaBoostRegressor",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE},
                "grid": {"n_estimators": [50, 100, 250], "learning_rate": [0.03, 0.1, 0.3]},
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingRegressor",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1], "max_depth": [1, 2, 3]},
            },
            "xgboost": {
                "class": "xgboost.XGBRegressor",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.3, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1, 0.3], "max_depth": [3, 6]},
            },
        },
    },
    "survival": {
        "primary_metric": "c_index",
        "splitter": "RepeatedStratifiedKFold(event indicator)",
        "models": {
            "coxph": {
                "class": "sksurv.linear_model.CoxPHSurvivalAnalysis",
                "requires_scaling": True,
                "defaults": {"alpha": 1.0},
                "grid": {"alpha": [0.1, 1.0, 10.0]},
            },
            "coxnet": {
                "class": "sksurv.linear_model.CoxnetSurvivalAnalysis",
                "requires_scaling": True,
                "defaults": {"l1_ratio": 0.5, "alpha_min_ratio": 0.01, "n_alphas": 50, "fit_baseline_model": True},
                "grid": {"l1_ratio": [0.1, 0.5, 0.9]},
            },
            "survival_svm": {
                "class": "sksurv.svm.FastSurvivalSVM",
                "requires_scaling": True,
                "defaults": {"alpha": 1.0, "random_state": RANDOM_STATE, "max_iter": 1000},
                "grid": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            },
            "survival_tree": {
                "class": "sksurv.tree.SurvivalTree",
                "requires_scaling": False,
                "defaults": {"min_samples_leaf": 10, "random_state": RANDOM_STATE},
                "grid": {"max_depth": [2, 3, 5], "min_samples_leaf": [5, 10, 20]},
            },
            "extra_survival_trees": {
                "class": "sksurv.ensemble.ExtraSurvivalTrees",
                "requires_scaling": False,
                "defaults": {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"min_samples_leaf": [3, 5, 10], "max_features": ["sqrt", 0.5]},
            },
            "random_survival_forest": {
                "class": "sksurv.ensemble.RandomSurvivalForest",
                "requires_scaling": False,
                "defaults": {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1},
                "grid": {"min_samples_leaf": [5, 10, 20], "max_features": ["sqrt", 0.5]},
            },
            "gradient_boosting_survival": {
                "class": "sksurv.ensemble.GradientBoostingSurvivalAnalysis",
                "requires_scaling": False,
                "defaults": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE},
                "grid": {"n_estimators": [100, 250], "learning_rate": [0.03, 0.1], "max_depth": [1, 2]},
            },
        },
    },
}


FEATURE_REDUCTION = {
    "none": {"description": "No feature reduction", "parameters": {}},
    "select_k_best": {
        "description": "Fold-local univariate task-specific ranking",
        "parameters": {"k": [10, 20, 50, 100]},
    },
    "pca": {
        "description": "Fold-local principal component analysis; components are not gene signatures",
        "parameters": {"n_components": [10, 20, 50]},
    },
    "mrmr": {
        "description": "Fold-local minimum-redundancy maximum-relevance selection",
        "parameters": {"k": [10, 20, 50]},
    },
    "fss": {
        "description": "Fold-local forward sequential selection after a 50-feature prescreen",
        "parameters": {"k": [5, 10, 20]},
    },
    "bss": {
        "description": "Fold-local backward sequential selection after a 50-feature prescreen",
        "parameters": {"k": [5, 10, 20]},
    },
}


UI_MODEL_ALIASES = {
    "classification": {
        "naivebayes": "naive_bayes", "svm": "svm", "logistic": "logistic_regression",
        "logisticregression": "logistic_regression", "knn": "knn",
        "decisiontree": "decision_tree", "randomforest": "random_forest",
        "adaboost": "adaboost", "gradientboost": "gradient_boosting",
        "gradientboosting": "gradient_boosting", "xgboost": "xgboost", "lightgbm": "lightgbm",
    },
    "regression": {
        "linearregression": "linear_regression", "regsvm": "svr", "svr": "svr",
        "ridge": "ridge", "lasso": "lasso", "regdecisiontree": "decision_tree",
        "decisiontree": "decision_tree", "regrandomforest": "random_forest",
        "randomforest": "random_forest", "regadaboost": "adaboost", "adaboost": "adaboost",
        "gradientboost": "gradient_boosting", "gradientboosting": "gradient_boosting",
        "regxgboost": "xgboost", "xgboost": "xgboost",
    },
    "survival": {
        "coxph": "coxph", "coxnet": "coxnet", "survivalsvm": "survival_svm",
        "survivaltree": "survival_tree", "extrasurvivaltrees": "extra_survival_trees",
        "randomsurvivalforest": "random_survival_forest",
        "gradientboostingsurvival": "gradient_boosting_survival",
    },
}


def normalize_model_key(task, key):
    normalized = str(key or "").strip().lower().replace("-", "").replace("_", "")
    aliases = UI_MODEL_ALIASES.get(task, {})
    if normalized in aliases:
        return aliases[normalized]
    if key in MODEL_REGISTRY.get(task, {}).get("models", {}):
        return key
    raise KeyError("Unknown %s model: %s" % (task, key))


def get_model_spec(task, key):
    canonical = normalize_model_key(task, key)
    return canonical, MODEL_REGISTRY[task]["models"][canonical]


def build_estimator(task, key):
    """Instantiate a trusted estimator from the versioned registry."""
    canonical, spec = get_model_spec(task, key)
    module_name, class_name = spec["class"].rsplit(".", 1)
    estimator_class = getattr(importlib.import_module(module_name), class_name)
    return canonical, estimator_class(**dict(spec.get("defaults", {})))


def pipeline_param_grid(task, key):
    canonical, spec = get_model_spec(task, key)
    return canonical, {"model__" + name: values for name, values in spec.get("grid", {}).items()}
