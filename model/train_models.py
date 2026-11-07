"""
train_models.py
---------------
Step 2 of the pipeline: train the five classifiers required by the assignment on
the Telco Customer Churn training split, evaluate them on the untouched hold-out
test split and persist everything the Streamlit app needs.

Models (all from the course syllabus)
    1. Logistic Regression
    2. Decision Tree Classifier     (entropy / information gain)
    3. k-Nearest Neighbour          (instance based learning)
    4. Gaussian Naive Bayes         (probabilistic generative)
    5. Support Vector Machine       (RBF kernel trick)

Every model is wrapped in a Pipeline whose first stage is a ColumnTransformer
that standardises the three numeric features and one-hot encodes the sixteen
categorical features, so the exact same preprocessing is replayed at prediction
time inside the app from the raw, human-readable CSV.

Run from the repository root (after prepare_data.py):
    python model/train_models.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_FOLDS = 5
TARGET = "Churn"
POS_LABEL = "Yes"  # "the customer churned" is the class we care about

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Key -> (display name, estimator, hyper-parameter grid searched with 5-fold CV).
# class_weight="balanced" is offered to the three models that support it because
# the data is imbalanced (~27% churn), and recall on the churn class matters.
MODEL_SPECS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
        {
            "clf__max_depth": [None, 5, 10, 15],
            "clf__min_samples_leaf": [1, 5, 20],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "knn": (
        "kNN",
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    ),
    "svm": (
        "Support Vector Machine",
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {"clf__C": [1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
}

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# F1 on the churn class; the default "f1" scorer assumes an integer positive
# label, but our target is the string "Yes".
F1_SCORER = make_scorer(f1_score, pos_label=POS_LABEL)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Standardise numeric columns, one-hot encode categorical columns."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def evaluate(fitted_pipeline, X_test, y_test) -> dict:
    """Compute the six evaluation metrics required by the assignment.

    The task is binary (churn / no churn), so AUC is the standard one-dimensional
    ROC AUC on the churn-class probability, and precision / recall / F1 report the
    positive (churn) class, which is the business-relevant one.
    """
    y_pred = fitted_pipeline.predict(X_test)
    pos_idx = list(fitted_pipeline.classes_).index(POS_LABEL)
    y_proba = fitted_pipeline.predict_proba(X_test)[:, pos_idx]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test == POS_LABEL, y_proba),
        "Precision": precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "F1": f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main() -> None:
    train_df = pd.read_csv(os.path.join(ROOT, "data", "train_data.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test_data.csv"))

    feature_names = [c for c in train_df.columns if c != TARGET]
    X_train, y_train = train_df[feature_names], train_df[TARGET]
    X_test, y_test = test_df[feature_names], test_df[TARGET]

    print(f"Training samples : {X_train.shape[0]}  |  features: {X_train.shape[1]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Classes          : {sorted(y_train.unique())}  (positive = {POS_LABEL})\n")

    rows, registry = [], {}

    for key, (display_name, estimator, grid) in MODEL_SPECS.items():
        print(f"--- {display_name} ---")
        pipeline = Pipeline(
            [("prep", build_preprocessor(X_train)), ("clf", estimator)]
        )

        search = GridSearchCV(
            pipeline, grid, cv=CV_FOLDS, scoring=F1_SCORER, n_jobs=-1, refit=True
        )
        started = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - started

        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        metrics = evaluate(search.best_estimator_, X_test, y_test)

        print(f"best params      : {best_params}")
        print(f"CV f1            : {search.best_score_:.4f}")
        print(
            "test metrics     : "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        print(f"fit time         : {elapsed:.1f}s\n")

        joblib.dump(search.best_estimator_, os.path.join(HERE, f"{key}.joblib"))
        rows.append({"Model": display_name, **metrics})
        registry[key] = {
            "display_name": display_name,
            "file": f"{key}.joblib",
            "best_params": {k: str(v) for k, v in best_params.items()},
            "cv_f1": round(float(search.best_score_), 4),
            "fit_seconds": round(elapsed, 1),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        }

    results = pd.DataFrame(rows).set_index("Model")[METRIC_COLUMNS]
    results.to_csv(os.path.join(HERE, "model_metrics.csv"))

    meta = {
        "dataset": "Telco Customer Churn",
        "target": TARGET,
        "positive_label": POS_LABEL,
        "feature_names": feature_names,
        "classes": sorted(y_train.unique().tolist()),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
        "averaging": f"binary (positive class = {POS_LABEL})",
        "models": registry,
        "winner": results["F1"].idxmax(),
    }
    with open(os.path.join(HERE, "model_registry.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("=" * 78)
    print("COMPARISON TABLE (hold-out test set, n = %d)" % X_test.shape[0])
    print("=" * 78)
    print(results.round(4).to_string())
    print(f"\nBest macro-F1: {meta['winner']}")
    print("\nSaved: model/*.joblib, model/model_metrics.csv, model/model_registry.json")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
