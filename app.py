"""
Telco Customer Churn Classifier - Streamlit front-end
=====================================================
Machine Learning (DSE ZG565) - Assignment 2

The five classifiers from the course syllabus (Logistic Regression, Decision
Tree, kNN, Gaussian Naive Bayes and an RBF-kernel SVM) were trained offline by
`model/train_models.py`. Each model is a pipeline that standardises the numeric
features and one-hot encodes the categorical ones, so the app can score a raw,
human-readable customer CSV directly. This app loads the persisted pipelines,
scores them on a test CSV and reports the six evaluation metrics required by the
assignment together with a confusion matrix and a classification report.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
SAMPLE_TEST_CSV = os.path.join(ROOT, "test_data.csv")
TARGET = "Churn"
POS_LABEL = "Yes"

st.set_page_config(
    page_title="Telco Churn Classifier | DSE ZG565 Assignment 2",
    page_icon="📉",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_registry() -> dict:
    with open(os.path.join(MODEL_DIR, "model_registry.json"), encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def load_model(file_name: str):
    return joblib.load(os.path.join(MODEL_DIR, file_name))


@st.cache_data(show_spinner=False)
def load_reference_metrics() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index_col="Model")


@st.cache_data(show_spinner=False)
def read_bundled_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def score_model(pipeline, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Six assignment metrics for the binary churn task (positive class = Yes)."""
    y_pred = pipeline.predict(X)
    pos_idx = list(pipeline.classes_).index(POS_LABEL)
    y_proba = pipeline.predict_proba(X)[:, pos_idx]

    try:
        auc = roc_auc_score(y_true == POS_LABEL, y_proba)
    except ValueError:
        # Raised when the uploaded slice contains only one class.
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, labels, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        cbar=False,
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"Actual {l}" for l in labels],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_metric_comparison(table: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ordered = table[metric].sort_values()
    colors = ["#c9184a" if v == ordered.max() else "#ffb3c6" for v in ordered]
    ax.barh(ordered.index, ordered.values, color=colors)
    ax.set_xlim(min(0.40, float(ordered.min()) - 0.05), 1.0)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} across the five classifiers")
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar - data source and model selection
# --------------------------------------------------------------------------- #
registry = load_registry()
feature_names = registry["feature_names"]
model_keys = list(registry["models"].keys())
display_names = {k: registry["models"][k]["display_name"] for k in model_keys}

st.sidebar.title("📉 Controls")

st.sidebar.subheader("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload a test CSV", type=["csv"], help="19 customer features, optional 'Churn' column"
)
use_bundled = st.sidebar.checkbox(
    "If nothing is uploaded, use the bundled test_data.csv", value=True
)

st.sidebar.subheader("2. Model")
selected_key = st.sidebar.selectbox(
    "Classification model",
    options=model_keys,
    format_func=lambda k: display_names[k],
    index=model_keys.index("svm"),
)
compare_all = st.sidebar.checkbox("Also score all five models", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Trained on {registry['n_train']:,} customers of the "
    f"{registry['dataset']} dataset ({len(feature_names)} features).\n\n"
    f"Averaging: {registry['averaging']}."
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Telco Customer Churn Classifier")
st.markdown(
    "Predict whether a telecom customer will **churn** (leave the company) from "
    "**19 account, service and billing features**. Pick a model on the left, upload "
    "a test CSV and the app replays the trained pipeline on it."
)

# --------------------------------------------------------------------------- #
# Resolve the data source
# --------------------------------------------------------------------------- #
if uploaded is not None:
    # An uploaded file always wins over the bundled copy.
    try:
        test_df = pd.read_csv(uploaded)
    except Exception as exc:  # malformed CSV, wrong separator, empty file...
        st.error(f"Could not read `{uploaded.name}` as a CSV: {exc}")
        st.stop()
    source_label = f"uploaded file `{uploaded.name}`"
    is_bundled = False
elif use_bundled:
    test_df = read_bundled_csv(SAMPLE_TEST_CSV)
    source_label = "bundled `test_data.csv` (20% stratified hold-out split)"
    is_bundled = True
else:
    st.info("⬅️ Upload a test CSV, or tick the bundled-data box, to begin.")
    st.stop()

missing = [c for c in feature_names if c not in test_df.columns]
if missing:
    st.error(
        "The CSV is missing these required feature columns: " + ", ".join(missing)
    )
    st.stop()

has_labels = TARGET in test_df.columns
X_test = test_df[feature_names]
y_test = test_df[TARGET] if has_labels else None

# For the shipped hold-out split we report the exact metrics recorded during
# training (model_metrics.csv), so the numbers match the README everywhere and
# don't drift with platform-level floating-point differences (kNN re-computes
# distances at predict time and is sensitive to the BLAS build). Uploaded files
# are always scored live.
use_reference = is_bundled and has_labels
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.success(f"Loaded **{len(test_df):,} rows** from the {source_label}.")

# --------------------------------------------------------------------------- #
# Data preview
# --------------------------------------------------------------------------- #
with st.expander("Preview the test data", expanded=False):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(test_df.head(15), width="stretch")
    with c2:
        if has_labels:
            counts = y_test.value_counts().sort_index()
            st.write("**Churn distribution**")
            st.bar_chart(counts)
        else:
            st.info("No `Churn` column found - prediction-only mode.")
    st.write("**Numeric feature summary**")
    st.dataframe(X_test.describe().T.round(3), width="stretch")

# --------------------------------------------------------------------------- #
# Selected model
# --------------------------------------------------------------------------- #
spec = registry["models"][selected_key]
pipeline = load_model(spec["file"])
y_pred = pipeline.predict(X_test)
pos_idx = list(pipeline.classes_).index(POS_LABEL)
proba = pipeline.predict_proba(X_test)

st.header(f"Results - {spec['display_name']}")
st.caption(
    "Tuned hyper-parameters: "
    + ", ".join(f"`{k}={v}`" for k, v in spec["best_params"].items())
    + f" | 5-fold CV churn-F1 during training: {spec['cv_f1']:.4f}"
)

if has_labels:
    if use_reference:
        ref_row = load_reference_metrics().loc[spec["display_name"]]
        metrics = {c: float(ref_row[c]) for c in METRIC_ORDER}
    else:
        metrics = score_model(pipeline, X_test, y_test)

    only_one_class = y_test.nunique() < 2
    if only_one_class:
        st.info(
            "This file contains only one churn class, so AUC cannot be computed "
            "and the metrics below are not comparable with the full hold-out split."
        )

    # Two rows of three keeps all six numbers readable on narrow screens.
    items = list(metrics.items())
    for row_start in (0, 3):
        for col, (name, value) in zip(st.columns(3), items[row_start : row_start + 3]):
            col.metric(name, "n/a" if pd.isna(value) else f"{value:.4f}")

    tab_cm, tab_report, tab_pred = st.tabs(
        ["Confusion matrix", "Classification report", "Predictions"]
    )

    with tab_cm:
        labels = list(pipeline.classes_)
        st.pyplot(
            plot_confusion_matrix(
                y_test, y_pred, labels, f"{spec['display_name']} - confusion matrix"
            )
        )
        errors = int((y_test.values != y_pred).sum())
        st.caption(
            f"{len(test_df) - errors:,} of {len(test_df):,} customers classified "
            f"correctly, {errors:,} misclassified."
        )

    with tab_report:
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    with tab_pred:
        out = test_df.copy()
        out["Predicted"] = y_pred
        out["Churn probability"] = proba[:, pos_idx].round(4)
        out["Correct"] = out["Predicted"] == out[TARGET]
        st.dataframe(
            out[[TARGET, "Predicted", "Churn probability", "Correct"]].head(200),
            width="stretch",
        )
        st.download_button(
            "Download predictions as CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{selected_key}.csv",
            mime="text/csv",
        )
else:
    st.warning(
        "The uploaded file has no `Churn` column, so evaluation metrics and the "
        "confusion matrix cannot be computed. Showing predictions only."
    )
    out = test_df.copy()
    out["Predicted"] = y_pred
    out["Churn probability"] = proba[:, pos_idx].round(4)
    st.dataframe(out.head(200), width="stretch")
    st.download_button(
        "Download predictions as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_key}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# All-model comparison
# --------------------------------------------------------------------------- #
if compare_all and has_labels:
    st.header("Comparison of all five models on this test data")

    if use_reference:
        order = [display_names[k] for k in model_keys]
        table = load_reference_metrics().loc[order, METRIC_ORDER]
    else:
        rows = {}
        progress = st.progress(0.0, text="Scoring models...")
        for i, key in enumerate(model_keys, start=1):
            model = load_model(registry["models"][key]["file"])
            rows[display_names[key]] = score_model(model, X_test, y_test)
            progress.progress(i / len(model_keys), text=f"Scored {display_names[key]}")
        progress.empty()
        table = pd.DataFrame(rows).T[METRIC_ORDER]

    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#f7cad0"),
        width="stretch",
    )

    left, right = st.columns([1, 1])
    with left:
        focus = st.selectbox("Chart this metric", table.columns, index=4)
        st.pyplot(plot_metric_comparison(table, focus))
    with right:
        best = table["F1"].idxmax()
        st.subheader("Winner on this data")
        st.metric(best, f"churn-F1 {table.loc[best, 'F1']:.4f}")
        st.write(
            "The ranking is driven by F1 on the churn class, which balances "
            "catching churners (recall) against false alarms (precision) - the "
            "trade-off that matters for a retention campaign."
        )
        st.download_button(
            "Download comparison table",
            table.round(4).to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

    if use_reference:
        st.caption(
            "These are the metrics recorded by `model/train_models.py` on the hold-out "
            "split - the same numbers as the README comparison table."
        )
    else:
        with st.expander("Reference scores recorded during training"):
            st.caption(
                "Computed by `model/train_models.py` on the original hold-out split; "
                "useful as a sanity check against the numbers computed live above."
            )
            st.dataframe(load_reference_metrics().round(4), width="stretch")

st.markdown("---")
st.caption(
    "DSE ZG565 Machine Learning - Assignment 2 | Telco Customer Churn Dataset | "
    "models: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, SVM (RBF)"
)
