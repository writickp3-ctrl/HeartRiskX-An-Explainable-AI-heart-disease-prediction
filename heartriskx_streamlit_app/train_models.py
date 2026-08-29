"""
train_models.py — HeartRiskX

Reconstructs the three final pipelines described in the HeartRiskX notebooks
(01_data_preprocessing -> 06_explainability_calibration) and saves deployable
bundles for the Streamlit app.

This is necessary because the original repo never checked in any trained
model artifacts (models/ was never exported from Google Drive), and every
notebook is hard-wired to Colab paths. This script:
  - reads the three raw CSVs that ARE present in the repo
  - reproduces the exact preprocessing + LightGBM architecture from
    06_explainability_calibration.ipynb (ColumnTransformer "prep" + LGBM "clf")
  - reproduces the F1-maximizing threshold search from 05/09
  - saves one bundle per dataset: pipeline, threshold, metrics, schema info,
    and a small cached background sample for SHAP

Run once: python train_models.py
Produces: models/{heart2020,cardio,uci}_bundle.joblib
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(
            ("num", StandardScaler() if scale_numeric else "passthrough", num_cols)
        )
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)


def best_f1_threshold(y_true, proba):
    prec, rec, th = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5


def eval_metrics(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(thr),
        "positive_rate": float(np.mean(y_true)),
    }


def schema_info(X: pd.DataFrame) -> dict:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    info = {"num_cols": num_cols, "cat_cols": cat_cols, "ranges": {}, "categories": {}}
    for c in num_cols:
        info["ranges"][c] = {
            "min": float(X[c].min()),
            "max": float(X[c].max()),
            "mean": float(X[c].mean()),
            "median": float(X[c].median()),
        }
    for c in cat_cols:
        info["categories"][c] = sorted(X[c].dropna().unique().tolist())
    return info


def build_bundle(name, X, y, lgbm_params, scale_numeric, calibration_bins=10):
    print(f"\n=== {name} ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    prep = make_preprocessor(X_tr, scale_numeric=scale_numeric)
    clf = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, **lgbm_params)
    pipe = Pipeline(steps=[("prep", prep), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    proba_te = pipe.predict_proba(X_te)[:, 1]
    thr = best_f1_threshold(y_te, proba_te)
    metrics = eval_metrics(y_te, proba_te, thr)
    print(
        f"  thr={thr:.3f}  acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
        f"roc_auc={metrics['roc_auc']:.3f}  pr_auc={metrics['pr_auc']:.3f}"
    )

    # Calibration curve (for the Model Performance tab)
    prob_true, prob_pred = calibration_curve(y_te, proba_te, n_bins=calibration_bins)
    metrics["calibration"] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

    # Feature importance (post-transform names)
    feat_names = pipe.named_steps["prep"].get_feature_names_out().tolist()
    importances = pipe.named_steps["clf"].feature_importances_.tolist()
    metrics["feature_importance"] = sorted(
        zip(feat_names, importances), key=lambda t: t[1], reverse=True
    )[:20]

    # Small background sample for on-demand SHAP (kept raw; app transforms via prep)
    bg = X_tr.sample(min(300, len(X_tr)), random_state=RANDOM_STATE)

    bundle = {
        "pipeline": pipe,
        "threshold": thr,
        "metrics": metrics,
        "schema": schema_info(X),
        "background_sample": bg,
        "target_name": "target",
    }
    path = os.path.join(MODEL_DIR, f"{name}_bundle.joblib")
    joblib.dump(bundle, path)
    print(f"  saved -> {path}")
    return metrics


def main():
    # ---------------- Heart2020 ----------------
    heart2020 = pd.read_csv(os.path.join(DATA_DIR, "heart_2020_clean.csv"))
    heart2020 = heart2020.drop(columns=["HeartDisease"])  # leakage column
    Xh = heart2020.drop(columns=["target"])
    yh = heart2020["target"]
    build_bundle(
        "heart2020",
        Xh,
        yh,
        lgbm_params=dict(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
        ),
        scale_numeric=False,
    )

    # ---------------- Cardio ----------------
    cardio = pd.read_csv(os.path.join(DATA_DIR, "cardio_train.csv"), sep=";")
    cardio = cardio.rename(columns={"cardio": "target"}).drop(columns=["id"])
    Xc = cardio.drop(columns=["target"])
    yc = cardio["target"]
    build_bundle(
        "cardio",
        Xc,
        yc,
        lgbm_params=dict(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
        ),
        scale_numeric=False,
    )

    # ---------------- UCI Cleveland ----------------
    uci = pd.read_csv(os.path.join(DATA_DIR, "uci_cleveland_clean.csv"))
    Xu = uci.drop(columns=["target"])
    yu = uci["target"]
    build_bundle(
        "uci",
        Xu,
        yu,
        lgbm_params=dict(n_estimators=200, learning_rate=0.03, num_leaves=31, max_depth=-1),
        scale_numeric=True,
        calibration_bins=6,  # small dataset -> fewer bins
    )

    print("\nAll bundles built.")


if __name__ == "__main__":
    main()
