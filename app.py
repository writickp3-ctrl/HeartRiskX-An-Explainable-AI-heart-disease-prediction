"""
HeartRiskX — Explainable Heart Disease Risk Prediction
Streamlit app built on top of the pipelines trained by train_models.py.

Run:
    streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# shap (and its numba dependency) is imported lazily, inside get_explainer(),
# because some locked-down Windows environments block numba's compiled
# extensions via Application Control / Smart App Control policy. When that
# happens the rest of the app should keep working — only the per-prediction
# SHAP chart falls back to global feature importance.

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DATASET_LABELS = {
    "heart2020": "Heart2020 (BRFSS survey)",
    "cardio": "Cardio (clinical checkup)",
    "uci": "UCI Cleveland (clinical + ECG)",
}
DATASET_BLURBS = {
    "heart2020": "319,795 US adults, CDC BRFSS 2020 survey. Lifestyle + self-reported health "
    "history. Strongly imbalanced (~8.6% positive) — good for population screening, weak on precision.",
    "cardio": "70,000 clinical checkup records. Blood pressure, cholesterol/glucose bands, "
    "lifestyle flags. Roughly balanced classes.",
    "uci": "297 patients, the classic UCI Cleveland heart-disease dataset. Clinical + ECG "
    "features, small sample — best test accuracy but least generalizable.",
}

# ---------------------------------------------------------------------------
# Friendly label <-> raw value mappings for datasets whose raw schema is
# numeric codes (cardio, uci). Heart2020 is already human-readable.
# ---------------------------------------------------------------------------
CARDIO_GENDER = {"Female": 1, "Male": 2}
CARDIO_CHOL_GLUC = {"Normal": 1, "Above normal": 2, "Well above normal": 3}
CARDIO_YESNO = {"No": 0, "Yes": 1}

UCI_SEX = {"Female": 0, "Male": 1}
UCI_CP = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
UCI_FBS = {"\u2264 120 mg/dl": 0, "> 120 mg/dl": 1}
UCI_RESTECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
UCI_EXANG = {"No": 0, "Yes": 1}
UCI_SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
UCI_THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}


@st.cache_resource(show_spinner=False)
def load_bundle(name):
    path = os.path.join(MODEL_DIR, f"{name}_bundle.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_explainer(name):
    try:
        import shap
    except ImportError as e:
        return None, str(e)
    bundle = load_bundle(name)
    clf = bundle["pipeline"].named_steps["clf"]
    return shap.TreeExplainer(clf), None


def predict_with_threshold(bundle, X_df):
    proba = bundle["pipeline"].predict_proba(X_df)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)
    return proba, pred


def align_batch_columns(bundle, df):
    """Best-effort alignment of an uploaded CSV to the raw training schema."""
    schema = bundle["schema"]
    expected = schema["num_cols"] + schema["cat_cols"]
    missing = [c for c in expected if c not in df.columns]
    out = df.copy()
    for c in schema["num_cols"]:
        if c not in out.columns:
            out[c] = schema["ranges"][c]["median"]
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(schema["ranges"][c]["median"])
    for c in schema["cat_cols"]:
        if c not in out.columns:
            out[c] = schema["categories"][c][0]
        out[c] = out[c].astype(str)
    return out.reindex(columns=expected), missing


def risk_badge(proba, thr):
    label = "Elevated risk" if proba >= thr else "Lower risk"
    color = "#c0392b" if proba >= thr else "#1e8449"
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:8px;background:{color}22;"
        f"border:1px solid {color};color:{color};font-weight:600;display:inline-block;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )


def render_predict_tab(name, bundle):
    schema = bundle["schema"]
    st.subheader("Enter patient / respondent details")
    inputs = {}

    with st.form(f"predict_form_{name}"):
        if name == "heart2020":
            c1, c2 = st.columns(2)
            with c1:
                inputs["BMI"] = st.slider("BMI", 12.0, 60.0, 27.0, 0.1)
                inputs["PhysicalHealth"] = st.slider(
                    "Days physical health 'not good' (last 30d)", 0, 30, 0
                )
                inputs["MentalHealth"] = st.slider(
                    "Days mental health 'not good' (last 30d)", 0, 30, 0
                )
                inputs["SleepTime"] = st.slider("Average sleep (hours/night)", 1, 16, 7)
                inputs["Sex"] = st.selectbox("Sex", schema["categories"]["Sex"])
                inputs["AgeCategory"] = st.selectbox(
                    "Age category", schema["categories"]["AgeCategory"], index=8
                )
                inputs["Race"] = st.selectbox("Race", schema["categories"]["Race"])
                inputs["GenHealth"] = st.selectbox(
                    "Self-rated general health", schema["categories"]["GenHealth"]
                )
                inputs["Diabetic"] = st.selectbox("Diabetic status", schema["categories"]["Diabetic"])
            with c2:
                inputs["Smoking"] = st.selectbox("Smokes (100+ cigarettes lifetime)", ["No", "Yes"])
                inputs["AlcoholDrinking"] = st.selectbox("Heavy alcohol use", ["No", "Yes"])
                inputs["Stroke"] = st.selectbox("Ever had a stroke", ["No", "Yes"])
                inputs["DiffWalking"] = st.selectbox("Difficulty walking/climbing stairs", ["No", "Yes"])
                inputs["PhysicalActivity"] = st.selectbox("Physical activity (last 30d)", ["Yes", "No"])
                inputs["Asthma"] = st.selectbox("Asthma", ["No", "Yes"])
                inputs["KidneyDisease"] = st.selectbox("Kidney disease", ["No", "Yes"])
                inputs["SkinCancer"] = st.selectbox("Skin cancer", ["No", "Yes"])

        elif name == "cardio":
            c1, c2 = st.columns(2)
            with c1:
                age_years = st.slider("Age (years)", 25, 70, 52)
                inputs["age"] = age_years * 365.25
                gender = st.selectbox("Sex", list(CARDIO_GENDER))
                inputs["gender"] = CARDIO_GENDER[gender]
                inputs["height"] = st.slider("Height (cm)", 120, 210, 165)
                inputs["weight"] = st.slider("Weight (kg)", 30.0, 180.0, 72.0, 0.5)
            with c2:
                inputs["ap_hi"] = st.slider("Systolic BP (ap_hi)", 80, 220, 120)
                inputs["ap_lo"] = st.slider("Diastolic BP (ap_lo)", 40, 160, 80)
                chol = st.selectbox("Cholesterol", list(CARDIO_CHOL_GLUC))
                inputs["cholesterol"] = CARDIO_CHOL_GLUC[chol]
                gluc = st.selectbox("Glucose", list(CARDIO_CHOL_GLUC))
                inputs["gluc"] = CARDIO_CHOL_GLUC[gluc]
                smoke = st.selectbox("Smokes", list(CARDIO_YESNO))
                inputs["smoke"] = CARDIO_YESNO[smoke]
                alco = st.selectbox("Drinks alcohol", list(CARDIO_YESNO))
                inputs["alco"] = CARDIO_YESNO[alco]
                active = st.selectbox("Physically active", list(CARDIO_YESNO))
                inputs["active"] = CARDIO_YESNO[active]

        else:  # uci
            c1, c2 = st.columns(2)
            with c1:
                inputs["age"] = st.slider("Age (years)", 25, 80, 54)
                sex = st.selectbox("Sex", list(UCI_SEX))
                inputs["sex"] = UCI_SEX[sex]
                cp = st.selectbox("Chest pain type", list(UCI_CP))
                inputs["cp"] = UCI_CP[cp]
                inputs["trestbps"] = st.slider("Resting BP (mm Hg)", 80, 220, 130)
                inputs["chol"] = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
                fbs = st.selectbox("Fasting blood sugar", list(UCI_FBS))
                inputs["fbs"] = UCI_FBS[fbs]
            with c2:
                restecg = st.selectbox("Resting ECG", list(UCI_RESTECG))
                inputs["restecg"] = UCI_RESTECG[restecg]
                inputs["thalach"] = st.slider("Max heart rate achieved", 60, 220, 150)
                exang = st.selectbox("Exercise-induced angina", list(UCI_EXANG))
                inputs["exang"] = UCI_EXANG[exang]
                inputs["oldpeak"] = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
                slope = st.selectbox("Slope of peak exercise ST segment", list(UCI_SLOPE))
                inputs["slope"] = UCI_SLOPE[slope]
                inputs["ca"] = st.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", list(UCI_THAL))
                inputs["thal"] = UCI_THAL[thal]

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        X = pd.DataFrame([inputs])
        # reindex to exact training column order
        X = X.reindex(columns=schema["num_cols"] + schema["cat_cols"])
        proba, pred = predict_with_threshold(bundle, X)
        proba, pred = float(proba[0]), int(pred[0])

        st.session_state[f"last_input_{name}"] = X
        st.session_state[f"last_proba_{name}"] = proba
        st.session_state[f"last_pred_{name}"] = pred

    if f"last_proba_{name}" in st.session_state:
        proba = st.session_state[f"last_proba_{name}"]
        thr = bundle["threshold"]
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted risk probability", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))
            risk_badge(proba, thr)
        with col2:
            st.caption(
                f"Classified using this model's F1-optimized decision threshold "
                f"({thr:.2f}), not the default 0.5 — see the Model Performance tab for why."
            )
            st.caption("Go to the **Explain** tab to see which inputs drove this specific prediction.")


def render_explain_tab(name, bundle):
    st.subheader("What drives this model")

    metrics = bundle["metrics"]
    fi = metrics["feature_importance"][:15]
    feats = [f for f, _ in fi][::-1]
    vals = [v for _, v in fi][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(feats, vals, color="#5b6ee1")
    ax.set_xlabel("LightGBM feature importance (split count)")
    ax.set_title("Global — top features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Why this prediction")

    if f"last_input_{name}" not in st.session_state:
        st.info("Run a prediction in the **Predict** tab first, then come back here.")
        return

    X = st.session_state[f"last_input_{name}"]
    pipe = bundle["pipeline"]
    prep = pipe.named_steps["prep"]
    explainer, shap_error = get_explainer(name)

    if explainer is None:
        st.warning(
            "Per-prediction SHAP explanations aren't available in this environment "
            "(the `shap`/`numba` import failed to load — often blocked by a Windows "
            "Application Control policy). Showing global feature importance above "
            "instead."
        )
        st.caption(f"Details: {shap_error}")
        return

    X_t = prep.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feat_names = prep.get_feature_names_out()

    shap_vals = explainer.shap_values(X_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    row = np.asarray(shap_vals)[0]

    order = np.argsort(np.abs(row))[::-1][:10]
    top_feats = [feat_names[i] for i in order][::-1]
    top_vals = [row[i] for i in order][::-1]
    colors = ["#c0392b" if v > 0 else "#1e8449" for v in top_vals]

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.barh(top_feats, top_vals, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (push toward higher risk \u2192)")
    ax2.set_title("This prediction — top contributing features")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Red bars push the prediction toward higher risk; green bars push it toward lower risk.")


def render_batch_tab(name, bundle):
    st.subheader("Batch prediction from CSV")
    schema = bundle["schema"]
    expected = schema["num_cols"] + schema["cat_cols"]
    st.caption(
        "Upload a CSV with the raw training columns (same names/units the model was trained on): "
        + ", ".join(expected)
    )
    up = st.file_uploader("CSV file", type=["csv"], key=f"upload_{name}")
    if up is not None and st.button("Run batch prediction", key=f"batch_btn_{name}"):
        try:
            df_raw = pd.read_csv(up)
            df_aligned, missing = align_batch_columns(bundle, df_raw)
            if missing:
                st.warning(f"Missing columns filled with defaults: {', '.join(missing)}")
            proba, pred = predict_with_threshold(bundle, df_aligned)
            out = df_raw.copy()
            out["risk_probability"] = proba
            out["risk_label"] = pred
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button(
                "Download predictions.csv",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"{name}_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


def render_performance_tab(name, bundle):
    m = bundle["metrics"]
    st.subheader("Held-out test performance")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    cols[1].metric("Precision", f"{m['precision']:.3f}")
    cols[2].metric("Recall", f"{m['recall']:.3f}")
    cols[3].metric("F1", f"{m['f1']:.3f}")
    cols2 = st.columns(4)
    cols2[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    cols2[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    cols2[2].metric("Brier score", f"{m['brier']:.3f}")
    cols2[3].metric("Decision threshold", f"{m['threshold']:.3f}")

    st.caption(
        f"Positive class prevalence in this dataset: {m['positive_rate']:.1%}. "
        "The threshold above is F1-optimized on the test split, not the default 0.5 — "
        "for a heavily imbalanced dataset like Heart2020, 0.5 collapses recall to near zero."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix** (at chosen threshold)")
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**Calibration curve**")
        cal = m["calibration"]
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
        ax2.plot(cal["prob_pred"], cal["prob_true"], "s-", color="#5b6ee1", label="Model")
        ax2.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Observed frequency")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def render_about_tab():
    st.subheader("About HeartRiskX")
    st.markdown(
        """
HeartRiskX predicts heart-disease risk from three independent public datasets and
explains each prediction with SHAP, instead of collapsing them into one shared schema.

**Datasets**
- Heart2020 — CDC BRFSS 2020 survey, 319,795 respondents, lifestyle + self-reported health
- Cardio — 70,000 clinical checkup records
- UCI Cleveland — 297 patients, clinical + ECG features (the original 1988 benchmark dataset)

**Pipeline (per dataset)**: median/passthrough numeric handling → one-hot encode
categoricals → LightGBM (`class_weight="balanced"`) → F1-optimized threshold on a
held-out test split → SHAP `TreeExplainer` for per-prediction attribution.

**Why three separate models instead of one merged model:** the three datasets don't
share a feature schema (survey answers vs. clinical measurements vs. ECG findings), and
forcing them into one shared feature set (as the project's own cross-dataset notebook
tried) throws away most of what each dataset actually measures. Pick the dataset that
matches the kind of input you have.
"""
    )
    st.warning(
        "Educational / research prototype only. Not a validated diagnostic tool and not a "
        "substitute for medical advice — no output here should be used to make a real "
        "clinical decision."
    )


def main():
    st.set_page_config(page_title="HeartRiskX", page_icon="\u2764\ufe0f", layout="wide")
    st.title("\u2764\ufe0f HeartRiskX — Explainable Heart Disease Risk")

    st.sidebar.title("HeartRiskX")
    dataset = st.sidebar.selectbox(
        "Dataset / model",
        list(DATASET_LABELS),
        format_func=lambda k: DATASET_LABELS[k],
    )
    st.sidebar.caption(DATASET_BLURBS[dataset])

    bundle = load_bundle(dataset)
    if bundle is None:
        st.error(
            f"No trained bundle found for '{dataset}'. Run `python train_models.py` first "
            "to generate models/*.joblib."
        )
        st.stop()

    tab_predict, tab_explain, tab_batch, tab_perf, tab_about = st.tabs(
        ["\U0001F52E Predict", "\U0001F50D Explain", "\U0001F4C1 Batch CSV", "\U0001F4CA Model Performance", "\u2139\ufe0f About"]
    )
    with tab_predict:
        render_predict_tab(dataset, bundle)
    with tab_explain:
        render_explain_tab(dataset, bundle)
    with tab_batch:
        render_batch_tab(dataset, bundle)
    with tab_perf:
        render_performance_tab(dataset, bundle)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
