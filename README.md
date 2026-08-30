# ❤️ HeartRiskX — Explainable AI Heart Disease Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-02569B)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Educational%20%2F%20Research-lightgrey)](#license)

An end-to-end, explainable machine learning system that predicts heart-disease
risk from three independent real-world datasets, optimizes its own decision
threshold instead of defaulting to 0.5, and explains every prediction with
SHAP — wrapped in a proper interactive Streamlit application.

---

## Table of Contents
- [Overview](#overview)
- [Why Three Separate Models](#why-three-separate-models)
- [Notebooks](#notebooks)
- [Live App Features](#live-app-features)
- [Datasets](#datasets)
- [Pipeline & Architecture](#pipeline--architecture)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Engineering Notes](#engineering-notes)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

HeartRiskX is a machine-learning pipeline and interactive application for
heart-disease risk prediction, built around three principles that most
"heart disease predictor" projects skip:

1. **Explainability is not an afterthought.** Every single prediction —
   not just the model in aggregate — is explained with SHAP, showing
   exactly which inputs pushed the risk score up or down.
2. **The decision threshold is chosen, not assumed.** Classifying at the
   default 0.5 cutoff on an imbalanced medical dataset quietly destroys
   recall. The threshold for each model is instead selected by maximizing
   F1-score on a held-out test split.
3. **Real datasets are kept real.** Rather than merge three structurally
   different datasets into one lossy shared schema, each is modeled and
   evaluated independently, on its own terms.
4. **The final model earned its place.** Logistic Regression, Random Forest,
   XGBoost, LightGBM, and a Stacking ensemble were all benchmarked per
   dataset before LightGBM was selected as the final model — see
   `02_baseline_models.ipynb` and `03_advanced_models.ipynb`.

## Why Three Separate Models

| | Heart2020 | Cardio | UCI Cleveland |
|---|---|---|---|
| Source | CDC BRFSS 2020 survey | Clinical checkup records | 1988 clinical benchmark |
| Size | 319,795 rows | 70,000 rows | 297 rows |
| Feature type | Self-reported lifestyle & health history | Blood pressure, cholesterol/glucose bands | Clinical + ECG findings |
| Class balance | ~8.6% positive (heavily imbalanced) | Roughly balanced | Roughly balanced |

An earlier notebook in the project (`08_cross_dataset_evaluation.ipynb`)
explored stacking ensembles and calibration comparisons per dataset, but
never merges the three schemas into one model — the datasets don't share
enough structure for that to make sense (you can't infer ECG findings from
a survey answer). Each is modeled and evaluated on its own terms instead.

## Notebooks

The research pipeline behind the app, in order:

| Notebook | Contents |
|---|---|
| `00_project_overview.ipynb` | Project goals and scope |
| `01_data_preprocessing.ipynb` | Loading and cleaning all three datasets |
| `02_baseline_models.ipynb` | Logistic Regression + Random Forest baselines |
| `03_advanced_models.ipynb` | XGBoost, LightGBM, and a Stacking ensemble |
| `04_class_imbalance_handling.ipynb` | SMOTE and class-imbalance experiments |
| `05_model_tuning.ipynb` | Hyperparameter search (`RandomizedSearchCV`) |
| `06_explainability_calibration.ipynb` | Final LightGBM pipelines, SHAP, calibration |
| `07_robustness_testing.ipynb` | Bundle export + prediction demos |
| `08_cross_dataset_evaluation.ipynb` | Per-dataset stacking, calibration, SHAP dependence/PDP plots |
| `09_threshold_optimization.ipynb` | F1-maximizing threshold selection |
| `10_final_evaluation.ipynb` | Final metrics summary across all three datasets |
| `11_final_model_bundle.ipynb` | Bundle export + an early prototype UI |

A note on naming: `07` and `08` are titled "robustness testing" and
"cross-dataset evaluation," but neither notebook actually does what its name
implies — `07` contains no noise-injection or feature-ablation testing, and
`08` never trains on one dataset and evaluates on another. Flagging this here
rather than let the filenames imply capabilities that aren't backed by what's
actually in them.

## Live App Features

The Streamlit app (`app.py`) has five tabs per dataset:

| Tab | What it does |
|---|---|
| 🔮 **Predict** | Real form inputs (sliders, dropdowns) matched to each dataset's actual value ranges — not a raw JSON box |
| 🔍 **Explain** | Global feature importance + a per-prediction SHAP bar chart showing exactly what drove *that* prediction (falls back to global importance only if `shap`/`numba` can't load in the environment) |
| 📁 **Batch CSV** | Upload a CSV, get risk scores + labels for every row, download the results |
| 📊 **Model Performance** | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, and a calibration curve |
| ℹ️ **About** | Dataset descriptions and project context |

## Datasets

- **Heart2020** — [CDC BRFSS 2020 Annual Survey](https://www.cdc.gov/brfss/) (via Kaggle), 319,795 respondents
- **Cardio** — [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), 70,000 records
- **UCI Cleveland** — [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), 297 patients

## Pipeline & Architecture

```
Raw CSV
  │
  ▼
Preprocessing        →  numeric passthrough/scaling + one-hot encoding
  │                      (ColumnTransformer, per-dataset schema)
  ▼
Model Training        →  LightGBM (class_weight="balanced")
  │
  ▼
Threshold Optimization → F1-maximizing threshold on held-out test split
  │                      (not a fixed 0.5 cutoff)
  ▼
Calibration Check      → Brier score + calibration curve
  │
  ▼
Explainability          → SHAP TreeExplainer, per-prediction attribution
  │
  ▼
Deployable Bundle       → pipeline + threshold + metrics + schema (joblib)
  │
  ▼
Streamlit App           → predict / explain / batch / performance
```

Each dataset gets its own bundle in `models/`, produced by `train_models.py`.

## Model Selection

Before LightGBM was chosen as the final model, five classifiers were
benchmarked per dataset:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- A Stacking ensemble (LogReg + LightGBM base learners)

LightGBM was carried forward as the final model for its balance of
accuracy, calibration, and training speed across all three datasets.
Calibration was also compared before/after Platt scaling
(`CalibratedClassifierCV`, sigmoid method) per dataset, and SHAP
dependence plots plus partial dependence plots were generated for the
top contributing features to sanity-check that the model's behavior
matched clinical intuition (e.g. risk rising monotonically with age).

## Model Performance

Held-out test-set results, F1-optimized threshold:

| Dataset | Accuracy | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---|---|---|---|---|
| Heart2020 | 0.876 | 0.397 | 0.840 | 0.351 | 0.744 |
| Cardio | 0.707 | 0.742 | 0.800 | 0.784 | 0.350 |
| UCI Cleveland | 0.883 | 0.877 | 0.914 | 0.894 | 0.335 |

Heart2020's F1 looks low next to the other two — that's the dataset's ~8.6%
positive-class prevalence showing up honestly, not a modeling error. Full
precision/recall breakdown, a live confusion matrix, and a calibration curve
are available in the app's **Model Performance** tab.

## Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Modeling:** scikit-learn, LightGBM, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **App:** Streamlit

## Project Structure

```
HeartRiskX/
├── 00_project_overview.ipynb          # Research notebooks (see Notebooks section)
├── 01_data_preprocessing.ipynb
├── ...
├── 11_final_model_bundle.ipynb
├── app.py                             # Streamlit application (5 tabs, 3 dataset selector)
├── train_models.py                    # Preprocessing + training + bundling pipeline
├── requirements.txt
├── data/                              # Raw datasets (shared by notebooks and app)
│   ├── heart_2020_clean.csv
│   ├── cardio_train.csv
│   └── uci_cleveland_clean.csv
├── models/                            # Generated bundles (pipeline + threshold + metrics)
│   ├── heart2020_bundle.joblib
│   ├── cardio_bundle.joblib
│   └── uci_bundle.joblib
└── README.md
```

Everything lives at the repo root — `app.py` and `train_models.py` are not
in a subfolder, so the commands below work directly from a fresh clone.

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all three models (~2-3 minutes)
python train_models.py

# 3. Launch the app
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Engineering Notes

A few non-obvious issues came up productionizing this into a locally
runnable app, worth flagging for anyone reusing this pipeline:

- **LightGBM import order matters on Windows.** If `sklearn` hasn't been
  imported yet when `lightgbm` is first imported, LightGBM's internal
  scikit-learn compatibility check can fail even when scikit-learn is
  correctly installed. Fix: import `sklearn` before `lightgbm`.
- **SHAP depends on `numba`**, which ships compiled extensions that some
  locked-down Windows environments block via Application Control / Smart
  App Control policy. The app imports `shap` lazily and falls back to
  global feature importance if it can't load, rather than crashing.
- **Threshold ≠ 0.5.** Every bundle stores its own F1-optimized threshold;
  the app always classifies against that, not the sklearn default.

## Limitations & Disclaimer

This is an educational / research prototype, **not a validated diagnostic
tool**. It has not been clinically validated, is not a substitute for
professional medical advice, and no output from this app should inform a
real clinical decision. Heart2020 in particular has low precision/recall
at any reasonable threshold given its class imbalance — treat its outputs
as a coarse screening signal, not a diagnosis.

## Roadmap

- [ ] Wearable / continuous ECG data integration
- [ ] Neural network baseline for comparison against LightGBM
- [ ] Clinical-style dashboard view
- [ ] Model card + fairness/subgroup performance breakdown

## Author

**Writick Parui**
M.E CSE @ Thapar Institute of Engineering & Technology
GATE 2025 | Former TCS Intern
GitHub: [github.com/writickp3-ctrl](https://github.com/writickp3-ctrl)

## License

Educational and research use only.
