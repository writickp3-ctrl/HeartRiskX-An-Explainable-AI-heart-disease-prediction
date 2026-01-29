# HeartRiskX – Explainable Heart Disease Prediction System ❤️

## Overview

HeartRiskX is a machine learning–based heart disease risk prediction system designed to provide accurate and explainable clinical insights. The project builds an end-to-end pipeline that preprocesses patient data, trains multiple ML models, optimizes decision thresholds, and explains predictions using SHAP.

The system integrates three real-world datasets (Heart2020, Kaggle Cardio, and UCI Cleveland) to ensure robustness and generalization across diverse populations.

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- SHAP (Explainable AI)  
- Matplotlib / Seaborn  

---

## Key Features

- End-to-end ML pipeline (preprocessing → modeling → evaluation → explainability)  
- Multiple classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking  
- Threshold optimization using F1-score (instead of fixed 0.5 cutoff)  
- SHAP-based global and local explainability  
- Calibration curves and Brier score analysis  
- Robustness testing (noise injection, feature drop, cross-dataset validation)  
- Final LightGBM model packaged as reusable bundle  

---

## Datasets

- Heart2020 (BRFSS)  
- Kaggle Cardio  
- UCI Cleveland  

Each dataset is preprocessed using a unified pipeline with imputation, scaling, and one-hot encoding.

---

## Workflow

1. Load and preprocess datasets  
2. Train multiple ML models  
3. Evaluate using Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC  
4. Perform threshold tuning  
5. Select LightGBM as final model  
6. Generate SHAP explanations  
7. Run robustness experiments  
8. Save final model bundles and metrics  

---

## Outcome

Built an explainable ML prototype capable of predicting heart disease risk with calibrated probabilities and transparent feature-level explanations, supporting trustworthy medical decision assistance.

---

## Future Scope

- Streamlit deployment for real-time predictions  
- Integration with wearable / ECG data  
- Neural network models  
- Clinical dashboard  

---

## Author

**Writick Parui**  
M.Tech CSE @ Thapar Institute of Engineering & Technology  
GATE 2025 | Former TCS Intern  

GitHub: https://github.com/writickp3-ctrl  

---

## License

Educational and research use only.
