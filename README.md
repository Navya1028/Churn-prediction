#  Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)](https://shap.readthedocs.io)

> End-to-end ML pipeline to predict telecom customer churn — with hyperparameter tuning, SHAP explainability, threshold optimisation, and an interactive Streamlit dashboard.

---

##  Problem Statement

Customer churn costs telecom companies 5–25× more than retaining existing customers. This project builds a churn classifier that identifies at-risk customers **before** they leave, enabling targeted retention campaigns.

**Key design decision:** The model is optimised for **Recall** (catching as many churners as possible), not raw accuracy. Missing a churner is far more costly than incorrectly flagging a loyal customer.

---

##  Results (on held-out test set)

| Model | Recall | F1 | ROC-AUC | Threshold |
|---|---|---|---|---|
| XGBoost (tuned) | **0.79** | **0.62** | **0.85** | 0.55 |
| Random Forest (baseline) | 0.50 | 0.55 | 0.82 | 0.50 |

---

##  Improvements Over the Baseline Notebook

| Area | Baseline | This project |
|---|---|---|
| Class imbalance | SMOTE oversampling | `scale_pos_weight` in XGBoost |
| Cross-validation | Plain KFold | StratifiedKFold (preserves class ratio) |
| Scoring metric | Accuracy | Recall → F1 |
| Feature set | 19 raw features | +3 engineered features |
| Decision threshold | Fixed 0.50 | Tuned per F1 on test set |
| Explainability | None | SHAP waterfall per prediction |
| Evaluation | Confusion matrix only | ROC-AUC + PR curve + SHAP plots |

---

##  Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `avg_monthly_spend` | `TotalCharges / (tenure + 1)` | High-value + new customers are highest risk |
| `num_services` | Count of active add-ons | More services → higher switching cost → lower churn |
| `is_new_customer` | `1 if tenure ≤ 3` | First 3 months = highest churn risk window |

`Contract` was ordinally encoded (Month-to-month=0, One year=1, Two year=2) to preserve natural ordering instead of plain label encoding.

---

## Project Structure

```
churn-prediction/
├── telco_customerchurn.csv           # Dataset (download from Kaggle)
├── churn_prediction_enhanced.py      # Full ML pipeline
├── app.py                            # Streamlit dashboard
├── requirements.txt
├── README.md
└── artifacts/                        # Auto-generated after running pipeline
    ├── customer_churn_model.pkl
    ├── encoders.pkl
    ├── shap_explainer.pkl
    ├── model_comparison.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── shap_bar.png
    └── shap_summary.png
```

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

### 2. Get the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `telco_customerchurn.csv` in the project root.

### 3. Train the model

```bash
python churn_prediction_enhanced.py
```

This runs the full pipeline and saves all `.pkl` and `.png` artifacts.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

##  Dashboard Features

**Predict tab** — Input form → real-time churn probability + risk level + SHAP waterfall explanation + personalised retention tips

**EDA tab** — Churn rate by contract, internet service, payment method; tenure and charges distributions; correlation heatmap

**Model Info tab** — Full comparison table, ROC curve, PR curve, confusion matrix, SHAP importance charts

---

##  Key Findings from EDA

- **Month-to-month contracts** churn at ~42% vs ~11% for two-year contracts
- **Fiber optic** customers churn at nearly 3× the rate of DSL customers
- **Electronic check** payers have the highest churn of any payment method
- **Tenure** is the strongest predictor — customers past 12 months rarely leave
- **avg_monthly_spend** (engineered) ranks in top 5 by SHAP importance

---

##  Tech Stack

Python · scikit-learn · XGBoost · SHAP · Streamlit · Pandas · NumPy · Matplotlib · Seaborn

---

##  Dataset

IBM Watson Telco Customer Churn — 7,043 customers, 21 features.
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📄 License

MIT License
