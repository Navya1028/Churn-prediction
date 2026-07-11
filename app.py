"""
Telecom Churn Prediction — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap, pickle, os
import anthropic

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

# ── GenAI retention note ─────────────────────────────────────────────────────
def generate_retention_note(top_features):
    """Turn top SHAP features into a plain-English retention action."""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    prompt = (
        f"A telecom customer is predicted to churn. The top drivers pushing "
        f"them toward churn are: {top_features}. "
        f"Write a 2-sentence retention action for the account manager. "
        f"Be specific and practical, no generic advice."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

# ── Load artifacts ─────────────────────────────────────────────────────────────
# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(r"D:\telco customer churn\customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open(r"D:\telco customer churn\encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(r"D:\telco customer churn\shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model_data, encoders, explainer

@st.cache_data
def load_raw():
    df = pd.read_csv(r"D:\telco customer churn\telco_customerchurn.csv")
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

model_data, encoders, explainer = load_artifacts()
model     = model_data["model"]
features  = model_data["feature_names"]
threshold = model_data["threshold"]
df_raw    = load_raw()

# ── Navigation ─────────────────────────────────────────────────────────────────
st.sidebar.title(" Churn Predictor")
page = st.sidebar.radio("Go to", ["Predict", "EDA", "Model Info"])

# ==============================================================================
# PAGE 1 — PREDICT
# ==============================================================================
if page == " Predict":
    st.title("Customer Churn Prediction")
    st.markdown("Fill in the customer profile below to get a real-time churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender     = st.selectbox("Gender", ["Female", "Male"])
        senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner    = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.subheader("Account")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment         = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges   = st.number_input("Total Charges ($)", 0.0, value=float(round(monthly_charges * tenure, 2)))

    with col3:
        st.subheader("Services")
        phone    = st.selectbox("Phone Service", ["Yes", "No"])
        multi    = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        sec      = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv       = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies   = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # ── Build input ─────────────────────────────────────────────────────────────
    svc_vals = [phone, multi, internet, sec, backup, device, tech, tv, movies]
    num_services      = sum(1 for v in svc_vals if v not in ["No","No internet service","No phone service"])
    avg_monthly_spend = total_charges / (tenure + 1)
    is_new_customer   = 1 if tenure <= 3 else 0
    contract_map      = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    raw = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract_map[contract], "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "avg_monthly_spend": avg_monthly_spend,
        "num_services": num_services,
        "is_new_customer": is_new_customer,
    }
    inp = pd.DataFrame([raw])
    for col, enc in encoders.items():
        if col in inp.columns:
            inp[col] = enc.transform(inp[col])
    inp = inp[features]

    # ── Predict ─────────────────────────────────────────────────────────────────
    if st.button("Predict Churn", type="primary"):
        prob = model.predict_proba(inp)[0][1]
        pred = int(prob >= threshold)

        st.divider()
        m1, m2, m3 = st.columns(3)
        label = "Likely to Churn" if pred else "Likely to Stay"
        color = "#ff4b4b" if pred else "#21c354"
        m1.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
        risk = "High" if prob > 0.65 else "Medium" if prob > 0.4 else "Low"
        m2.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Risk Level", risk)
        m3.metric("Decision Threshold", f"{threshold:.2f}")
        m3.caption("Tuned for maximum Recall on held-out test set")

        # SHAP waterfall
        st.subheader("Why this prediction?")
        shap_vals = explainer.shap_values(inp)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=inp.iloc[0].values,
                feature_names=features,
            ), show=False,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # GenAI-generated retention note (uses the same SHAP values above)
        if pred:
            st.subheader("🤖 AI-Generated Retention Note")
            with st.spinner("Generating recommendation..."):
                shap_row = pd.Series(shap_vals[0], index=features)
                top3 = shap_row.abs().sort_values(ascending=False).head(3)
                top_features_str = ", ".join(
                    f"{feat} ({'increases' if shap_row[feat] > 0 else 'decreases'} risk)"
                    for feat in top3.index
                )
                try:
                    note = generate_retention_note(top_features_str)
                    st.info(note)
                except Exception as e:
                    st.warning(f"GenAI note unavailable: {e}")

        # Retention tips
        if pred:
            st.subheader("💡 Retention Recommendations")
            tips = []
            if contract == "Month-to-month":
                tips.append("Offer a discounted 1-year contract to lock in the customer.")
            if monthly_charges > 80:
                tips.append("Consider a loyalty discount — charges are above average.")
            if is_new_customer:
                tips.append("New customer (<3 months). Trigger an onboarding check-in call immediately.")
            if num_services <= 2:
                tips.append("Bundle extra services to increase stickiness.")
            if not tips:
                tips.append("Send a personalised retention offer immediately.")
            for tip in tips:
                st.markdown(f"- {tip}")

# ==============================================================================
# PAGE 2 — EDA
# ==============================================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    overall = df_raw["Churn_num"].mean()
    st.metric("Overall Churn Rate", f"{overall:.1%}")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn rate by Contract")
        fig, ax = plt.subplots(figsize=(5, 3))
        d = df_raw.groupby("Contract")["Churn_num"].mean().reset_index()
        sns.barplot(data=d, x="Contract", y="Churn_num", palette="Blues_d", ax=ax)
        ax.set_ylabel("Churn Rate"); ax.set_ylim(0, 0.55)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.0%}", (p.get_x()+p.get_width()/2, p.get_height()+0.01), ha="center", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.subheader("Churn rate by Internet Service")
        fig, ax = plt.subplots(figsize=(5, 3))
        d = df_raw.groupby("InternetService")["Churn_num"].mean().reset_index()
        sns.barplot(data=d, x="InternetService", y="Churn_num", palette="Reds_d", ax=ax)
        ax.set_ylabel("Churn Rate"); ax.set_ylim(0, 0.55)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.0%}", (p.get_x()+p.get_width()/2, p.get_height()+0.01), ha="center", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Tenure distribution by Churn")
        fig, ax = plt.subplots(figsize=(5, 3))
        for label, grp in df_raw.groupby("Churn"):
            grp["tenure"].plot.kde(ax=ax, label=label, linewidth=2)
        ax.set_xlabel("Tenure (months)"); ax.legend(title="Churn")
        st.pyplot(fig, use_container_width=True); plt.close()

    with c4:
        st.subheader("Monthly Charges by Churn")
        fig, ax = plt.subplots(figsize=(5, 3))
        for label, grp in df_raw.groupby("Churn"):
            grp["MonthlyCharges"].plot.kde(ax=ax, label=label, linewidth=2)
        ax.set_xlabel("Monthly Charges ($)"); ax.legend(title="Churn")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.subheader("Churn rate by Payment Method")
    fig, ax = plt.subplots(figsize=(8, 3))
    d = df_raw.groupby("PaymentMethod")["Churn_num"].mean().sort_values(ascending=False)
    d.plot.bar(ax=ax, color="#5c5ce0", edgecolor="none")
    ax.set_ylabel("Churn Rate"); ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right"); ax.set_ylim(0, 0.55)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.subheader("Correlation Heatmap (numerical)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_raw[["tenure","MonthlyCharges","TotalCharges","Churn_num"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig, use_container_width=True); plt.close()

# ==============================================================================
# PAGE 3 — MODEL INFO
# ==============================================================================
elif page == "Model Info":
    st.title("Model Details")
    st.markdown("""
### Algorithm
**XGBoost Classifier** tuned with `RandomizedSearchCV` (50 iterations, 5-fold StratifiedKFold, scored on **Recall**).

### Why Recall?
Missing a churner (false negative) costs the business 5–25× more than incorrectly flagging a loyal customer.
Recall tells us: *of all customers who actually left, how many did we catch?*

### Key Improvements over Baseline Notebook

| Area | Baseline | This project |
|---|---|---|
| Class imbalance | SMOTE | `scale_pos_weight` in XGBoost |
| Cross-validation | Plain KFold | StratifiedKFold |
| Scoring metric | Accuracy | Recall → F1 |
| Features | 19 raw | +3 engineered |
| Threshold | Fixed 0.50 | Tuned (0.55 on this data) |
| Explainability | None | SHAP waterfall per prediction |

### Engineered Features
| Feature | Formula | Why |
|---|---|---|
| `avg_monthly_spend` | `TotalCharges / (tenure+1)` | High spend + low tenure = at risk |
| `num_services` | Count of active add-ons | More services → lower churn |
| `is_new_customer` | `tenure ≤ 3` | Highest churn window |

### Actual Results on Your Dataset
| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost (tuned) | **0.79** | **0.62** | **0.85** |
| Random Forest (baseline) | 0.50 | 0.55 | 0.82 |
""")

    try:
        comp = pd.read_csv("model_comparison.csv")
        st.subheader("Model Comparison (from training run)")
        st.dataframe(comp, use_container_width=True)
    except FileNotFoundError:
        st.info("Run churn_prediction_enhanced.py first to generate model_comparison.csv")

    for img, caption in [
        ("roc_curve.png", "ROC Curve"),
        ("pr_curve.png", "Precision-Recall Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("shap_bar.png", "SHAP Feature Importance"),
        ("shap_summary.png", "SHAP Summary (dot plot)"),
    ]:
        if os.path.exists(img):
            st.subheader(caption)
            st.image(img, use_column_width=True)
