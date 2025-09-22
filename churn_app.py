import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# 🔹 Must be the first Streamlit command
st.set_page_config(page_title="Churn Prediction App", page_icon="🛒", layout="wide")

# ---------- Title ----------
st.title("🛒 E-commerce Churn Prediction App")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("churn_pipeline.pkl")
    with open("feature_list.json", "r") as f:
        FEATURES = json.load(f)
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    return pipe, FEATURES, metadata

pipe, FEATURES, metadata = load_artifacts()
feature_medians = metadata.get("feature_medians", {})
recency_strategy = metadata.get("recency_strategy", "recency_ratio_2x")
avg_gap_median = metadata.get("avg_gap_median", 30)

if recency_strategy == "recency_ratio_2x":
    st.caption(f"⚙️ Model trained with Recency Ratio strategy: churn if inactivity > 2× usual purchase cycle "
               f"(typical cycle ≈ {avg_gap_median:.0f} days).")

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction"])

# ---------- Utility Functions ----------
def coerce_and_fill_row(df_row):
    """Ensure numeric types, fill missing with medians, return df with FEATURES order."""
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors='coerce')
    for f in FEATURES:
        if f not in df_row.columns:
            df_row[f] = feature_medians.get(f, 0)
    df_row = df_row[FEATURES]
    for f in FEATURES:
        if df_row[f].isna().any():
            df_row[f] = df_row[f].fillna(feature_medians.get(f, 0))
    return df_row

# -----------------------
# Tab 1: Single Prediction
# -----------------------
with tab1:
    st.subheader("🔮 Predict for a Single Customer")
    with st.form("single_form"):
        inputs = {}
        inputs["frequency"] = st.number_input("🛍️ Number of Orders", min_value=0, value=1)
        inputs["avg_order_value"] = st.number_input("💰 Average Order Value", min_value=0.0, value=500.0)
        inputs["total_spent"] = st.number_input("💳 Total Spent", min_value=0.0, value=500.0)
        inputs["avg_review_score"] = st.slider("⭐ Average Review Score", 0.0, 5.0, 4.0)
        inputs["avg_delivery_delay"] = st.number_input("⏱️ Avg Delivery Delay (days)", min_value=0, value=0)
        inputs["recency_days"] = st.number_input("📅 Recency (days since last purchase)", min_value=0, value=30)
        inputs["tenure_days"] = st.number_input("📆 Tenure (days since first purchase)", min_value=0, value=365)
        submit_single = st.form_submit_button("🔮 Predict Churn")

    if submit_single:
        try:
            user_df = pd.DataFrame([inputs])
            user_df = coerce_and_fill_row(user_df)

            probs = pipe.predict_proba(user_df)[:, 1]
            p = float(probs[0])

            # Compute recency ratio explanation
            recency_days = inputs["recency_days"]
            ratio = recency_days / avg_gap_median if avg_gap_median > 0 else np.nan

            if p >= 0.66:
                st.markdown("### 🔴 High Risk of Churn")
                st.info(f"This customer has been inactive for **{ratio:.1f}× longer** than a typical cycle "
                        f"(~{avg_gap_median:.0f} days). Immediate retention action recommended.")
            elif p >= 0.33:
                st.markdown("### 🟡 Moderate Risk of Churn")
                st.info(f"Inactivity ≈ {ratio:.1f}× usual cycle (~{avg_gap_median:.0f} days). "
                        f"Consider targeted reminders or discounts.")
            else:
                st.markdown("### 🟢 Low Risk of Churn")
                st.info(f"Inactivity ≈ {ratio:.1f}× usual cycle (~{avg_gap_median:.0f} days). "
                        f"No urgent action required.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# -----------------------
# Tab 2: Batch Prediction
# -----------------------
with tab2:
    st.subheader("📂 Batch Prediction via CSV")
    st.write("Upload a CSV with customer features. Extra columns ignored; missing features filled with medians.")

    # --- Provide downloadable template ---
    template_df = pd.DataFrame([{
        "frequency": 5,
        "avg_order_value": 250.0,
        "total_spent": 1250.0,
        "avg_review_score": 4.5,
        "avg_delivery_delay": 1,
        "recency_days": 45,
        "tenure_days": 400
    }])
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Template", template_csv,
                       "sample_customers_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Add missing features
            for f in FEATURES:
                if f not in data.columns:
                    data[f] = feature_medians.get(f, 0)

            # Process
            data_proc = data[FEATURES].copy()
            for f in FEATURES:
                data_proc[f] = pd.to_numeric(
                    data_proc[f], errors='coerce'
                ).fillna(feature_medians.get(f, 0))

            max_rows = 10000
            if len(data_proc) > max_rows:
                st.warning(f"Large file detected. Only first {max_rows} rows will be processed.")
                data_proc = data_proc.head(max_rows)

            probs = pipe.predict_proba(data_proc)[:, 1]

            # Risk + explanation
            risk_labels = []
            explanations = []
            for i, p in enumerate(probs):
                recency_days = data_proc.iloc[i]["recency_days"]
                ratio = recency_days / avg_gap_median if avg_gap_median > 0 else np.nan
                if p >= 0.66:
                    risk_labels.append("High Risk")
                    explanations.append(f"{ratio:.1f}× longer than usual cycle")
                elif p >= 0.33:
                    risk_labels.append("Moderate Risk")
                    explanations.append(f"{ratio:.1f}× usual cycle")
                else:
                    risk_labels.append("Low Risk")
                    explanations.append(f"{ratio:.1f}× usual cycle")

            data_out = data.copy()
            data_out["Risk Level"] = risk_labels
            data_out["Inactivity vs Cycle"] = explanations

            st.subheader("📊 Results")
            st.dataframe(data_out)

            csv = data_out.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv,
                               "batch_churn_predictions.csv", "text/csv")
            st.success("✅ Predictions complete. Risk Level + Explanation added.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")
