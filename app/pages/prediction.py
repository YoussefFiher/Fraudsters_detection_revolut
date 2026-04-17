import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🔍 Fraud Prediction Engine")
st.markdown("""
Evaluate the fraud risk of a user by selecting an instance from the dataset, editing its values, and running the model.
""")

# -----------------------------
# Load data, models, scaler
# -----------------------------
df = pd.read_csv("datasets/final_df.csv")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "Gradient Boosting": joblib.load("model/gradient_boosting.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Model selection
# -----------------------------
st.subheader("Model Selection")

model_name = st.selectbox("Choose a model:", list(models.keys()))
model = models[model_name]

threshold = st.slider(
    f"Decision threshold for {model_name}:",
    0.0, 1.0, 0.50, 0.01
)

# -----------------------------
# Instance selection
# -----------------------------
st.subheader("Select an instance from the dataset")

idx = st.number_input("Row index:", 0, len(df)-1, 0)

# Extract row
row = df.iloc[[idx]].copy()

# Remove target column
cols_to_remove = ["IS_FRAUDSTER", "ID", "USER_ID"]
row = row.drop(columns=[c for c in cols_to_remove if c in row.columns])

# -----------------------------
# Editable table
# -----------------------------
st.subheader("Edit feature values")
edited_row = st.data_editor(row, num_rows="fixed", use_container_width=True)
st.markdown(""" For further explanation of how each feature was engineered, please refer to the Jupyter Notebook where the full feature engineering pipeline is documented. """)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    X = edited_row.copy()

    # Scale only for Logistic Regression
    if model_name == "Logistic Regression":
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    proba = model.predict_proba(X_scaled)[0][1]
    pred = int(proba >= threshold)

    st.metric("Fraud Probability", f"{proba:.2%}")
    st.metric("Prediction", "🚨 Fraudster" if pred else "✅ Legit")


    st.markdown("""
    ### Interpretation
    This prediction is based on the edited feature values.
    Adjusting behavioural or transactional attributes changes the fraud risk score.
    """)

# -----------------------------
# Threshold Optimization Tables
# -----------------------------
st.markdown("---")
st.subheader("📊 Threshold Optimization Results")

log_reg_table = pd.DataFrame({
    "Threshold": [0.5,0.4,0.3,0.2,0.1,0.05,0.01],
    "Recall": ["81.67%","86.67%","88.33%","93.33%","95.00%","98.33%","100.00%"],
    "Precision": ["14.16%","11.82%","9.20%","7.66%","6.26%","5.57%","3.77%"],
    "Accuracy": ["84.51%","80.09%","73.35%","65.86%","56.91%","49.62%","23.03%"],
    "F1": ["24.14%","20.80%","16.67%","14.16%","11.74%","10.54%","7.27%"],
    "FN": [11,8,7,4,3,1,0],
    "FP": [297,388,523,675,854,1001,1531]
})

rf_table = pd.DataFrame({
    "Threshold": [0.5,0.4,0.3,0.2,0.1,0.05,0.01],
    "Recall": ["80.00%","86.67%","95.00%","98.33%","100.00%","100.00%","100.00%"],
    "Precision": ["16.16%","11.53%","7.63%","5.35%","4.17%","3.85%","3.60%"],
    "Accuracy": ["86.88%","79.54%","65.16%","47.46%","30.62%","24.69%","19.21%"],
    "F1": ["26.89%","20.35%","14.13%","10.15%","8.00%","7.42%","6.95%"],
    "FN": [12,8,3,1,0,0,0],
    "FP": [249,399,690,1044,1380,1498,1607]
})

gb_table = pd.DataFrame({
    "Threshold": [0.5,0.4,0.3,0.2,0.1,0.05,0.01],
    "Recall": ["81.67%","83.33%","85.00%","86.67%","88.33%","93.33%","96.67%"],
    "Precision": ["16.39%","14.79%","13.28%","11.50%","9.64%","8.50%","5.85%"],
    "Accuracy": ["86.88%","85.02%","82.81%","79.49%","74.66%","69.48%","52.99%"],
    "F1": ["27.30%","25.13%","22.97%","20.31%","17.38%","15.58%","11.04%"],
    "FN": [11,10,9,8,7,4,2],
    "FP": [250,288,333,400,497,603,933]
})

with st.expander("Logistic Regression — Threshold Performance"):
    st.dataframe(log_reg_table, use_container_width=True)

with st.expander("Random Forest — Threshold Performance"):
    st.dataframe(rf_table, use_container_width=True)

with st.expander("Gradient Boosting — Threshold Performance"):
    st.dataframe(gb_table, use_container_width=True)

st.markdown("""
### Summary
- Logistic Regression reaches perfect recall at very low thresholds but generates extremely high false alarms.
- Random Forest achieves **100% recall at threshold = 0.10**, with a better balance between recall and false positives.
- Gradient Boosting performs well but does not reach 100% recall without a large increase in false alarms.

Random Forest is the most stable and suitable model for fraud detection in this context.
""")


