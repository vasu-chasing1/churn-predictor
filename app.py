import streamlit as st
import pandas as pd
import joblib

# ---------- Load saved model, scaler, and column order ----------
model = joblib.load('churn_model.pkl')
scaler = joblib.load('churn_scaler.pkl')
columns = joblib.load('churn_columns.pkl')

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to predict churn risk.")

# ---------- User inputs ----------
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# ---------- Build a single-row DataFrame matching training features ----------
if st.button("Predict"):
    input_data = dict.fromkeys(columns, 0)

    # Binary / direct fields
    input_data['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
    input_data['Partner'] = 1 if partner == 'Yes' else 0
    input_data['Dependents'] = 1 if dependents == 'Yes' else 0
    input_data['tenure'] = tenure
    input_data['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges

    # One-hot fields — set to 1 only if that exact column exists
    # (the baseline category for each group, e.g. DSL/No/Month-to-month, has no column — leaving all its group's columns at 0 correctly represents it)
    one_hot_selections = {
        f'InternetService_{internet_service}': 1,
        f'OnlineSecurity_{online_security}': 1,
        f'OnlineBackup_{online_backup}': 1,
        f'DeviceProtection_{device_protection}': 1,
        f'TechSupport_{tech_support}': 1,
        f'StreamingTV_{streaming_tv}': 1,
        f'StreamingMovies_{streaming_movies}': 1,
        f'Contract_{contract}': 1,
        f'PaymentMethod_{payment_method}': 1,
    }
    for col, val in one_hot_selections.items():
        if col in input_data:
            input_data[col] = val

    # Build DataFrame in the exact column order the model expects
    input_df = pd.DataFrame([input_data])[columns]

    # Scale the same numeric columns scaled during training
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # ---------- Predict ----------
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High churn risk — probability: {probability:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {probability:.1%}")