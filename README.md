# ============================================
# Credit Risk Scorer — Streamlit App
# ============================================

import streamlit as st          # builds the web app interface
import pandas as pd             # handles data as dataframe
import numpy as np              # numerical operations
import joblib                   # loads saved model and scaler


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

# ---- App Header ----
st.title("💳 Credit Risk Scorer")
st.markdown("Predict whether a customer is likely to default on their loan in the next 2 years.")
st.markdown("---")

# ---- Input Section ----
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)   # creates two side by side columns for clean layout

with col1:
    age = st.slider("Age", min_value=18, max_value=98, value=45)
    # age slider — min 18, max 98, default 45

    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=5400)
    # monthly income input — default is dataset median

    revolving_utilization = st.slider("Revolving Credit Utilization (0 to 1.5)", min_value=0.0, max_value=1.5, value=0.3)
    # how much of credit limit is being used

    debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=2842.0, value=0.3)
    # monthly debt payments divided by monthly income

    number_of_dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=0)
    # number of people financially dependent on customer

with col2:
    late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    # number of times 30-59 days past due

    late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)
    # number of times 60-89 days past due

    late_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)
    # number of times 90+ days past due

    open_credit_lines = st.number_input("Open Credit Lines and Loans", min_value=0, max_value=50, value=8)
    # total number of open loans and credit lines

    real_estate_loans = st.number_input("Real Estate Loans or Lines", min_value=0, max_value=20, value=1)
    # number of mortgage and real estate loans

# ---- Auto Calculate Engineered Features ----
total_late_payments = late_30_59 + late_60_89 + late_90
# TotalLatePayments = sum of all 3 late payment columns

income_per_dependent = monthly_income / (number_of_dependents + 1)
# IncomePerDependent = income divided by dependents+1 to avoid division by zero

# ---- Predict Button ----
st.markdown("---")

if st.button("🔍 Predict Default Risk"):   # runs everything below when button is clicked

    # Build input dataframe in exact same column order as training data
    input_data = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [revolving_utilization],
        'age': [age],
        'NumberOfTime30-59DaysPastDueNotWorse': [late_30_59],
        'DebtRatio': [debt_ratio],
        'MonthlyIncome': [monthly_income],
        'NumberOfOpenCreditLinesAndLoans': [open_credit_lines],
        'NumberOfTimes90DaysLate': [late_90],
        'NumberRealEstateLoansOrLines': [real_estate_loans],
        'NumberOfTime60-89DaysPastDueNotWorse': [late_60_89],
        'NumberOfDependents': [number_of_dependents],
        'TotalLatePayments': [total_late_payments],
        'IncomePerDependent': [income_per_dependent]
    })

    # Scale input using same scaler used during training
    input_scaled = scaler.transform(input_data)   # applies saved scaler to input

    # Get prediction and probability
    prediction = model.predict(input_scaled)[0]             # 0 or 1
    probability = model.predict_proba(input_scaled)[0][1]   # probability of default

    # ---- Display Result ----
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This customer is likely to default")
    else:
        st.success(f"✅ LOW RISK — This customer is unlikely to default")

    st.metric(
        label="Default Probability",
        value=f"{round(probability * 100, 2)}%"
    )

    # Show auto calculated features
    st.markdown("---")
    st.subheader("🔧 Auto Calculated Features")
    st.write(f"Total Late Payments: **{total_late_payments}**")
    st.write(f"Income Per Dependent: **${round(income_per_dependent, 2)}**")
