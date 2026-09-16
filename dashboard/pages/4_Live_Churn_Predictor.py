import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import os
import joblib
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

st.set_page_config(page_title="Live Churn Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Live Churn Predictor")
st.markdown("Enter customer details to predict churn risk and get an AI-generated retention insight.")

@st.cache_data
def load_medians():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT"),
        sslmode="require"
    )

    df = pd.read_sql("SELECT balance, estimated_salary FROM customers_bank;", conn)
    conn.close()
    return df["balance"].median(), df["estimated_salary"].median()


median_balance, median_salary = load_medians()


@st.cache_resource
def load_model_and_scaler():
    MODEL_DIR = BASE_DIR / "models"

    model = joblib.load(MODEL_DIR / "random_forest_churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    return model, scaler


model, scaler = load_model_and_scaler()

st.markdown("---")
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)

with col3:
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])


if st.button("Predict Churn Risk", type="primary"):
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val = 1 if is_active_member == "Yes" else 0

    balance_salary_ratio = balance / estimated_salary if estimated_salary != 0 else 0
    high_value_customer = int(
        (balance > median_balance) and (estimated_salary > median_salary)
    )
    engagement_score = has_cr_card_val + is_active_val + num_of_products

    geography_Germany = 1 if geography == "Germany" else 0
    geography_Spain = 1 if geography == "Spain" else 0
    gender_Male = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([{
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": has_cr_card_val,
        "is_active_member": is_active_val,
        "estimated_salary": estimated_salary,
        "balance_salary_ratio": balance_salary_ratio,
        "high_value_customer": high_value_customer,
        "engagement_score": engagement_score,
        "geography_Germany": geography_Germany,
        "geography_Spain": geography_Spain,
        "gender_Male": gender_Male
    }])

    input_df = input_df[model.feature_names_in_]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ **High Churn Risk**")
        else:
            st.success("✅ **Low Churn Risk**")

    with col2:
        st.metric("Churn Probability", f"{probability:.1f}%")

    st.markdown("---")
    st.subheader("AI-Generated Insight")

    with st.spinner("Generating personalized insight..."):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""You are a bank customer retention analyst. A machine learning model 
predicted the following customer has a {probability:.1f}% probability of churning 
(prediction: {'WILL CHURN' if prediction == 1 else 'WILL STAY'}).

Customer profile:
- Geography: {geography}
- Gender: {gender}
- Age: {age}
- Credit Score: {credit_score}
- Balance: ${balance:,.0f}
- Estimated Salary: ${estimated_salary:,.0f}
- Tenure: {tenure} years
- Number of Products: {num_of_products}
- Has Credit Card: {has_cr_card}
- Active Member: {is_active_member}

Write a short, personalized business insight (3-4 sentences) explaining likely 
reasons for this prediction based on the profile, and suggest 1-2 specific 
retention actions if the risk is high, or engagement actions if the risk is low."""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )

        insight = response.choices[0].message.content

    st.markdown(insight)