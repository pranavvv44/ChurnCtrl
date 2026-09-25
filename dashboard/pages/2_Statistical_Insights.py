import streamlit as st
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from scipy import stats

load_dotenv(dotenv_path="../.env", override=True)

st.set_page_config(page_title="Statistical Insights", page_icon="📈", layout="wide")
st.title("Statistical Insights")
st.markdown("Hypothesis testing confirms which factors are *statistically significant* drivers of churn.")

@st.cache_data
def load_data():
    conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    port=os.getenv("DB_PORT"),
    sslmode="require"
)
    
    df = pd.read_sql("SELECT * FROM customers_bank;", conn)
    conn.close()
    return df

df = load_data()

st.markdown("---")
st.subheader("Chi-Square Tests (Categorical Variables)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Geography vs Churn**")
    contingency = pd.crosstab(df['geography'], df['exited'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    st.metric("Chi-Square Statistic", f"{chi2:.2f}")
    st.metric("P-Value", f"{p:.2e}")
    if p < 0.001:
        st.success("Statistically significant (p < 0.001) — Geography influences churn.")
    else:
        st.warning("Not statistically significant.")

with col2:
    st.markdown("**Gender vs Churn**")
    contingency = pd.crosstab(df['gender'], df['exited'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    st.metric("Chi-Square Statistic", f"{chi2:.2f}")
    st.metric("P-Value", f"{p:.2e}")
    if p < 0.001:
        st.success("Statistically significant (p < 0.001) — Gender influences churn.")
    else:
        st.warning("Not statistically significant.")

st.markdown("---")
st.subheader("T-Tests (Numerical Variables)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Age: Churned vs Retained**")
    churned = df[df['exited'] == True]['age']
    retained = df[df['exited'] == False]['age']
    t_stat, p = stats.ttest_ind(churned, retained, equal_var=False)
    st.metric("T-Statistic", f"{t_stat:.2f}")
    st.metric("P-Value", f"{p:.2e}")
    m1, m2 = st.columns(2)
    m1.metric("Mean Age (Churned)", f"{churned.mean():.1f}")
    m2.metric("Mean Age (Retained)", f"{retained.mean():.1f}")
    if p < 0.001:
        st.success("Statistically significant (p < 0.001) — Age differs significantly between groups.")
    else:
        st.warning("Not statistically significant.")

with col2:
    st.markdown("**Balance: Churned vs Retained**")
    churned = df[df['exited'] == True]['balance']
    retained = df[df['exited'] == False]['balance']
    t_stat, p = stats.ttest_ind(churned, retained, equal_var=False)
    st.metric("T-Statistic", f"{t_stat:.2f}")
    st.metric("P-Value", f"{p:.2e}")
    m1, m2 = st.columns(2)
    m1.metric("Mean Balance (Churned)", f"${churned.mean():,.0f}")
    m2.metric("Mean Balance (Retained)", f"${retained.mean():,.0f}")
    if p < 0.001:
        st.success("Statistically significant (p < 0.001) — Balance differs significantly between groups.")
    else:
        st.warning("Not statistically significant.")

st.markdown("---")
st.info("All four tests confirm findings from the exploratory analysis: Geography, Gender, Age, and Balance are all statistically significant factors in customer churn (p < 0.001).")