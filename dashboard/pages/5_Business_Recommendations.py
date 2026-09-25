import streamlit as st
import pandas as pd
import psycopg2
import os
import plotly.express as px
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)

st.set_page_config(page_title="Business Recommendations", page_icon="💼", layout="wide")
st.title(" Business Recommendations")
st.markdown("Translating statistical findings and model insights into actionable retention strategy.")

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
st.subheader("Executive Summary")
st.markdown("""
With an overall churn rate of **20.37%**, roughly 1 in 5 customers leave the bank — a significant 
retention challenge. Statistical testing and machine learning both confirm that **age, geography, 
gender, and balance** are strong, statistically validated drivers of churn (all p < 0.001). The 
Random Forest model (F1 = 0.6034) can identify at-risk customers with 80% precision and 87% accuracy, enabling 
targeted, cost-effective retention campaigns rather than broad, expensive blanket outreach.
""")

st.markdown("---")
st.subheader("Key Risk Segments")

col1, col2, col3 = st.columns(3)

with col1:
    older_churn = df[df['age'] >= 51]['exited'].mean() * 100
    st.metric("Churn Rate (Age 51+)", f"{older_churn:.1f}%")
    st.caption("vs. overall 20.37% — older customers churn at a substantially higher rate.")

with col2:
    germany_churn = df[df['geography'] == 'Germany']['exited'].mean() * 100
    st.metric("Churn Rate (Germany)", f"{germany_churn:.1f}%")
    st.caption("Highest among all three geographies — worth investigating local competition or service gaps.")

with col3:
    single_product_churn = df[df['num_of_products'] == 1]['exited'].mean() * 100
    st.metric("Churn Rate (1 Product Only)", f"{single_product_churn:.1f}%")
    st.caption("Customers with only one product show elevated churn — weak relationship depth.")

st.markdown("---")
st.subheader("Recommendations")

st.markdown("""
**1. Prioritize retention outreach for customers aged 51+**  
This segment shows a substantially higher churn rate than the overall average. Proactive relationship 
management — dedicated account reviews, loyalty benefits, or senior-focused financial products — could 
meaningfully reduce attrition here.

**2. Investigate the Germany market specifically**  
Germany's churn rate significantly exceeds France and Spain despite similar product offerings. This may 
indicate competitive pressure, service quality gaps, or regional expectations not being met — warrants 
a dedicated market research review.

**3. Drive cross-sell to single-product customers**  
Customers holding only one product are far more likely to leave. Bundling incentives (e.g. discounted 
rates for a second product, a welcome bonus for adding a savings account) can deepen the relationship 
and reduce churn risk.

**4. Use the Random Forest model for proactive, targeted campaigns**  
Rather than blanket retention offers, use the model's 80% precision to identify the highest-risk 
customers specifically, focusing budget and staff time where it will have the greatest impact.

**5. Monitor credit_score interactions, not just balance**  
Feature importance analysis revealed credit_score interacts meaningfully with age and balance in ways 
simple EDA didn't show. Consider incorporating credit health check-ins as part of retention conversations, 
even for customers who don't appear "at risk" on the surface.
""")

st.markdown("---")
st.subheader("Churn Rate by Age Group")

df['age_group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
age_group_churn = df.groupby('age_group', observed=True)['exited'].mean().reset_index()
age_group_churn['exited'] = age_group_churn['exited'] * 100

fig = px.bar(age_group_churn, x='age_group', y='exited',
             title="Churn Rate by Age Group (%)",
             color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("These recommendations are derived from statistically validated findings (Chi-square and T-tests, all p < 0.001) and machine learning feature importance analysis, ensuring business decisions are grounded in data rather than assumption.")