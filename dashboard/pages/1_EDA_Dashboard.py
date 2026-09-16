import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=True)


st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 EDA Dashboard")

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

df_full = load_data()
overall_churn_rate = df_full['exited'].mean() * 100
overall_avg_age = df_full['age'].mean()
overall_avg_balance = df_full['balance'].mean()
overall_count = len(df_full)

# Sidebar filters
st.sidebar.header("Filters")
geo_filter = st.sidebar.multiselect("Geography", options=df_full['geography'].unique(), default=df_full['geography'].unique())
gender_filter = st.sidebar.multiselect("Gender", options=df_full['gender'].unique(), default=df_full['gender'].unique())
age_range = st.sidebar.slider("Age Range", int(df_full['age'].min()), int(df_full['age'].max()),
                               (int(df_full['age'].min()), int(df_full['age'].max())))
products_filter = st.sidebar.multiselect("Number of Products", options=sorted(df_full['num_of_products'].unique()),
                                          default=sorted(df_full['num_of_products'].unique()))
active_filter = st.sidebar.radio("Active Member", options=["All", "Active Only", "Inactive Only"])
cr_card_filter = st.sidebar.radio("Has Credit Card", options=["All", "Yes Only", "No Only"])

df = df_full[
    (df_full['geography'].isin(geo_filter)) &
    (df_full['gender'].isin(gender_filter)) &
    (df_full['age'].between(age_range[0], age_range[1])) &
    (df_full['num_of_products'].isin(products_filter))
]

if active_filter == "Active Only":
    df = df[df['is_active_member'] == True]
elif active_filter == "Inactive Only":
    df = df[df['is_active_member'] == False]

if cr_card_filter == "Yes Only":
    df = df[df['has_cr_card'] == True]
elif cr_card_filter == "No Only":
    df = df[df['has_cr_card'] == False]

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
churn_rate = df['exited'].mean() * 100 if len(df) > 0 else 0
avg_age = df['age'].mean() if len(df) > 0 else 0
avg_balance = df['balance'].mean() if len(df) > 0 else 0

col1.metric("Total Customers", f"{len(df):,}",
            delta=f"{len(df) - overall_count:,} vs full dataset", delta_color="off")
col2.metric("Churn Rate", f"{churn_rate:.2f}%",
            delta=f"{churn_rate - overall_churn_rate:+.2f}% vs overall", delta_color="inverse")
col3.metric("Avg Age", f"{avg_age:.1f}" if len(df) > 0 else "N/A",
            delta=f"{avg_age - overall_avg_age:+.1f} yrs vs overall" if len(df) > 0 else None, delta_color="off")
col4.metric("Avg Balance", f"${avg_balance:,.0f}" if len(df) > 0 else "N/A",
            delta=f"${avg_balance - overall_avg_balance:+,.0f} vs overall" if len(df) > 0 else None, delta_color="off")

st.markdown("---")

# Gauge chart + composition donut
st.subheader("Risk Overview")
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_rate,
        delta={'reference': overall_churn_rate, 'increasing': {'color': "#E8674A"}, 'decreasing': {'color': "#4A9D6E"}},
        title={'text': "Segment Churn Rate (%) vs Baseline", 'font': {'size': 18}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "#E8674A"},
            'steps': [
                {'range': [0, 15], 'color': "#1F2733"},
                {'range': [15, 30], 'color': "#2E333D"},
                {'range': [30, 50], 'color': "#3D4450"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': overall_churn_rate
            }
        }
    ))
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    churned_df = df[df['exited'] == True]
    revenue_at_risk = churned_df['balance'].sum()
    avg_churned_balance = churned_df['balance'].mean() if len(churned_df) > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=revenue_at_risk,
        number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 60, 'color': '#E8674A'}},
        title={'text': "Estimated Revenue at Risk<br><span style='font-size:0.85em;color:#CBD5E0'>Total balance held by churned customers in this segment</span>", 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0.35, 1]}
    ))
    fig.add_trace(go.Indicator(
        mode="number",
        value=avg_churned_balance,
        number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 32, 'color': '#A0AEC0'}},
        title={'text': "Avg balance per churned customer", 'font': {'size': 18}},
        domain={'x': [0, 1], 'y': [0, 0.3]}
    ))
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Churn by Geography and Gender
col1, col2 = st.columns(2)
with col1:
    geo_churn = df.groupby('geography')['exited'].mean().reset_index()
    geo_churn['exited'] = geo_churn['exited'] * 100
    fig = px.bar(geo_churn, x='geography', y='exited',
                 title="Churn Rate by Geography (%)",
                 color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    gender_churn = df.groupby('gender')['exited'].mean().reset_index()
    gender_churn['exited'] = gender_churn['exited'] * 100
    fig = px.bar(gender_churn, x='gender', y='exited',
                 title="Churn Rate by Gender (%)",
                 color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

# Compound risk: Geography x Gender heatmap
st.markdown("---")
st.subheader("Compound Risk Analysis")
st.markdown("Does being both a specific gender AND from a specific geography compound churn risk?")

compound = df.groupby(['geography', 'gender'])['exited'].mean().reset_index()
compound['exited'] = compound['exited'] * 100
compound_pivot = compound.pivot(index='gender', columns='geography', values='exited')

fig = px.imshow(compound_pivot, text_auto=".1f", aspect="auto",
                 title="Churn Rate (%) by Geography × Gender",
                 color_continuous_scale=['#1F2733', '#E8674A'],
                 labels=dict(color="Churn Rate (%)"))
st.plotly_chart(fig, use_container_width=True)

# Sunburst: Geography -> Gender -> Churn
col1, col2 = st.columns(2)
with col1:
    df_sunburst = df.copy()
    df_sunburst['churn_status'] = df_sunburst['exited'].map({True: 'Churned', False: 'Retained'})
    fig = px.sunburst(df_sunburst, path=['geography', 'gender', 'churn_status'],
                       title="Customer Breakdown: Geography → Gender → Churn Status",
                       color_discrete_sequence=['#E8674A', '#1F2733', '#4A5568', '#A0AEC0', '#8B4A3D', '#2E333D'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    tenure_churn = df.groupby('tenure')['exited'].mean().reset_index()
    tenure_churn['exited'] = tenure_churn['exited'] * 100
    fig = px.line(tenure_churn, x='tenure', y='exited', markers=True,
                   title="Churn Rate by Tenure (Years)",
                   color_discrete_sequence=['#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

# Age distribution and products
col1, col2 = st.columns(2)
with col1:
    df_labeled = df.copy()
    df_labeled['exited'] = df_labeled['exited'].map({False: 'Retained', True: 'Churned'})
    fig = px.box(df_labeled, x='exited', y='age', color='exited',
                 title="Age Distribution by Churn Status",
                 color_discrete_sequence=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    products_churn = df.groupby('num_of_products')['exited'].mean().reset_index()
    products_churn['exited'] = products_churn['exited'] * 100
    fig = px.bar(products_churn, x='num_of_products', y='exited',
                 title="Churn Rate by Number of Products (%)",
                 color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

# Active member and credit score bins
col1, col2 = st.columns(2)
with col1:
    df_labeled2 = df.copy()
    df_labeled2['active_status'] = df_labeled2['is_active_member'].map({True: 'Active', False: 'Inactive'})
    active_churn = df_labeled2.groupby('active_status')['exited'].mean().reset_index()
    active_churn['exited'] = active_churn['exited'] * 100
    fig = px.bar(active_churn, x='active_status', y='exited',
                 title="Churn Rate by Active Member Status (%)",
                 color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    df_credit = df.copy()
    df_credit['credit_bin'] = pd.cut(df_credit['credit_score'], bins=[300, 500, 600, 700, 800, 900],
                                       labels=['300-500', '500-600', '600-700', '700-800', '800-900'])
    credit_churn = df_credit.groupby('credit_bin', observed=True)['exited'].mean().reset_index()
    credit_churn['exited'] = credit_churn['exited'] * 100
    fig = px.bar(credit_churn, x='credit_bin', y='exited',
                 title="Churn Rate by Credit Score Band (%)",
                 color='exited', color_continuous_scale=['#1F2733', '#E8674A'])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Notice how flat this is — credit score alone shows little relationship with churn, but interacts significantly with other features in the ML model (see Model Performance page).")

# Correlation heatmap
st.markdown("---")
st.subheader("Correlation Heatmap")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                 color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)