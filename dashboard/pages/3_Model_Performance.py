import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

load_dotenv(dotenv_path="../.env", override=True)

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")
st.title("Model Performance")
st.markdown("Comparison of 5 classification models trained to predict customer churn.")

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

@st.cache_data
def train_and_evaluate():
    df = load_data()

    drop_cols = [c for c in ['row_number', 'customer_id', 'surname'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['geography'] = le_geo.fit_transform(df['geography'])
    df['gender'] = le_gender.fit_transform(df['gender'])

    X = df.drop(columns=['exited'])
    y = df['exited'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    results = []
    feature_importance = None
    cm = None
    fpr = tpr = roc_auc = None
    prec_curve = rec_curve = None

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "F1 Score": f1_score(y_test, preds),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds)
        })

        if name == "Random Forest":
            feature_importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            cm = confusion_matrix(y_test, preds)
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)

    return pd.DataFrame(results), feature_importance, cm, fpr, tpr, roc_auc, prec_curve, rec_curve

with st.spinner("Training models... this may take a few seconds"):
    results_df, feature_importance, cm, fpr, tpr, roc_auc, prec_curve, rec_curve = train_and_evaluate()

st.markdown("---")
st.subheader("Model Comparison")

results_display = results_df.copy()
for col in ["F1 Score", "Accuracy", "Precision", "Recall"]:
    results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}")
st.dataframe(results_display, use_container_width=True, hide_index=True)

fig = px.bar(results_df, x="Model", y=["F1 Score", "Accuracy", "Precision", "Recall"],
             barmode="group", title="Model Performance Comparison",
             color_discrete_sequence=['#E8674A', '#1F2733', '#4A5568', '#A0AEC0'])
st.plotly_chart(fig, use_container_width=True)

rf_row = results_df[results_df["Model"] == "Random Forest"].iloc[0]
xgb_row = results_df[results_df["Model"] == "XGBoost"].iloc[0]
st.success(
    f"**Best Model: Random Forest** — F1 Score: {rf_row['F1 Score']:.4f}, "
    f"offering the best balance of precision ({rf_row['Precision']*100:.2f}%) and recall ({rf_row['Recall']*100:.2f}%) "
    f"for identifying at-risk customers while minimizing false alarms. "
    f"XGBoost is a close second (F1: {xgb_row['F1 Score']:.4f}) and could serve as an alternative "
    f"if marginally higher recall is prioritized over precision."
)

st.markdown("---")
st.subheader("Feature Importance (Random Forest)")

fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
             title="What Drives Churn Predictions",
             color="Importance", color_continuous_scale=['#1F2733', '#E8674A'])
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Random Forest — Detailed Diagnostics")

col1, col2 = st.columns(2)

with col1:
    cm_flipped = cm[::-1]
    fig = go.Figure(data=go.Heatmap(
        z=cm_flipped, x=["Predicted: Retained", "Predicted: Churned"],
        y=["Actual: Churned", "Actual: Retained"],
        text=cm_flipped, texttemplate="%{text}", textfont={"size": 20},
        colorscale=[[0, '#1F2733'], [1, '#E8674A']]
    ))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(color='#E8674A', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='#4A5568', dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode='lines', name='Precision-Recall', line=dict(color='#E8674A', width=3)))
fig.update_layout(title="Precision-Recall Curve (Random Forest)", xaxis_title="Recall", yaxis_title="Precision")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Key Insight")
st.markdown("""
Feature importance confirms **age** as the dominant predictor, consistent with earlier EDA and hypothesis testing.

Notably, **credit_score** ranks highly in the model despite appearing to have almost no relationship with churn in isolated EDA analysis (flat ~19-22% across credit bands). This suggests credit_score **interacts** with other variables (like age or balance) in ways that univariate analysis cannot detect — a key advantage of tree-based models over simple correlation or single-variable EDA.

**Geography, gender,** and **has_cr_card** contribute comparatively little to the model's predictions, despite geography and gender showing strong standalone statistical significance in the Chi-square tests — indicating their effect may be partially captured or absorbed by other correlated features like age and balance.
""")

st.markdown("---")
st.info("Random Forest was selected as the production model due to its strong F1 score and interpretability via feature importance, balancing performance and explainability for business stakeholders.")