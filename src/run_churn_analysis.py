import pandas as pd
import numpy as np
import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
import joblib
from groq import Groq
from datetime import datetime
from scipy.stats import chi2_contingency, ttest_ind

load_dotenv(".env")

def load_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT"),
        sslmode="require"
    )
    df = pd.read_sql("SELECT * FROM customers_bank", conn)
    conn.close()
    return df

def feature_engineer(df):
    df = df.copy()
    df['balance_salary_ratio'] = df['balance'] / df['estimated_salary']
    df['high_value_customer'] = ((df['balance'] > df['balance'].median()) & 
                                   (df['estimated_salary'] > df['estimated_salary'].median())).astype(int)
    df['engagement_score'] = df['has_cr_card'].astype(int) + df['is_active_member'].astype(int) + df['num_of_products']
    return df

def run_hypothesis_tests(df):
    geo_table = pd.crosstab(df['geography'], df['exited'])
    chi2_geo, p_geo, _, _ = chi2_contingency(geo_table)
    
    gender_table = pd.crosstab(df['gender'], df['exited'])
    chi2_gender, p_gender, _, _ = chi2_contingency(gender_table)
    
    churned_age = df[df['exited']==True]['age']
    stayed_age = df[df['exited']==False]['age']
    t_age, p_age = ttest_ind(churned_age, stayed_age)
    
    return {
        'geo_significant': p_geo < 0.05,
        'p_geo': p_geo,
        'gender_significant': p_gender < 0.05,
        'p_gender': p_gender,
        'age_significant': p_age < 0.05,
        'p_age': p_age
    }

def run_model_predictions(df):
    model = joblib.load('models/random_forest_churn_model.pkl')
    
    df_model = df.drop(columns=['row_number', 'customer_id', 'surname', 'exited'], errors='ignore')
    df_model = pd.get_dummies(df_model, columns=['geography', 'gender'], drop_first=True)
    
    for col in model.feature_names_in_:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[model.feature_names_in_]
    
    predictions = model.predict(df_model)
    probabilities = model.predict_proba(df_model)[:, 1]
    
    high_risk_count = predictions.sum()
    avg_risk_prob = probabilities.mean() * 100
    
    return high_risk_count, avg_risk_prob

def calculate_stats(df, hyp_results, high_risk_count, avg_risk_prob):
    churn_rate = df['exited'].mean() * 100
    geo_churn = df.groupby('geography')['exited'].mean() * 100
    gender_churn = df.groupby('gender')['exited'].mean() * 100
    age_churn_51_60 = df[(df['age']>=51) & (df['age']<=60)]['exited'].mean() * 100
    
    stats = f"""
Overall churn rate: {churn_rate:.2f}% (based on {len(df)} customers)

Statistically validated findings:
- Geography effect on churn: {'SIGNIFICANT' if hyp_results['geo_significant'] else 'not significant'} (p={hyp_results['p_geo']:.4f}). Germany {geo_churn.get('Germany', 0):.2f}%, France {geo_churn.get('France', 0):.2f}%, Spain {geo_churn.get('Spain', 0):.2f}%
- Gender effect on churn: {'SIGNIFICANT' if hyp_results['gender_significant'] else 'not significant'} (p={hyp_results['p_gender']:.4f}). Female {gender_churn.get('Female', 0):.2f}%, Male {gender_churn.get('Male', 0):.2f}%
- Age effect on churn: {'SIGNIFICANT' if hyp_results['age_significant'] else 'not significant'} (p={hyp_results['p_age']:.4f}). 51-60 age group churn rate: {age_churn_51_60:.2f}%

Machine Learning model predictions (Random Forest, F1=0.60, Precision=80%, Recall=48%):
- Model flags {high_risk_count} customers as high churn risk out of {len(df)} total
- Average predicted churn probability across all customers: {avg_risk_prob:.2f}%
"""
    return stats

def generate_report(stats):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""You are a data analyst writing a business report for bank leadership.
Based on the following current, statistically validated churn statistics and 
machine learning model predictions, write a professional business report with: 
Executive Summary, Key Findings, and Actionable Recommendations. Keep it 
concise and business-focused.

Findings:
{stats}
"""
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def save_report(report):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"outputs/churn_report_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {filename}")
    return filename

def main():
    print("Starting churn analysis pipeline...")
    df = load_data()
    print(f"Loaded {len(df)} rows from database")
    
    df_engineered = feature_engineer(df)
    print("Feature engineering complete")
    
    hyp_results = run_hypothesis_tests(df)
    print("Hypothesis tests complete")
    
    high_risk_count, avg_risk_prob = run_model_predictions(df_engineered)
    print(f"Model predictions complete: {high_risk_count} high-risk customers identified")
    
    stats = calculate_stats(df, hyp_results, high_risk_count, avg_risk_prob)
    print("Statistics compiled")
    
    report = generate_report(stats)
    print("AI report generated")
    
    filepath = save_report(report)
    print("Pipeline complete")
    return filepath

if __name__ == "__main__":
    main()