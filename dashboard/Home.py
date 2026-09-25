import streamlit as st

st.set_page_config(
    page_title="ChurnCtrl",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" ChurnCtrl")
st.markdown("### Predictive Banking Customer Churn Analysis and Retention Insights Platform")

st.markdown("""
---
This project analyzes customer churn for a bank using **10,000 customer records** with 14 features, including demographics, account information, and activity status. The goal is to identify churn drivers, build predictive models, and provide actionable insights for retention strategies.
""")

st.markdown("---")
st.subheader("Business Problem")
st.markdown("""
A Bank's leadership has identified an overall customer churn rate of **20.37%** through routine 
reporting and is concerned about its impact on revenue, given that acquiring a new customer costs 
significantly more than retaining an existing one. However, the underlying drivers of this churn 
and the specific customer segments most at risk remain unclear. The analytics team has been asked 
to investigate **why** customers are leaving and **where** retention efforts should be prioritized.
""")

st.subheader("Key Business Questions")
st.markdown("""
1. **Segment risk** — Which customer segments (geography, age, gender, activity status, number of 
   products) show the highest churn risk?
2. **Compound risk** — Do these risk factors interact (e.g., does being both female and German 
   compound churn risk beyond either factor alone)?
3. **Statistical validity** — Are the differences observed across segments statistically significant, 
   or could they be explained by random variation?
4. **Financial behavior** — Does account balance, credit score, or tenure relate meaningfully to churn?
5. **Predictive modeling** — Can we build a model to flag at-risk customers in advance, and which 
   factors matter most in that prediction?
6. **Business narrative** — Can we generate clear, automated, business-ready insights and recommendations 
   without manual reporting each time, and build an interactive dashboard for client presentations?
""")

st.markdown("---")
st.subheader("Key Findings")
st.markdown("""
- Overall churn rate: **20.37%**
- Best model: **Random Forest** — F1: **0.6034**, Accuracy: **87.05%**, Precision: **80.08%**, Recall: **48.40%**
- Top churn driver: **Age** (22.4% feature importance)
""")

st.markdown("---")
st.subheader("Navigate Using the Sidebar")
st.markdown("""
-  **EDA Dashboard** — visual exploration of churn patterns
-  **Statistical Insights** — hypothesis test results
-  **Model Performance** — model comparison + feature importance
-  **Live Predictor** — enter customer details, get churn prediction + AI-generated insight
-  **Business Recommendations** — actionable strategy based on findings
""")

st.markdown("---")
st.caption("Built with Neon & PostgreSQL · Python · Pandas & NumPy · Seaborn & Matplotlib · Scipy · Scikit-learn  · Random Forest XGBoost KNN · Groq API · Streamlit")