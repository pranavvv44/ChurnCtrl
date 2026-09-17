🏦 ChurnCtrl — Predictive Banking Customer Churn Analysis & Retention Insights Platform

End-to-End Data Analytics + Machine Learning project analyzing banking customer churn across 10,000 customer records and 14 features. The project combines Python, SQL, statistical analysis, machine learning, automated AI insights, and an interactive Streamlit dashboard to identify churn drivers, predict at-risk customers, and support retention decisions.

🎯 Business Problem

A bank has identified an overall customer churn rate of 20.37% and is concerned about its impact on revenue, since acquiring new customers costs significantly more than retaining existing ones.

However, underlying churn drivers and high-risk customer segments remain unclear.

The analytics team needs to determine:

Which customer segments have highest churn risk?
Do multiple risk factors compound churn risk?
Are observed differences statistically significant?
Does financial behavior relate to churn?
Can customers at risk be identified in advance?
Which factors matter most for prediction?
Can insights and recommendations be generated automatically?
Can findings be presented through an interactive dashboard?
🏗️ Project Architecture
📊 Dataset

10,000 customer records • 14 features

Key Features
Geography
Gender
Age
Credit Score
Tenure
Account Balance
Number of Products
Credit Card Status
Active Membership
Estimated Salary
Customer activity information
Churn status

Target
Exited

🛠️ Tools & Technologies

Tool	Purpose
🐍 Python	Data analysis & preprocessing
🐼 Pandas	Data manipulation
🔢 NumPy	Numerical analysis
🗄️ PostgreSQL	Database & SQL analysis
SQL	Business analysis
📐 Scikit-learn , Scipy 	Statistical analysis & ML
🌲 Random Forest	Churn prediction
📊 Seaborn, Matplotlib,Plotly	Interactive visualization
🎈 Streamlit	Interactive dashboard
🤖 Groq	Automated business insights
💾 Joblib	Model serialization
📝 GitHub	Version control & documentation

🔄 Project Workflow
Raw Customer Dataset
        ↓
Python / Pandas
        ↓
EDA & Data Preparation
        ↓
PostgreSQL
        ↓
SQL Business Analysis
        ↓
Statistical Analysis
        ↓
Machine Learning
        ↓
Random Forest Churn Model
        ↓
Automated AI Insights
        ↓
Streamlit Dashboard
        ↓
Live Prediction + Recommendations

🔎 Key Analysis Areas

Churn by geography, age, gender, activity status, and products
Compound risk-factor analysis
Statistical significance testing
Credit score, balance, and tenure analysis
Customer segmentation
Churn-driver identification
Predictive modeling
Feature importance
Automated business recommendations

🤖 Machine Learning

Best Model: Random Forest

Metric	Score
Accuracy	87.05%
Precision	80.08%
Recall	48.40%
F1 Score	0.6034
Top Churn Driver

Age — 22.4% feature importance

Model artifacts:

models/
├── random_forest_churn_model.pkl
└── scaler.pkl

📊 Streamlit Dashboard
Live Application

https://churnctrl.streamlit.app/

Modules
🏠 Home
📊 EDA Dashboard
📐 Statistical Insights
🤖 Model Performance
🎯 Live Churn Predictor
💡 Business Recommendations

Features
Interactive churn analysis
Customer segmentation
Statistical insights
Model performance metrics
Feature importance
Live churn prediction
Automated business recommendations

💡 Business Insights & Recommendations

The analysis is translated into retention-focused business actions.

🎯 Risk-Based Retention

Identify customer segments showing elevated churn risk and prioritize retention efforts accordingly.

👤 Age-Based Retention Analysis

Age accounts for 22.4% of model feature importance, making age an important factor for further churn segmentation and retention analysis.

🏦 Product & Activity Strategy

Investigate customers based on number of products and activity status to identify engagement patterns associated with churn.

🌍 Geographic Segmentation

Use geography-level churn patterns to identify locations requiring deeper investigation and potentially targeted retention strategies.

📁 Project Structure
Banking Churn analysis and prediction/
│
├── Churn_Modelling.csv
├── customers_bank_export.csv
├── export_to_csv.py
│
├── models/
│   ├── random_forest_churn_model.pkl
│   └── scaler.pkl
│
├── dashboard/
│   ├── .streamlit/
│   ├── pages/
│   │   ├── 2_EDA_Dashboard.py
│   │   ├── 3_Model_Performance.py
│   │   └── 4_Live_Churn_Predictor.py
│   └── Home.py
│
├── notebooks/
├── outputs/
├── sql/
├── src/
├── requirements.txt
└── README.md

▶️ How to Run
1. Clone Repository
git clone https://github.com/pranavvv44/ChurnCtrl.git
cd ChurnCtrl
2. Install Dependencies
pip install -r requirements.txt
3. Run Dashboard
streamlit run dashboard/Home.py
📦 Project Deliverables
🐍 Python/Pandas analysis
🗄️ SQL/PostgreSQL analysis
📐 Statistical analysis
🤖 Random Forest churn model
💾 Trained model + scaler
📊 Interactive Streamlit dashboard
🎯 Live churn prediction
🤖 Automated AI insights
💡 Business recommendations

🚀 Key Skills Demonstrated

Data Analysis • EDA • Python • Pandas • SQL • PostgreSQL • Statistical Analysis • Machine Learning • Random Forest • Predictive Analytics • Feature Importance • Streamlit • Plotly • Customer Segmentation • Churn Analysis • Business Intelligence • AI Integration • Business Insights • Data Storytelling

👤 Author

Pranav Sharma

Data and Analytics Engineer | AI/ML Automated Analysis

⭐ Explore the live dashboard, predictive model, SQL analysis, and business insights to see the complete workflow.