This project is an interactive Streamlit dashboard designed to analyze the Beijing PM2.5 pollution dataset using Machine Learning, Clustering, Classification, and Association Rule Mining techniques.

The goal is to:

Study air pollution trends (2010–2014)

Identify high-risk air quality periods

Predict pollution levels

Group pollution patterns using clustering

Discover hidden relationships (wind, temperature, pollution spikes)

Provide actionable insights for analysts, public users, and city planners

This application supports two user modes:

Common User → simple air-quality advice

Analyst → full ML models, ARM, feature importance, deep insights

Key Features
✅ Interactive Dashboard

Filter by year, month, and wind direction

See real-time filtered datasets

🔍 Machine Learning

Random Forest Classification to predict high-risk PM2.5 days

Classification report + Confusion Matrix

Feature importance visualization

🌐 Clustering (K-Means)

Automatically determines optimal cluster count (silhouette score)

Visualizes PM2.5 clusters & pattern groups

Color-coded clusters for easy interpretation

Daily cluster breakdown for common users

🔗 Association Rule Mining (Apriori)

Discovers strong correlations between:

high PM2.5

wind direction

temperature

Visualized with Lift bar chart & Support–Confidence bubble chart

🧭 Business Insights

Role-based actions for:

Common Users

Health Departments

Traffic Authorities

Environmental Agencies

City Planners

How to Add the Dataset

You must place the dataset available in the GIT manually into your project folder.

Download the dataset
Move it into your project folder

Ensure the file name is: PRSA_data_2010.1.1-2014.12.31.csv

Your folder should now look like:
FinalProject/
│
├── app.py
└── PRSA_data_2010.1.1-2014.12.31.csv


Installation Instructions
1️⃣ Clone the Repository

git clone https://github.com/Ilakkiya1498/PM2.5-Based-Air-pollution-Analysis.git
cd PM2.5-Based-Air-pollution-Analysis

2️⃣ Install Required Libraries

Recommended: Create a virtual environment
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn mlxtend

▶️ How to Run the Streamlit App

Once dependencies are installed:
streamlit run app.py

👤 User Roles
🔹 Common User

No login required

Sees simple pollution-risk messages

Gets daily clean air recommendations

Ideal for public users

🔹 Analyst

Login required:

Username: analyst
Password: password123


Unlocks:

Classification model

Confusion matrix

Feature importance

K-Means clustering results

Association Rule Mining

Deep business insights

What This Project Achieves

✔ Identifies pollution patterns across seasons and years
✔ Predicts high-risk pollution days
✔ Clusters similar pollution conditions
✔ Links environmental factors with pollution spikes
✔ Offers actionable environmental recommendations
✔ Serves both analysts & everyday users

Sample Output Visuals (Automatically Generated)

Scatter plot of clusters

Bar chart of feature importance

Confusion matrix heatmap

Lift bar chart (Association Rules)

Support vs Confidence bubble chart

Technologies Used

Python

Streamlit

Pandas, Numpy

Scikit-Learn

Plotly, Seaborn, Matplotlib

mlxtend (Apriori)

Author

Ilakkiya / Naveen Raj
Master’s Student – Data Mining
Beijing PM2.5 Data Analysis Project

