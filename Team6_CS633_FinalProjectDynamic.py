# =====================================================
# app.py — Beijing PM2.5 Deep Data Mining Dashboard
# =====================================================

# ----------------------
# Core Libraries
# ----------------------
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# ML & Data Mining Libraries
# ----------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================================
# Load & Clean Data
# =====================================================
@st.cache_data
def load_clean_data(path):
    df = pd.read_csv(path)
    
    # Fill missing numeric values with median
    for col in ['pm2.5', 'TEMP', 'DEWP', 'PRES']:
        df[col] = df[col].fillna(df[col].median())
    
    # Combine year, month, day, hour to datetime
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df = df.sort_values('datetime')
    
    # Feature engineering
    df['season'] = df['month'] % 12 // 3 + 1      # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
    df['high_risk'] = (df['pm2.5'] > 150).astype(int)  # Label for high PM2.5
    
    return df

# Load dataset
df = load_clean_data("/Users/naveenrajkaliarajan/Downloads/Semester_3/Data_Mining/FinalProject/PRSA_data_2010.1.1-2014.12.31.csv")  

# =====================================================
#  Streamlit Layout & Sidebar Filters
# =====================================================
st.set_page_config(page_title="Beijing PM2.5 Dashboard", layout="wide")
st.title(" Beijing PM2.5 Deep Data Mining Dashboard")

# Sidebar filters
st.sidebar.header(" Filter Options")
selected_year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()))
selected_month = st.sidebar.slider("Select Month", 1, 12, 1)
selected_cbwd = st.sidebar.multiselect(
    "Select Wind Direction", 
    df['cbwd'].unique(), 
    default=list(df['cbwd'].unique())
)

# Role Selection
st.sidebar.header("👤 Select Role")
role = st.sidebar.radio("Who are you?", ["Analyst", "Common User"])

# ----------------------
# Analyst Login
# ----------------------
if role == "Analyst":
    st.sidebar.subheader("🔒 Analyst Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if username != "analyst" or password != "password123":
        st.warning("Enter correct username and password to access analyst panel.")
        st.stop()
    else:
        st.success(f"Welcome, {username}! You now have full access.")

# Filter dataset based on selections
filtered = df[
    (df['year'] == selected_year) &
    (df['month'] == selected_month) &
    (df['cbwd'].isin(selected_cbwd))
]

st.subheader(f"📅 Filtered Data — {selected_year}-{selected_month:02d}")
st.dataframe(filtered.head(10))

# =====================================================
# Classification (Analyst Only)
# =====================================================
if role == "Analyst":
    st.header(" Classification: Predict High-Risk PM2.5")

    X = filtered[['TEMP','DEWP','PRES','Iws','is_weekend','season']]
    X = pd.get_dummies(X, columns=['season','is_weekend'], drop_first=True)
    y = filtered['high_risk']

    if len(filtered) > 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # Feature Importance
        importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(importance)

# =====================================================
# Clustering: PM2.5 Pattern Groups
# =====================================================
st.header(" Clustering: PM2.5 Pattern Groups")

cluster_features = filtered[['pm2.5','TEMP','DEWP','PRES','Iws']].fillna(0)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Determine optimal number of clusters
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled_features)
    sil_scores.append(silhouette_score(scaled_features, labels))

best_k = sil_scores.index(max(sil_scores)) + 2

# Fit final KMeans
kmeans = KMeans(n_clusters=best_k, random_state=42)
filtered['cluster'] = kmeans.fit_predict(scaled_features)

#  High-contrast color palette
cluster_colors = ["#FF0000", "#008000", "#0000FF", "#FFA500", "#800080", "#00CED1"]

fig = px.scatter(
    filtered,
    x="TEMP",
    y="pm2.5",
    color="cluster",
    hover_data=['datetime', 'cbwd'],
    color_discrete_sequence=cluster_colors[:best_k],
    title=f"PM2.5 Clusters (Optimal k = {best_k})"
)
st.plotly_chart(fig, use_container_width=True)

# Cluster summary
cluster_summary = filtered.groupby("cluster")[['pm2.5', 'TEMP']].mean().reset_index()
st.dataframe(cluster_summary)

cluster_summary_full = filtered.groupby('cluster')[['pm2.5', 'TEMP', 'DEWP']].mean().round(2)
print(cluster_summary_full)

# -----------------------------------------------------
#  Clean Day Formatting for Users
# -----------------------------------------------------
cluster_days_map = (
    filtered.assign(date_hour=filtered['datetime'].dt.strftime("%b %d %H:00"))
    .groupby('cluster')['date_hour']
    .apply(lambda dh: sorted(set(dh)))
)
#cluster_days_map = (
#    filtered.assign(date_only=filtered['datetime'].dt.date)
#    .groupby('cluster')['date_only']
#    .apply(lambda dates: sorted(set(dates)))
#    .apply(lambda dates: [d.strftime("%b %d") for d in dates])
#)

# -----------------------------------------------------
# Common User Cluster Messages
# -----------------------------------------------------
if role == "Common User":
    st.subheader(" Cluster-Based Air Quality Insights")

    for i, row in cluster_summary.iterrows():
        pm = row['pm2.5']
        days_list = ", ".join(cluster_days_map[i])

        if pm > 250:
            msg = f"☠️ Hazardous pollution ({pm:.1f}). Avoid outdoor activity."
            st.error(f"Cluster {i}: {msg} Days: {days_list}")
        elif pm > 150:
            msg = f"🔴 Very High pollution ({pm:.1f}). Wear N95 masks outside."
            st.error(f"Cluster {i}: {msg} Days: {days_list}")
        elif pm > 80:
            msg = f"🟡 Moderate pollution ({pm:.1f}). Limit prolonged outdoor exposure."
            st.warning(f"Cluster {i}: {msg} Days: {days_list}")
        elif pm > 40:
            msg = f"🟢 Clean air ({pm:.1f}). Safe for outdoor activities."
            st.success(f"Cluster {i}: {msg} Days: {days_list}")
        else:
            msg = f"🌤 Very low pollution ({pm:.1f}). Excellent air quality."
            st.info(f"Cluster {i}: {msg} Days: {days_list}")

# =====================================================
# Association Rules (Analyst Only)
# =====================================================
if role == "Analyst":
    st.header(" Association Rule Mining: Discover Hidden PM2.5 Patterns")

    # -------------------------------
    # Prepare data for ARM
    # -------------------------------
    df_rules = filtered[['pm2.5','TEMP','DEWP','cbwd']].copy()
    df_rules['high_pm'] = (df_rules['pm2.5'] > 150).astype(int)
    df_rules['low_temp'] = (df_rules['TEMP'] < df_rules['TEMP'].median()).astype(int)

    # One-hot encode (True/False matrix)
    df_trans = pd.get_dummies(df_rules[['high_pm','low_temp','cbwd']]).astype(bool)

    # -------------------------------
    # Apply Apriori
    # -------------------------------
    rules = apriori(df_trans, min_support=0.05, use_colnames=True)
    rules = association_rules(rules, metric="lift", min_threshold=1.2)

    # -------------------------------
    # Convert antecedents & consequents to strings
    # -------------------------------
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    # Display top rules
    st.subheader("Top 10 Rules by Lift")
    rules_display = rules[['antecedents_str','consequents_str','support','confidence','lift']]
    st.dataframe(rules_display.sort_values(by='lift', ascending=False).head(10))

    # =================================================
    #  Visualization 1 — Lift Bar Chart
    # =================================================
    st.subheader(" Lift of Rules (Top 10)")
    top10 = rules.sort_values(by="lift", ascending=False).head(10)

    fig_lift = px.bar(
        top10,
        x="lift",
        y="antecedents_str",
        orientation="h",
        color="lift",
        text="consequents_str",
        title="Top Association Rules by Lift",
        color_continuous_scale="Plasma"
    )
    fig_lift.update_layout(yaxis_title="Antecedent", xaxis_title="Lift")
    st.plotly_chart(fig_lift, use_container_width=True)

    # =================================================
    #  Visualization 2 — Support vs Confidence Bubble Chart
    # =================================================
    st.subheader(" Support vs Confidence (Bubble = Lift)")

    fig_scatter = px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_data=["antecedents_str", "consequents_str"],
        title="Support vs Confidence (Bubble size represents Lift)",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)



# =====================================================
# Role-Specific Business Insights
# =====================================================
st.header(" Business Insights for Decision Makers")

# Role-based strategy dictionary
role_actions = {
    "hazardous": {
        "Common Users": "Stay indoors, use air purifiers, avoid exercise.",
        "Health Dept": "Issue emergency alerts, increase hospital readiness.",
        "Traffic Dept": "Shut down high-emission zones, restrict vehicles.",
        "Environmental Agencies": "Enforce industrial shutdowns, monitor factories.",
        "City Planners": "Activate emergency plan, deploy public warnings."
    },
    "high": {
        "Common Users": "Wear N95 masks outdoors.",
        "Health Dept": "Alert sensitive groups (kids, elderly).",
        "Traffic Dept": "Reduce peak-hour congestion.",
        "Environmental Agencies": "Increase emission inspections.",
        "City Planners": "Promote public transit."
    },
    "medium": {
        "Common Users": "Limit outdoor activity if sensitive.",
        "Health Dept": "Monitor respiratory cases.",
        "Traffic Dept": "Encourage carpooling.",
        "Environmental Agencies": "Track pollution trends.",
        "City Planners": "Plan awareness campaigns."
    },
    "clean": {
        "Common Users": "Outdoor activities are safe.",
        "Health Dept": "Normal operations.",
        "Traffic Dept": "No special measures needed.",
        "Environmental Agencies": "Routine monitoring.",
        "City Planners": "Promote parks & events."
    },
    "very_low": {
        "Common Users": "Excellent weather!",
        "Health Dept": "Lowest risk level.",
        "Traffic Dept": "Normal traffic control.",
        "Environmental Agencies": "Air quality ideal.",
        "City Planners": "Encourage outdoor tourism."
    }
}

# Pollution severity detector
def get_level(pm):
    if pm > 250: return "hazardous"
    if pm > 150: return "high"
    if pm > 80: return "medium"
    if pm > 40: return "clean"
    return "very_low"

# Display insights per cluster
for i, row in cluster_summary.iterrows():
    pm = row["pm2.5"]
    days_list = ", ".join(cluster_days_map[i])
    level = get_level(pm)

    st.markdown(f"### 🔷 Cluster {i} — Avg PM2.5: {pm:.1f} ({level.upper()})")
    st.markdown(f"**Days:** {days_list}")

    st.write("#### 👥 Recommended Actions by Role:")
    for role_name, action_text in role_actions[level].items():
        st.markdown(f"- **{role_name}:** {action_text}")

    st.markdown("---")