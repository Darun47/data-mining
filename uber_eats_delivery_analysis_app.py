import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("uber-eats-deliveries.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Data Preprocessing
def preprocess_data(df):
    df = df.dropna()
    
    # Ensure all expected columns exist
    expected_cols = ["Weather_conditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle"]
    available_cols = [col for col in expected_cols if col in df.columns]
    
    # Encode only available categorical columns
    for col in available_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Normalize numerical columns safely
    numeric_cols = ['Time_taken(min)', 'Order_Size', 'Vehicle_condition']
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    scaler = MinMaxScaler()
    df[available_numeric_cols] = scaler.fit_transform(df[available_numeric_cols])

    return df

# Clustering using K-Means
def perform_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Time_taken(min)', 'Order_Size', 'Vehicle_condition']])
    return df

# Association Rule Mining
def association_analysis(df):
    try:
        basket = df[['Weather_conditions', 'Road_traffic_density', 'Type_of_order']]
        basket = basket.applymap(lambda x: True if x > 0 else False)
        frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
        assoc_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        return assoc_rules
    except Exception as e:
        st.error(f"Error in association analysis: {e}")
        return pd.DataFrame()

# Anomaly Detection
def detect_anomalies(df):
    try:
        z_scores = np.abs(stats.zscore(df['Time_taken(min)']))
        anomalies = df[z_scores > 3]
        return anomalies
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")
        return pd.DataFrame()

# Streamlit UI
def main():
    st.title("Uber Eats Delivery Analysis App")
    st.sidebar.header("Options")
    
    df = load_data()
    
    if df is not None:
        df = preprocess_data(df)
        df = perform_clustering(df)
        assoc_rules = association_analysis(df)
        anomalies = detect_anomalies(df)

        option = st.sidebar.selectbox("Choose Analysis", 
                                      ["Data Overview", "Clustering Analysis", "Association Rules", "Anomaly Detection"])

        if option == "Data Overview":
            st.subheader("Sample Data")
            st.dataframe(df.head())

            st.subheader("Delivery Time Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Time_taken(min)'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

        elif option == "Clustering Analysis":
            st.subheader("Clustering Results")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Order_Size'], y=df['Time_taken(min)'], hue=df['Cluster'], palette='viridis', ax=ax)
            st.pyplot(fig)

        elif option == "Association Rules":
            st.subheader("Association Rules Analysis")
            st.dataframe(assoc_rules)

        elif option == "Anomaly Detection":
            st.subheader("Anomalies in Delivery Times")
            st.dataframe(anomalies)

if __name__ == "__main__":
    main()
