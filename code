import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Load the Uber Eats delivery data
def load_data():
    df = pd.read_csv("uber-eats-deliveries.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    df.dropna(inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    scaler = MinMaxScaler()
    df[['Time_taken(min)', 'Order_Size', 'Vehicle_condition']] = scaler.fit_transform(df[['Time_taken(min)', 'Order_Size', 'Vehicle_condition']])
    return df

# Perform clustering
def perform_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Time_taken(min)', 'Order_Size', 'Vehicle_condition']])
    return df

# Perform association analysis
def association_analysis(df):
    basket = df[['Weather_conditions_Clear', 'Road_traffic_density_High', 'Time_taken(min)']]
    basket = basket.applymap(lambda x: True if x > 0 else False)
    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    assoc_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return assoc_rules

# Detect anomalies
def detect_anomalies(df):
    z_scores = np.abs(stats.zscore(df['Time_taken(min)']))
    anomalies = df[z_scores > 3]
    return anomalies

# Main function to run the Streamlit app
def main():
    st.title("Uber Eats Delivery Analysis App")
    st.sidebar.header("Options")
    
    df = load_data()
    df = preprocess_data(df)
    df = perform_clustering(df)
    assoc_rules = association_analysis(df)
    anomalies = detect_anomalies(df)
    
    option = st.sidebar.selectbox("Choose Analysis", ["Data Overview", "Clustering Analysis", "Association Rules", "Anomaly Detection"])
    
    if option == "Data Overview":
        st.write("### Sample Data")
        st.dataframe(df.head())
        st.write("### Data Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Time_taken(min)'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    
    elif option == "Clustering Analysis":
        st.write("### Clustering Results")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Order_Size'], y=df['Time_taken(min)'], hue=df['Cluster'], palette='viridis', ax=ax)
        st.pyplot(fig)
    
    elif option == "Association Rules":
        st.write("### Association Rules Analysis")
        st.dataframe(assoc_rules)
    
    elif option == "Anomaly Detection":
        st.write("### Anomalies in Delivery Times")
        st.dataframe(anomalies)
    
if __name__ == "__main__":
    main()
