Project

Overview
This Uber Eats Delivery Analysis App is a Streamlit-based dashboard designed to analyze delivery patterns, 
identify factors affecting delivery times, and apply machine learning techniques for
clustering and association rule mining.

Features

Data Preprocessing: Cleans and transforms the dataset for analysis.
Exploratory Data Analysis (EDA): Provides visual insights into delivery times, traffic conditions, and efficiency.
Clustering Analysis: Uses K-Means clustering to segment deliveries based on various attributes.
Association Rule Mining: Finds relationships between factors such as weather, traffic, and delivery conditions.
Anomaly Detection: Identifies outliers in delivery times.
Interactive UI: Allows users to upload datasets, explore patterns, and analyze results in real-time.

 Folder Structure

📂 UberEats-Delivery-Analysis
│-- 📄 uber_eats_delivery_analysis_app.py  # Main Streamlit app
│-- 📄 requirements.txt  # List of required dependencies
│-- 📂 data
│   └── 📄 uber-eats-deliveries.csv  # Sample dataset
│-- 📄 README.md  # Project documentation

 Installation and Setup

1. Clone the Repository
https://github.com/Darun47/uber-eats-delivery-analysis.git

2️. Install Dependencies
pip install -r requirements.txt

3️. Run the Streamlit App
streamlit run uber_eats_delivery_analysis_app.py

Steps for running this app 

Upload the dataset (.csv format) using the sidebar.
Navigate through different analysis sections:
Data Overview: View raw data and summary statistics.
Visualizations: Explore delivery time distributions and traffic impact.
Clustering Analysis: View grouped delivery patterns.
Association Rules: Identify relationships between features.
Anomaly Detection: Detect unusual delivery times.
Interact with the dashboard to explore insights dynamically.

Technologies Used During making the App

Python
Streamlit (for UI & interactivity)
Pandas (data processing)
Seaborn & Matplotlib (visualization)
Scikit-learn (clustering)
MLxtend (association rule mining)
Scipy (anomaly detection)
