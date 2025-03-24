elif page == "Clustering Analysis":
    st.subheader("🌀 Clustering Analysis of Delivery Times")
    
    # Define clustering features
    features = ["Time_taken(min)", "Delivery_person_Age", "Delivery_person_Ratings"]
    
    # Ensure columns exist and are numeric
    available_features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # If not enough valid features, show error
    if len(available_features) < 2:
        st.error("❌ Not enough numerical columns for clustering. Check the dataset.")
    else:
        # Drop missing values for clustering
        X = df[available_features].dropna()

        # Ensure there are enough unique values
        for col in available_features:
            unique_values = X[col].nunique()
            if unique_values < 2:
                st.warning(f"⚠️ Not enough unique values in '{col}'. Clustering may not work well.")

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Debug: Print the shape of data
        st.write("📊 Data Shape Before Clustering:", X_scaled.shape)

        # Apply KMeans only if enough data points exist
        if X_scaled.shape[0] < 3:  # Less than 3 rows cause clustering failure
            st.error("❌ Not enough data points for clustering. Upload a larger dataset.")
        else:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df.loc[X.index, "Cluster"] = kmeans.fit_predict(X_scaled)

            # Debug: Print assigned clusters
            st.write("✅ Cluster Assignments:", df["Cluster"].value_counts())

            # Visualization
            fig, ax = plt.subplots()
            sns.scatterplot(x="Delivery_person_Age", y="Time_taken(min)", hue=df["Cluster"], palette="viridis", ax=ax)
            st.pyplot(fig)
            st.success("✅ Clustering Analysis Completed!")
