import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Features for clustering
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans clustering
n_clusters = 5  # You can adjust this number based on your needs
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Save the clustering model
with open('models/clustering_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("Clustering model has been trained and saved successfully!")
