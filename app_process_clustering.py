import sqlite3
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap  # Thêm UMAP

from data_clustering.elbow_visualize import get_elbow
from data_clustering.standardize_data import standardize_data


@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT * FROM real_estate_processed", conn)
    df = df.dropna(subset=['area', 'price_total', 'ward', 'district', 'long', 'lat']).copy()
    return df


df = load_data()
eps = 0.5
min_samples = 5

df = standardize_data(df)

# Title and Info
st.title("KMeans Clustering Process")

# ELBOW Method
st.subheader("Elbow Method")
data_scaled, scaler, elbow_inertia = get_elbow(df)
fig, ax = plt.subplots()
ax.plot(range(1, 11), elbow_inertia, marker='o')
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
st.pyplot(fig)
# END ELBOW

# Silhouette Score Method
st.subheader("Silhouette Score Method")
silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ biểu đồ Silhouette Score
fig, ax = plt.subplots()
ax.plot(range_n_clusters, silhouette_scores, marker='o')
ax.set_title('Silhouette Score Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Silhouette Score')
st.pyplot(fig)

# In số K tối ưu dựa trên Silhouette Score
optimal_k_silhouette = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
st.write(f"Optimal number of clusters based on Silhouette Score: {optimal_k_silhouette}")
# END Silhouette Score

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df['cluster'] = labels

cluster_name_mapping = {
    0: "Mid-range, near center",
    1: "High-end, prime location",
    2: "Affordable, suburban",
    3: "Luxury, large area",
    4: "Budget, remote"
}

st.write("Number of real estate listings in each cluster:")
cluster_counts = df['cluster'].map(cluster_name_mapping).value_counts()
st.write(cluster_counts)

st.write("Cluster centroids (in standardized feature space):")
st.write(pd.DataFrame(centroids, columns=['price_total', 'price_m2', 'area', 'long', 'lat']))
# END KMeans Clustering

# Visualization with UMAP
st.subheader("KMeans Cluster Visualization (UMAP)")
st.write("Scatter plot of clusters in 2D UMAP space")

# Giảm chiều với UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
data_umap = umap_reducer.fit_transform(data_scaled)

# Giảm chiều tâm cụm
centroids_umap = umap_reducer.transform(centroids)

colors = ['blue', 'green', 'red', 'purple', 'orange']
fig, ax = plt.subplots(figsize=(10, 6))

for cluster in range(5):
    cluster_indices = df['cluster'] == cluster
    ax.scatter(data_umap[cluster_indices, 0], data_umap[cluster_indices, 1],
               c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

# Vẽ tâm cụm
ax.scatter(centroids_umap[:, 0], centroids_umap[:, 1],
           c='black', marker='x', s=200, linewidths=3, label='Centroid')

ax.set_title('Real Estate Clustering with KMeans (UMAP)')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.legend()

st.pyplot(fig)