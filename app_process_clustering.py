import sqlite3
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

# st.info("""
#     **KMeans – Clustering with a specific number of groups**
#     - Divides the dataset into K distinct clusters.
#     - Suitable when you want to **predict the cluster of a new real estate listing**.
#     - Requires predefined number of clusters and can be less effective if the data has noise.
#     """)

# ELBOW Method
st.subheader("Elbow Method")
data_scaled, scaler, elbow_inertia = get_elbow(df)
fig, ax = plt.subplots()
ax.plot(range(1, 11), elbow_inertia, marker='o')
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
st.pyplot(fig)
# st.info("""
# In the Elbow method, you look for the "elbow point" where the inertia starts to decrease more slowly. This indicates that increasing the number of clusters no longer significantly improves compactness.

# From the chart, the inertia decreases sharply from k=1 to k=3, then the rate slows down from k=4 onward. The elbow point is most noticeable at k=3, where the curve begins to flatten.

# Therefore, we choose k = 3.
# """)
# END ELBOW

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df['cluster'] = labels

cluster_name_mapping = {
    0: "Mid-range, near center",
    1: "High-end, prime location",
    2: "Affordable, suburban"
}

st.write("Number of real estate listings in each cluster:")
cluster_counts = df['cluster'].map(cluster_name_mapping).value_counts()
st.write(cluster_counts)

st.write("Cluster centroids (in standardized feature space):")
st.write(pd.DataFrame(centroids, columns=['price_total', 'price_m2', 'area', 'long', 'lat']))
# END KMeans Clustering

# Visualization
st.subheader("KMeans Cluster Visualization")
st.write("Scatter plot of clusters (based on total price and area)")

colors = ['blue', 'green', 'red']
fig, ax = plt.subplots(figsize=(10, 6))

for cluster in range(3):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['price_total'], cluster_data['area'],
               c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

centroids_original = scaler.inverse_transform(centroids)
ax.scatter(centroids_original[:, 0], centroids_original[:, 2],
           c='black', marker='x', s=200, linewidths=3, label='Centroid')

ax.set_title('Real Estate Clustering with KMeans')
ax.set_xlabel('Total Price (price_total, million VND)')
ax.set_xscale('log')
ax.set_ylabel('Area (m²)')
ax.set_yscale('log')
ax.legend()

st.pyplot(fig)
