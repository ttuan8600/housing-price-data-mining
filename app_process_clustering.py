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


st.title("Kmeans clustering process")

st.info("""
    **KMeans – Phân cụm theo số lượng cụ thể**
    - Phân chia dữ liệu thành K cụm rõ ràng.
    - Phù hợp khi muốn **dự đoán cụm cho bất động sản mới**.
    - Cần chỉ định số cụm trước và có thể kém hiệu quả nếu dữ liệu chứa nhiễu.
    """)

# ELBOW
st.subheader("Elbow Method")
data_scaled, scaler, elbow_inertia = get_elbow(df)
fig, ax = plt.subplots()
ax.plot(range(1, 11), elbow_inertia, marker='o')
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
st.pyplot(fig)
st.info("""
Trong phương pháp Elbow, bạn cần tìm điểm "khuỷu tay" (elbow point) nơi mà giá trị inertia giảm chậm lại, nghĩa là việc tăng số lượng cluster (k) không còn mang lại cải thiện đáng kể về độ chặt chẽ của các cluster.

Nhìn vào biểu đồ, inertia giảm mạnh từ k=1 đến k=3, sau đó tốc độ giảm chậm dần từ k=4 trở đi. Elbow point xuất hiện rõ ràng nhất ở k=3, vì sau điểm này, đường cong trở nên phẳng hơn.

Vì vậy, chọn k = 3.
""")
# ELBOW

# K-Means Process
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df['cluster'] = labels
cluster_name_mapping = {
    0: "Trung cấp, gần trung tâm",
    1: "Cao cấp, đắc địa",
    2: "Phổ thông, xa trung tâm, ngoại ô"
}

st.write("Số lượng bất động sản trong mỗi cụm:")
cluster_counts = df['cluster'].map(cluster_name_mapping).value_counts()
st.write(cluster_counts)
st.write("Tâm cụm (centroids) trong không gian chuẩn hóa:")
st.write(pd.DataFrame(centroids, columns=['price_total', 'price_m2', 'area', 'long', 'lat']))
# K-Means Process

# Visualization
st.subheader("Trực quan hóa các cụm KMeans")
st.write("Biểu đồ phân tán các cụm (dựa trên giá tổng và diện tích)")
colors = ['blue', 'green', 'red']
fig, ax = plt.subplots(figsize=(10, 6))

for cluster in range(3):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['price_total'], cluster_data['area'],
               c=colors[cluster], label=f'Cụm {cluster}', alpha=0.6)


centroids_original = scaler.inverse_transform(centroids)
ax.scatter(centroids_original[:, 0], centroids_original[:, 2],
           c='black', marker='x', s=200, linewidths=3, label='Tâm cụm')

ax.set_title('Phân cụm bất động sản dựa trên KMeans')
ax.set_xlabel('Giá tổng (price_total, triệu VNĐ)')
ax.set_xscale('log')
ax.set_ylabel('Diện tích (area, m²)')
ax.set_yscale('log')
ax.legend()

st.pyplot(fig)