import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from matplotlib import pyplot as plt

from data_clustering.elbow_visualize import get_elbow, check_tables
from preprocessing import encode_ward, scale_features
from model import train_kmeans, train_dbscan
from streamlit_folium import st_folium
import folium
from folium.features import CustomIcon
from sklearn.cluster import KMeans

# Load data từ SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT * FROM real_estate_processed", conn)
    df = df.dropna(subset=['area', 'price_total', 'ward', 'district', 'long', 'lat']).copy()
    return df

logo_url = 'https://www.pngplay.com/wp-content/uploads/7/Home-Logo-Background-PNG-Image.png'

st.title("🏡 Phân tích giá nhà đất - Phân cụm")
df = load_data()

# Chọn thuật toán phân cụm
algo = st.selectbox("Chọn thuật toán phân cụm:", ["KMeans"])

# Giải thích thuật toán đã chọn
if algo == "KMeans":
    st.info("""
    **KMeans – Phân cụm theo số lượng cụ thể**
    - Phân chia dữ liệu thành K cụm rõ ràng.
    - Phù hợp khi muốn **dự đoán cụm cho bất động sản mới**.
    - Cần chỉ định số cụm trước và có thể kém hiệu quả nếu dữ liệu chứa nhiễu.
    """)

eps = 0.5
min_samples = 5

# Load data again to ensure latest
df = load_data()

# ELBOW
st.subheader("Elbow Method")
data_scaled, scaler, elbow_inertia = get_elbow(load_data())
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
st.write("Số lượng bất động sản trong mỗi cụm:")
st.write(df['cluster'].value_counts())
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


# Vẽ tâm cụm
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
# Visualization

# Folium Map
map_center = [df['long'].mean(), df['lat'].mean()]
m = folium.Map(location=map_center, zoom_start=12)

popup_style = """
<style>
    .leaflet-popup-content {
        width: 300px !important;
        padding: 10px;
        font-family: Arial, sans-serif;
        line-height: 1.5;
    }
    .leaflet-popup-content b {
        color: #2c3e50;
    }
    .leaflet-popup-content div {
        margin-bottom: 5px;
    }
</style>
"""
custom_icon = CustomIcon(
        icon_image=logo_url,
        icon_size=(20, 20)
    )

districts = sorted(df['district'].unique())
colors = {0: 'blue', 1: 'green', 2: 'purple'}

for district in districts:
    # Tạo FeatureGroup cho quận
    fg = folium.FeatureGroup(name=district, show=False)  # Mặc định ẩn
    # Lọc dữ liệu theo quận
    district_df = df[df['district'] == district]
    # Thêm markers
    for idx, row in district_df.iterrows():
        popup_content = (
            f"{popup_style}"
            f"<div><b>Tên:</b> {row['name']}</div>"         
            f"<div><b>Diện tích:</b> {row['area']:,.0f} m2</div>"
            f"<div><b>Giá/m2:</b> {row['price_m2']:,.0f} VNĐ</div>"
            f"<div><b>Giá:</b> {row['price_total']:,.0f} VNĐ</div>"
            f"<div><b>Quận:</b> {row['district']}</div>"
            f"<div><b>Địa chỉ:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
            f"<div><b>Cụm:</b> {row['cluster']}</div>"
        )
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=6,
            color=colors[row['cluster']],
            fill=True,
            fill_color=colors[row['cluster']],
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(fg)
    fg.add_to(m)

# Thêm layer tất cả
fg_all = folium.FeatureGroup(name='Tất cả', show=True)  # Mặc định hiển thị
for idx, row in df.iterrows():
    popup_content = (
        f"{popup_style}"
        f"<div><b>Tên:</b> {row['name']}</div>"
        f"<div><b>Diện tích:</b> {row['area']:,.0f} m2</div>"
        f"<div><b>Giá/m2:</b> {row['price_m2']:,.0f} VNĐ</div>"
        f"<div><b>Giá:</b> {row['price_total']:,.0f} VNĐ</div>"
        f"<div><b>Quận:</b> {row['district']}</div>"
        f"<div><b>Địa chỉ:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
        f"<div><b>Cụm:</b> {row['cluster']}</div>"
    )
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=6,
        color=colors[row['cluster']],
        fill=True,
        fill_color=colors[row['cluster']],
        fill_opacity=0.7,
        popup=folium.Popup(popup_content, max_width=300)
    ).add_to(fg_all)
fg_all.add_to(m)

# Thêm LayerControl
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=700, height=500)

# Folium Map

# Predict with KMeans
# st.subheader("Nhập thông tin nhà cần dự đoán")
# area = st.number_input("Diện tích (m²):", min_value=10.0, max_value=2000.0, value=100.0)
# price_m2 = st.number_input("Giá mỗi m² (ước lượng):", min_value=1000000.0, value=20000000.0)
#
# selected_district = st.selectbox("Chọn quận/huyện:", sorted(df['district'].unique()))
# wards_in_district = df[df['district'] == selected_district]['ward'].unique()
# ward_input = st.selectbox("Chọn phường/xã:", sorted(wards_in_district))
#
# ward_subset = df[(df['district'] == selected_district) & (df['ward'] == ward_input)]
#
# if not ward_subset.empty:
#     # Encode phường đã chọn
#     ward_input_df = pd.DataFrame({'ward': [ward_input]})
#     ward_input_df['ward'] = ward_input_df['ward'].astype(ward_dtype)
#     ward_encoded = ward_input_df['ward'].cat.codes.values[0]
#
#     if st.button("Dự đoán giá"):
#         new_data = scaler.transform([[area, price_m2, ward_encoded]])  # chuẩn hóa dữ liệu mới
#         cluster = int(model.predict(new_data)[0])  # Dự đoán cụm
#
#         similar = df_clustered[df_clustered['cluster'] == cluster]
#         avg_price = similar['price_total'].mean()
#
#         # Lưu kết quả vào session_state để không bị mất khi cập nhật giao diện
#         st.session_state.prediction = f"✅ Nhà này thuộc cụm số {cluster}. Giá trung bình cụm này là: {avg_price:,.0f} VND"
#         st.session_state.similar_properties = similar[['name', 'area', 'price_total', 'district', 'ward']].head(10)
#
# else:
#     st.error("❌ Không tìm thấy thông tin phường đã chọn. Vui lòng kiểm tra lại.")
#
# if st.session_state.prediction:
#     st.success(st.session_state.prediction)
#     st.write("### Một số bất động sản tương tự:")
#     st.dataframe(st.session_state.similar_properties)
# Predict with KMeans
