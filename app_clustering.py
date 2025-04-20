import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from matplotlib import pyplot as plt

from data_clustering.elbow_visualize import get_elbow, check_tables
from data_clustering.standardize_data import standardize_data
from preprocessing import encode_ward, scale_features
from model import train_kmeans, train_dbscan
from streamlit_folium import st_folium
import folium
from folium.features import CustomIcon
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder
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

eps = 0.5
min_samples = 5

df = standardize_data(df)

# ELBOW
data_scaled, scaler, elbow_inertia = get_elbow(df)
# ELBOW

# K-Means Process
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df['cluster'] = labels
# K-Means Process

# Hiển thị thông tin cụm

cluster_name_mapping = {
    0: "Trung cấp, gần trung tâm",
    1: "Cao cấp, đắc địa",
    2: "Phổ thông, xa trung tâm, ngoại ô"
}

st.write("Số lượng bất động sản trong mỗi cụm:")
cluster_counts = df['cluster'].map(cluster_name_mapping).value_counts()
st.write(cluster_counts)

# Dropdown để chọn cụm
st.subheader("Tra cứu dữ liệu theo cụm")
# Tạo danh sách tùy chọn cho dropdown, bao gồm 'All' và các tên cụm
cluster_options = ['All'] + list(cluster_name_mapping.values())
selected_cluster_display = st.selectbox("Chọn cụm", options=cluster_options, index=0)

# Lọc dữ liệu theo cụm được chọn
if selected_cluster_display == 'All':
    cluster_df = df  # Hiển thị toàn bộ dữ liệu
else:
    # Chuyển tên cụm trở lại giá trị số để lọc dữ liệu
    selected_cluster = [k for k, v in cluster_name_mapping.items() if v == selected_cluster_display][0]
    cluster_df = df[df['cluster'] == selected_cluster]  # Lọc theo cụm
# Định dạng dữ liệu
cluster_df_display = cluster_df.copy()
cluster_df_display['price_m2'] = cluster_df_display['price_m2'].apply(lambda x: f"{x:,.2f} triệu VNĐ/m²")
cluster_df_display['price_total'] = cluster_df_display['price_total'].apply(lambda x: f"{x:,.0f} triệu VNĐ")
cluster_df_display['cluster'] = cluster_df_display['cluster'].map(cluster_name_mapping)
cluster_df_display = cluster_df_display.reset_index()
columns_to_display = ['id', 'name', 'district', 'price_total', 'price_m2']
cluster_df_display = cluster_df_display[columns_to_display]

# Cấu hình AgGrid
gb = GridOptionsBuilder.from_dataframe(cluster_df_display)
gb.configure_column('id', headerName='ID', width=80)
gb.configure_column('name', headerName='Tên')
gb.configure_column('district', headerName='Quận')
gb.configure_column('price_total', headerName='Giá')
gb.configure_column('price_m2', headerName='Giá/m²')
gb.configure_selection('single', use_checkbox=False)  # Cho phép chọn 1 hàng
grid_options = gb.build()

# Hiển thị bảng với AgGrid
st.subheader(f"Dữ liệu của {'Tất cả các cụm' if selected_cluster_display == 'All' else selected_cluster_display}:")
grid_response = AgGrid(cluster_df_display, gridOptions=grid_options, height=400, allow_unsafe_jscode=True)


# Nhập ID từ người dùng
search_text = st.text_input("🔍 Nhập từ khóa để tìm (tên BĐS, đường, phường, quận)", "")

# Tạo placeholder cho bản đồ
map_placeholder = st.empty()
# Visualization

# Folium Map
map_center = [df['lat'].mean(), df['long'].mean()]
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
            f"<div><b>Cụm:</b> {cluster_name_mapping[row['cluster']]}</div>"
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
        f"<div><b>Cụm:</b> {cluster_name_mapping[row['cluster']]}</div>"
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

if search_text:
    mask = (
            df['name'].str.contains(search_text, case=False, na=False) |
            df['street'].str.contains(search_text, case=False, na=False) |
            df['ward'].str.contains(search_text, case=False, na=False) |
            df['district'].str.contains(search_text, case=False, na=False)
    )
    filtered_df = df[mask]

    if not filtered_df.empty:
        for idx, row in filtered_df.iterrows():
            popup_content = (
                f"{popup_style}"
                f"<div><b>Tên:</b> {row['name']}</div>"
                f"<div><b>Diện tích:</b> {row['area']:,.0f} m2</div>"
                f"<div><b>Giá/m2:</b> {row['price_m2']:,.0f} VNĐ</div>"
                f"<div><b>Giá:</b> {row['price_total']:,.0f} VNĐ</div>"
                f"<div><b>Quận:</b> {row['district']}</div>"
                f"<div><b>Địa chỉ:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
                f"<div><b>Cụm:</b> {cluster_name_mapping[row['cluster']]}</div>"
            )
            folium.Marker(
                location=[row['lat'], row['long']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='search')
            ).add_to(m)

        # Zoom tới vị trí kết quả đầu tiên
        first_row = filtered_df.iloc[0]
        m.location = [first_row['lat'], first_row['long']]
        m.zoom_start = 16

# Thêm LayerControl
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=700, height=500)

# Folium Map
