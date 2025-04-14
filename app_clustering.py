import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from preprocessing import encode_ward, scale_features
from model import train_kmeans, train_dbscan
from streamlit_folium import st_folium
import folium
from folium.features import CustomIcon

# Load data từ SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT * FROM real_estate_processed", conn)
    df = df.dropna(subset=['area', 'price_total', 'ward', 'district', 'lat', 'long']).copy()
    return df

logo_url = 'https://www.pngplay.com/wp-content/uploads/7/Home-Logo-Background-PNG-Image.png'

st.title("🏡 Phân tích giá nhà đất - Phân cụm")
df = load_data()

# Chọn thuật toán phân cụm
algo = st.selectbox("Chọn thuật toán phân cụm:", ["KMeans", "DBSCAN"])

# Giải thích thuật toán đã chọn
if algo == "KMeans":
    st.info("""
    **KMeans – Phân cụm theo số lượng cụ thể**
    - Phân chia dữ liệu thành K cụm rõ ràng.
    - Phù hợp khi muốn **dự đoán cụm cho bất động sản mới**.
    - Cần chỉ định số cụm trước và có thể kém hiệu quả nếu dữ liệu chứa nhiễu.
    """)
else:
    st.info("""
    **DBSCAN – Phân cụm theo mật độ**
    - Tự động xác định số cụm, tốt khi dữ liệu có hình dạng phức tạp hoặc nhiễu.
    - Những điểm không thuộc cụm nào sẽ có **giá trị cluster là -1**.
    - **Không hỗ trợ dự đoán** cho điểm mới như KMeans.
    """)

# Cấu hình cho DBSCAN nếu được chọn
if algo == "DBSCAN":
    eps = st.slider("Chọn eps (ε):", 0.1, 2.0, step=0.1, value=0.5)
    min_samples = st.slider("Chọn min_samples:", 2, 10, step=1, value=5)
else:
    eps = 0.5
    min_samples = 5

@st.cache_resource
def process_and_train(df, algo, eps, min_samples):
    df, ward_dtype = encode_ward(df)
    scaler, X_scaled = scale_features(df)

    if algo == "KMeans":
        model, clusters, algo_name = train_kmeans(X_scaled)
    else:
        model, clusters, algo_name = train_dbscan(X_scaled, eps=eps, min_samples=min_samples)

    df['cluster'] = clusters
    return model, scaler, df, algo_name, ward_dtype

# Load data again to ensure latest
df = load_data()

# Kiểm tra trạng thái trong session_state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.similar_properties = None

# Phần hiển thị kết quả phân cụm
with st.spinner("Đang xử lý dữ liệu..."):
    model, scaler, df_clustered, algo_name, ward_dtype = process_and_train(df, algo, eps, min_samples)

# Xóa giá trị dự đoán nếu thuật toán là DBSCAN
if algo == "DBSCAN":
    st.session_state.prediction = None
    st.session_state.similar_properties = None

# Kiểm tra nếu chọn KMeans để thực hiện dự đoán
if algo_name == "kmeans":
    st.subheader("Nhập thông tin nhà cần dự đoán")
    area = st.number_input("Diện tích (m²):", min_value=10.0, max_value=2000.0, value=100.0)
    price_m2 = st.number_input("Giá mỗi m² (ước lượng):", min_value=1000000.0, value=20000000.0)

    selected_district = st.selectbox("Chọn quận/huyện:", sorted(df['district'].unique()))
    wards_in_district = df[df['district'] == selected_district]['ward'].unique()
    ward_input = st.selectbox("Chọn phường/xã:", sorted(wards_in_district))

    ward_subset = df[(df['district'] == selected_district) & (df['ward'] == ward_input)]

    if not ward_subset.empty:
        # Encode phường đã chọn
        ward_input_df = pd.DataFrame({'ward': [ward_input]})
        ward_input_df['ward'] = ward_input_df['ward'].astype(ward_dtype)
        ward_encoded = ward_input_df['ward'].cat.codes.values[0]

        if st.button("Dự đoán giá"):
            new_data = scaler.transform([[area, price_m2, ward_encoded]])  # chuẩn hóa dữ liệu mới
            cluster = int(model.predict(new_data)[0])  # Dự đoán cụm

            similar = df_clustered[df_clustered['cluster'] == cluster]
            avg_price = similar['price_total'].mean()

            # Lưu kết quả vào session_state để không bị mất khi cập nhật giao diện
            st.session_state.prediction = f"✅ Nhà này thuộc cụm số {cluster}. Giá trung bình cụm này là: {avg_price:,.0f} VND"
            st.session_state.similar_properties = similar[['name', 'area', 'price_total', 'district', 'ward']].head(10)

    else:
        st.error("❌ Không tìm thấy thông tin phường đã chọn. Vui lòng kiểm tra lại.")
else:
    # Nếu thuật toán không phải là KMeans (DBSCAN), hiển thị cảnh báo hoặc thông báo phù hợp
    st.warning("⚠️ DBSCAN không hỗ trợ dự đoán cụm cho điểm mới.")


# Hiển thị kết quả dự đoán nếu có
if st.session_state.prediction:
    st.success(st.session_state.prediction)
    st.write("### Một số bất động sản tương tự:")
    st.dataframe(st.session_state.similar_properties)

# Biểu đồ giá theo diện tích (màu theo cụm) - Cho phép chọn cụm
st.write("### Biểu đồ giá theo diện tích (màu theo cụm)")

# Tạo danh sách các cụm duy nhất để người dùng chọn
clusters_list = sorted(df_clustered['cluster'].unique())

# Cho phép người dùng chọn cụm
selected_cluster = st.selectbox("Chọn cụm để hiển thị:", clusters_list)

# Lọc các bất động sản theo cụm đã chọn
df_filtered = df_clustered[df_clustered['cluster'] == selected_cluster]

# Lấy danh sách cluster duy nhất
clusters_list = sorted(df_clustered['cluster'].unique())

# Gán màu thủ công (giả sử có tối đa 10 cụm + noise)
domain = clusters_list
range_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff0000']  # màu đỏ cuối cùng cho -1

# Nếu cluster -1 tồn tại, đảm bảo nó nằm ở cuối danh sách
if -1 in clusters_list:
    domain = [c for c in clusters_list if c != -1] + [-1]

# Vẽ biểu đồ
chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
    x='area',
    y='price_total',
    color=alt.Color('cluster:N', scale=alt.Scale(domain=domain, range=range_colors)),
    tooltip=['name', 'area', 'price_total', 'district', 'ward', 'cluster']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Tạo bản đồ với Folium
map_center = [df['long'].mean(), df['lat'].mean()]  # Vị trí trung tâm của bản đồ
m = folium.Map(location=map_center, zoom_start=12)

# Thêm các bất động sản vào bản đồ
for index, row in df.iterrows():
    # Tạo custom icon với logo
    custom_icon = CustomIcon(
        icon_image=logo_url,
        icon_size=(20, 20)  # Kích thước của logo trên bản đồ
    )

    # Thêm điểm vào bản đồ với custom icon
    folium.Marker(
        location=[row['long'], row['lat']],
        popup=f"{row['name']}<br>Diện tích: {row['area']} m²<br>Giá: {row['price_total']} VND",
        tooltip=row['name'],
        icon=custom_icon  # Sử dụng custom_icon cho Marker
    ).add_to(m)

# Hiển thị bản đồ trong Streamlit
st_folium(m, width=700, height=500)
