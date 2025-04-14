import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from preprocessing import encode_ward, scale_features
from model import detect_outliers_zscore, detect_outliers_iqr

# Load data từ SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT * FROM real_estate_processed", conn)
    df = df.dropna(subset=['area', 'price_total', 'ward', 'district']).copy()
    return df

st.title("🏡 Phân tích giá nhà đất - Phát hiện bất thường")
df = load_data()

# Thanh trượt chọn phạm vi giá trị tổng (VND)
min_price, max_price = st.slider("Chọn phạm vi giá trị tổng (VND)", 
                                 min_value=int(df['price_total'].min()), 
                                 max_value=int(df['price_total'].max()), 
                                 value=(int(df['price_total'].min()), int(df['price_total'].max())))

# Lọc dữ liệu theo phạm vi giá trị đã chọn
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price)]

# Phát hiện outliers với phương pháp đã chọn
method = st.selectbox("Chọn phương pháp phát hiện bất thường:", ["Z-score", "IQR"])

if method == "Z-score":
    st.info("""
    ## 📚 Giải thích về Z-score
    **Z-score** đo lường độ lệch của một điểm dữ liệu so với giá trị trung bình của toàn bộ dữ liệu, tính bằng đơn vị độ lệch chuẩn. 
    - **Cách hiểu đơn giản:** Nếu bạn tưởng tượng dữ liệu của mình là các bài kiểm tra, **Z-score** cho bạn biết điểm nào "cao" hoặc "thấp" như thế nào so với điểm trung bình của lớp học. 
    - **Ví dụ:** 
      - **Z-score = 2** có nghĩa là điểm cách trung bình 2 độ lệch chuẩn, có thể là một **outlier**.
      - **Z-score = 0** có nghĩa là điểm bằng với trung bình.
    - **Ưu điểm:** Dễ hiểu và tính toán nhanh chóng. Tốt cho dữ liệu phân phối chuẩn.
    - **Nhược điểm:** Cần dữ liệu có phân phối chuẩn.
    """)
else:
    st.info("""
    ## 📚 Giải thích về IQR
    **IQR** là khoảng cách giữa phân vị thứ ba (Q3) và phân vị thứ nhất (Q1). Nó giúp xác định những điểm nào nằm ngoài khoảng bình thường.
    - **Cách hiểu đơn giản:** Hãy tưởng tượng bạn xếp các bài kiểm tra từ thấp đến cao và chia nhóm chúng thành 4 phần. **IQR** giúp bạn hiểu được khoảng giữa 25% số điểm thấp nhất và 25% số điểm cao nhất.
    - **Ví dụ:** 
      - Nếu điểm nằm ngoài khoảng từ **Q1 - 1.5 * IQR** đến **Q3 + 1.5 * IQR**, nó có thể là **outlier**.
    - **Ưu điểm:** Không cần dữ liệu phân phối chuẩn. Tốt cho dữ liệu có sự phân tán lớn.
    - **Nhược điểm:** Không phát hiện được các outliers nếu chúng nằm trong khoảng giữa các phân vị.
    """)

# Phát hiện outliers
if method == "Z-score":
    outliers = detect_outliers_zscore(filtered_df)
else:
    outliers = detect_outliers_iqr(filtered_df)

# Gắn nhãn outlier vào DataFrame
filtered_df['outlier'] = outliers
n_outliers = filtered_df['outlier'].sum()

# Hiển thị tổng số bất động sản nghi ngờ bất thường
st.markdown(f"### 🔍 Tổng số bất động sản bị nghi ngờ bất thường: **{n_outliers}**")

if n_outliers > 0:
    display_df = filtered_df[filtered_df['outlier']][['name', 'area', 'price_total', 'district', 'ward']].head(10).copy()
    display_df['price_total'] = display_df['price_total'].apply(lambda x: f"{x:,.0f}")  # format với dấu phẩy, không có số lẻ
    st.dataframe(display_df)

# Thêm lựa chọn cho người dùng: Hiển thị chỉ outliers hay non-outliers
view_option = st.radio("Chọn loại dữ liệu muốn hiển thị trên biểu đồ:", ["Tất cả", "Chỉ Outliers", "Chỉ Non-outliers"])

# Lọc dữ liệu theo lựa chọn của người dùng
if view_option == "Chỉ Outliers":
    data_to_plot = filtered_df[filtered_df['outlier'] == 1]
elif view_option == "Chỉ Non-outliers":
    data_to_plot = filtered_df[filtered_df['outlier'] == 0]
else:
    data_to_plot = filtered_df

# Biểu đồ phân bố diện tích và giá trị tổng theo outliers
st.write("### Biểu đồ phân bố - Outliers đánh dấu màu đỏ")
chart_outlier = alt.Chart(data_to_plot).mark_circle(size=60).encode(
    x='area',
    y='price_total',
    color=alt.condition("datum.outlier", alt.value("red"), alt.value("steelblue")),
    tooltip=['name', 'area', 'price_total', 'district', 'ward']
).interactive()
st.altair_chart(chart_outlier, use_container_width=True)

# Biểu đồ Bar Plot cho giá trung bình mỗi m² theo Quận
st.write("### Biểu đồ Bar Plot - Giá trung bình mỗi m² theo Quận")
avg_price_by_district = filtered_df.groupby('district')['price_m2'].mean().reset_index()
chart_bar = alt.Chart(avg_price_by_district).mark_bar().encode(
    x='district',
    y='price_m2',
    color='district:N',
    tooltip=['district', 'price_m2']
).properties(
    title='Giá trung bình mỗi m² theo Quận'
)
st.altair_chart(chart_bar, use_container_width=True)

# Biểu đồ phân phối giá mỗi m²
st.write("### Biểu đồ phân phối giá mỗi m²")
chart_price_m2 = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('price_m2', bin=alt.Bin(maxbins=30), title='Giá mỗi m² (VND)'),
    alt.Y('count()', title='Số lượng'),
    color=alt.Color('outlier:N', legend=alt.Legend(title="Outlier"))
).properties(
    title='Phân phối giá mỗi m² của bất động sản'
)
st.altair_chart(chart_price_m2, use_container_width=True)
