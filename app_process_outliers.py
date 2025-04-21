import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import folium
from streamlit_folium import st_folium
from data_outlier.outliers_detection import detect_zscore_outliers, detect_iqr_outliers, IQR_method, z_scoremod_method
from matplotlib import pyplot as plt
import seaborn as sns
import re
from scipy.stats import median_abs_deviation
import numpy as np
from collections import Counter
import locale
from matplotlib.ticker import FuncFormatter

# Load data từ SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query(
        "SELECT property_type, street, ward, district, price_total, price_m2, area, long, lat FROM real_estate_processed",
        conn)
    df = df.dropna(
        subset=['property_type', 'street', 'ward', 'district', 'price_total', 'price_m2', 'area', 'long', 'lat']).copy()
    return df


df = load_data()
feature_list = ['price_m2', 'area', 'price_total']
method = st.selectbox("Chọn phương pháp phát hiện bất thường:", ["Z-score", "IQR"])
st.title(f"🏡 Outliers Detection - {method} ")


# Custom formatter function for price_total
def format_price(value):
    return "{:,.0f} VND".format(value)


# Custom formatter function for area
def format_area(value):
    return "{:,.0f} m²".format(value)


# Slider for price_total
min_price, max_price = st.slider(
    "Chọn phạm vi giá trị tổng (VND)",
    min_value=int(df['price_total'].min()),
    max_value=int(df['price_total'].max()),
    value=(int(df['price_total'].min()), int(df['price_total'].max())),
    step=1000000,  # optional, to make sliding smoother for big numbers
    format='%d VND'
)
st.markdown(f"**Giá từ:** &nbsp; {min_price:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_price:,} VND")

# Slider for area
min_area, max_area = st.slider(
    "Chọn phạm vi diện tích (m²)",
    min_value=int(df['area'].min()),
    max_value=int(df['area'].max()),
    value=(int(df['area'].min()), int(df['area'].max())),
    format='%d VND'
)
st.markdown(f"**Diện tích từ:** &nbsp; {min_area:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_area:,} m²")

# Filter data based on selected price_total and area ranges
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price) &
                 (df['area'] >= min_area) & (df['area'] <= max_area)]
filtered_df = filtered_df[~filtered_df.eq("N/A").any(axis=1)]
# Formatter for thousand separators (e.g., 1,000,000)
def thousand_formatter(x, pos):
    return "{:,.0f}".format(x)


# Alternative formatter for millions (e.g., 1 triệu)
def million_formatter(x, pos):
    return "{:.0f} triệu".format(x / 1_000_000)


# Choose formatter (uncomment the one you want)
formatter = thousand_formatter  # For 1,000,000
# formatter = million_formatter  # For 1 triệu



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
    st.image("./img/zscore.png", caption="Overview of Property Data", use_container_width=True)
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
    st.image("./img/iqr.png", caption="Overview of Property Data", use_container_width=True)

# Biểu đồ Bar Plot cho giá trung bình mỗi m² theo Quận
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

# Plot 3: Average Price per m² by District and Property Type
st.subheader('Average Price per m² by District and Property Type')
avg_price_by_district_type = filtered_df.groupby(['district', 'property_type'])['price_m2'].mean().unstack()
fig3, ax3 = plt.subplots(figsize=(14, 8))
avg_price_by_district_type.plot(kind='barh', stacked=False, colormap='tab10', ax=ax3)
plt.title('Average Price per m² by District and Property Type')
plt.xlabel('Average Price per m² (VND)')
plt.ylabel('District')
ax3.xaxis.set_major_formatter(FuncFormatter(formatter))  # Format price_m2 (x-axis)
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig3)

# Plot 4: Average Price per m² by Property Type
st.subheader('Average Price per m² by Property Type')
avg_price_by_type = filtered_df.groupby('property_type')['price_m2'].mean().reset_index()
avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False, ax=ax4)
plt.title('Average Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price per m² (VND)')
ax4.yaxis.set_major_formatter(FuncFormatter(formatter))  # Format price_m2 (y-axis)
plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig4)
