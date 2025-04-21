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


st.title("🏡 Phân tích giá nhà đất - Phát hiện bất thường")
df = load_data()
feature_list = ['price_m2', 'area', 'price_total']


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

st.info("""
## 📚 Giải thích về IQR
**IQR** là khoảng cách giữa phân vị thứ ba (Q3) và phân vị thứ nhất (Q1). Nó giúp xác định những điểm nào nằm ngoài khoảng bình thường.
- **Cách hiểu đơn giản:** Hãy tưởng tượng bạn xếp các bài kiểm tra từ thấp đến cao và chia nhóm chúng thành 4 phần. **IQR** giúp bạn hiểu được khoảng giữa 25% số điểm thấp nhất và 25% số điểm cao nhất.
- **Ví dụ:**
  - Nếu điểm nằm ngoài khoảng từ **Q1 - 1.5 * IQR** đến **Q3 + 1.5 * IQR**, nó có thể là **outlier**.
- **Ưu điểm:** Không cần dữ liệu phân phối chuẩn. Tốt cho dữ liệu có sự phân tán lớn.
- **Nhược điểm:** Không phát hiện được các outliers nếu chúng nằm trong khoảng giữa các phân vị.
""")

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


