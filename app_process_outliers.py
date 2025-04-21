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

# Load data from SQLite
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
method = st.selectbox("Choose outlier detection method:", ["Z-score", "IQR"])
st.title(f"🏡 Outliers Detection - {method} ")

# Custom formatter function for price_total
def format_price(value):
    return "{:,.0f} VND".format(value)

# Custom formatter function for area
def format_area(value):
    return "{:,.0f} m²".format(value)

# Slider for price_total
min_price, max_price = st.slider(
    "Select total price range (VND)",
    min_value=int(df['price_total'].min()),
    max_value=int(df['price_total'].max()),
    value=(int(df['price_total'].min()), int(df['price_total'].max())),
    step=1000000,
    format='%d VND'
)
st.markdown(f"**From price:** &nbsp; {min_price:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_price:,} VND")

# Slider for area
min_area, max_area = st.slider(
    "Select area range (m²)",
    min_value=int(df['area'].min()),
    max_value=int(df['area'].max()),
    value=(int(df['area'].min()), int(df['area'].max())),
    format='%d m²'
)
st.markdown(f"**From area:** &nbsp; {min_area:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_area:,} m²")

# Filter data based on selected price_total and area ranges
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price) &
                 (df['area'] >= min_area) & (df['area'] <= max_area)]
filtered_df = filtered_df[~filtered_df.eq("N/A").any(axis=1)]

# Formatter for thousand separators (e.g., 1,000,000)
def thousand_formatter(x, pos):
    return "{:,.0f}".format(x)

# Alternative formatter for millions (e.g., 1 million)
def million_formatter(x, pos):
    return "{:.0f} million".format(x / 1_000_000)

# Choose formatter
formatter = thousand_formatter
# formatter = million_formatter

if method == "Z-score":
    st.info("""
    ## 📚 Explanation of Z-score
    **Z-score** measures how far a data point deviates from the mean of the dataset, in units of standard deviation.
    - **Simple idea:** Imagine your data as exam scores. Z-score tells you how "high" or "low" a score is compared to the class average.
    - **Example:**
      - **Z-score = 2** means the value is 2 standard deviations away from the mean, possibly an **outlier**.
      - **Z-score = 0** means it's equal to the mean.
    - **Pros:** Easy to understand and fast to compute. Good for normally distributed data.
    - **Cons:** Assumes data is normally distributed.
    """)
    st.image("./img/zscore.png", caption="Overview of Property Data", use_container_width=True)
else:
    st.info("""
    ## 📚 Explanation of IQR
    **IQR (Interquartile Range)** is the range between the third quartile (Q3) and the first quartile (Q1). It helps identify points that lie outside the normal spread.
    - **Simple idea:** Imagine sorting exam scores from low to high and dividing them into 4 parts. IQR tells you the range between the middle 50%.
    - **Example:**
      - A value outside **Q1 - 1.5 * IQR** to **Q3 + 1.5 * IQR** could be an **outlier**.
    - **Pros:** Doesn’t require normal distribution. Good for skewed data.
    - **Cons:** Might miss outliers that lie within the interquartile range.
    """)
    st.image("./img/iqr.png", caption="Overview of Property Data", use_container_width=True)

# Bar chart: Average price per m² by district
avg_price_by_district = filtered_df.groupby('district')['price_m2'].mean().reset_index()
chart_bar = alt.Chart(avg_price_by_district).mark_bar().encode(
    x='district',
    y='price_m2',
    color='district:N',
    tooltip=['district', 'price_m2']
).properties(
    title='Average Price per m² by District'
)
st.altair_chart(chart_bar, use_container_width=True)

# Plot: Average price per m² by district and property type
st.subheader('Average Price per m² by District and Property Type')
avg_price_by_district_type = filtered_df.groupby(['district', 'property_type'])['price_m2'].mean().unstack()
fig3, ax3 = plt.subplots(figsize=(14, 8))
avg_price_by_district_type.plot(kind='barh', stacked=False, colormap='tab10', ax=ax3)
plt.title('Average Price per m² by District and Property Type')
plt.xlabel('Average Price per m² (VND)')
plt.ylabel('District')
ax3.xaxis.set_major_formatter(FuncFormatter(formatter))
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig3)

# Plot: Average price per m² by property type
st.subheader('Average Price per m² by Property Type')
avg_price_by_type = filtered_df.groupby('property_type')['price_m2'].mean().reset_index()
avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False, ax=ax4)
plt.title('Average Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price per m² (VND)')
ax4.yaxis.set_major_formatter(FuncFormatter(formatter))
plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig4)
