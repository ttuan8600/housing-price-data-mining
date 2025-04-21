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


st.title("🏡 Real Estate Price Analysis - Outlier Detection")
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
    "Select Total Price Range (VND)",
    min_value=int(df['price_total'].min()),
    max_value=int(df['price_total'].max()),
    value=(int(df['price_total'].min()), int(df['price_total'].max())),
    step=1000000,
    format='%d VND'
)
st.markdown(f"**Price from:** &nbsp; {min_price:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_price:,} VND")

# Slider for area
min_area, max_area = st.slider(
    "Select Area Range (m²)",
    min_value=int(df['area'].min()),
    max_value=int(df['area'].max()),
    value=(int(df['area'].min()), int(df['area'].max())),
    format='%d VND'
)
st.markdown(f"**Area from:** &nbsp; {min_area:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_area:,} m²")

# Filter data based on selected price_total and area ranges
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price) &
                 (df['area'] >= min_area) & (df['area'] <= max_area)]
filtered_df = filtered_df[~filtered_df.eq("N/A").any(axis=1)]

# Formatter for thousand separators
def thousand_formatter(x, pos):
    return "{:,.0f}".format(x)


# Formatter for millions
def million_formatter(x, pos):
    return "{:.0f} million".format(x / 1_000_000)


formatter = thousand_formatter

# Plot 1: Price per m² by Property Type
st.subheader('Price per m² by Property Type')
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='property_type', y='price_m2', hue='property_type', ax=ax1)
plt.title('Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price per m² (VND)')
ax1.yaxis.set_major_formatter(FuncFormatter(formatter))
plt.tight_layout()
st.pyplot(fig1)

# Plot 2: Area vs Total Price
st.subheader('Area vs Total Price')
scatter_plot = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X('area:Q', title='Area (m²)', axis=alt.Axis(format=',.0f')),
    y=alt.Y('price_total:Q', title='Total Price (VND)'),
    color=alt.Color('property_type:N', title='Property Type', scale=alt.Scale(scheme='tableau10')),
    tooltip=[
        alt.Tooltip('property_type:N', title='Property Type'),
        alt.Tooltip('area:Q', title='Area (m²)', format=',.0f'),
        alt.Tooltip('price_total:Q', title='Total Price (VND)', format=',.0f'),
        alt.Tooltip('price_m2:Q', title='Price per m² (VND)', format=',.0f'),
        alt.Tooltip('district:N', title='District'),
        alt.Tooltip('ward:N', title='Ward'),
        alt.Tooltip('Outlier_Reason:N', title='**Reason for Outlier**')
    ]
).properties(
    width=600,
    height=400
).interactive()

st.altair_chart(scatter_plot, use_container_width=True)

# Outlier detection and adding Outlier_Reason
filtered_df['is_outlier'] = False
filtered_df['Outlier_Reason'] = ''
method = st.selectbox("Select Outlier Detection Method:", ["Z-score", "IQR"])
if method == "IQR":
    outliers = filtered_df.groupby('property_type', group_keys=False).apply(
        lambda g: IQR_method(g, n=1, features=feature_list))
    st.markdown(f"### 🔍 Total outlier properties by {method}: **{len(outliers)}**")
    outliers['price_total'] = outliers['price_total'].apply(lambda x: "{:,}".format(int(x)))
    outliers['price_m2'] = outliers['price_m2'].apply(lambda x: "{:,}".format(int(x)))
    st.dataframe(outliers[['street', 'ward', 'district', 'price_total', 'price_m2', 'area']], height=6 * 35,
                 use_container_width=True)
else:
    outliers = filtered_df.groupby('property_type', group_keys=False).apply(
        lambda g: z_scoremod_method(g, n=1, features=feature_list))
    st.markdown(f"### 🔍 Total outlier properties by {method}: **{len(outliers)}**")
    outliers['price_total'] = outliers['price_total'].apply(lambda x: "{:,}".format(int(x)))
    outliers['price_m2'] = outliers['price_m2'].apply(lambda x: "{:,}".format(int(x)))
    st.dataframe(outliers[['street', 'ward', 'district', 'price_total', 'price_m2', 'area']], height=6 * 35,
                 use_container_width=True)

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

map_center = [outliers['lat'].mean(), outliers['long'].mean()]
m = folium.Map(location=map_center, zoom_start=13)
colors = {'apartment': 'blue', 'house': 'green', 'land': 'red'}
districts = sorted(outliers['district'].unique())

for district in districts:
    fg = folium.FeatureGroup(name=district, show=False)
    district_df = outliers[outliers['district'] == district]
    for _, row in district_df.iterrows():
        popup_content = (
            f"{popup_style}"
            f"<div><b>Property:</b> {row['property_type']}</div>"
            f"<div><b>Street:</b> {row['street']}</div>"
            f"<div><b>District:</b> {row['district']}</div>"
            f"<div><b>Price/m²:</b> {row['price_m2']} VND</div>"
            f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
            f"<div><b>Total Price:</b> {row['price_total']} VND</div>"
            f"<div><b>Reason:</b> {row['Outlier_Reason']}</div>"
        )
        folium.Marker(
            location=[row['lat'], row['long']],
            radius=6,
            color=colors.get(row['property_type'], 'gray'),
            fill=True,
            fill_color=colors.get(row['property_type'], 'gray'),
            fill_opacity=0.7,
            popup=folium.Popup(popup_content),
            icon=folium.Icon(color=colors.get(row['property_type'], 'gray'))
        ).add_to(fg)
    fg.add_to(m)

fg_all = folium.FeatureGroup(name='All', show=True)
for _, row in outliers.iterrows():
    popup_content = (
        f"{popup_style}"
        f"<div><b>Property:</b> {row['property_type']}</div>"
        f"<div><b>Street:</b> {row['street']}</div>"
        f"<div><b>District:</b> {row['district']}</div>"
        f"<div><b>Price/m²:</b> {row['price_m2']} VND</div>"
        f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
        f"<div><b>Total Price:</b> {row['price_total']} VND</div>"
        f"<div><b>Reason:</b> {row['Outlier_Reason']}</div>"
    )
    folium.Marker(
        location=[row['lat'], row['long']],
        radius=6,
        color=colors.get(row['property_type'], 'gray'),
        fill=True,
        fill_color=colors.get(row['property_type'], 'gray'),
        fill_opacity=0.7,
        popup=folium.Popup(popup_content),
        icon=folium.Icon(color=colors.get(row['property_type'], 'gray'))
    ).add_to(fg_all)
fg_all.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

st_folium(m, width=700, height=500)
