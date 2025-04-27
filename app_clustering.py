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

# Load data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT * FROM real_estate_processed", conn)
    df = df.dropna(subset=['area', 'price_total', 'ward', 'district', 'long', 'lat']).copy()
    return df


st.title("🏡 Real Estate Price Analysis - Clustering")
df = load_data()

eps = 0.5
min_samples = 5

df = standardize_data(df)

# Elbow Method
data_scaled, scaler, elbow_inertia = get_elbow(df)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df['cluster'] = labels

# Cluster name mapping
cluster_name_mapping = {
    0: "Mid-range, near center",
    1: "High-end, prime location",
    2: "Affordable, suburban",
    3: "Luxury, large area",
    4: "Budget, remote"
}

# Cluster counts
st.write("Number of properties in each cluster:")
cluster_counts = df['cluster'].map(cluster_name_mapping).value_counts()
st.write(cluster_counts)

# Dropdown to select cluster
st.subheader("Search Data by Cluster")
cluster_options = ['All'] + list(cluster_name_mapping.values())
selected_cluster_display = st.selectbox("Select Cluster", options=cluster_options, index=0)

# Filter data based on selected cluster
if selected_cluster_display == 'All':
    cluster_df = df
else:
    selected_cluster = [k for k, v in cluster_name_mapping.items() if v == selected_cluster_display][0]
    cluster_df = df[df['cluster'] == selected_cluster]

# Format data
cluster_df_display = cluster_df.copy()
cluster_df_display['price_m2'] = cluster_df_display['price_m2'].apply(lambda x: f"{x:,.2f} million VND/m²")
cluster_df_display['price_total'] = cluster_df_display['price_total'].apply(lambda x: f"{x:,.0f} million VND")
cluster_df_display['cluster'] = cluster_df_display['cluster'].map(cluster_name_mapping)
cluster_df_display = cluster_df_display.reset_index()
columns_to_display = ['id', 'name', 'district', 'price_total', 'price_m2']
cluster_df_display = cluster_df_display[columns_to_display]

# AgGrid configuration
gb = GridOptionsBuilder.from_dataframe(cluster_df_display)
gb.configure_column('id', headerName='ID', width=80)
gb.configure_column('name', headerName='Name')
gb.configure_column('district', headerName='District')
gb.configure_column('price_total', headerName='Total Price')
gb.configure_column('price_m2', headerName='Price per m²')
gb.configure_selection('single', use_checkbox=False)
grid_options = gb.build()

# Display data table
st.subheader(f"Data for {'All Clusters' if selected_cluster_display == 'All' else selected_cluster_display}:")
grid_response = AgGrid(cluster_df_display, gridOptions=grid_options, height=400, allow_unsafe_jscode=True)

# Keyword search
search_text = st.text_input("🔍 Enter keyword to search (name, street, ward, district)", "")

# Map placeholder
map_placeholder = st.empty()

# Folium map
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
districts = sorted(df['district'].unique())
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Add layers for each district
for district in districts:
    fg = folium.FeatureGroup(name=district, show=False)
    district_df = df[df['district'] == district]
    for idx, row in district_df.iterrows():
        popup_content = (
            f"{popup_style}"
            f"<div><b>Name:</b> {row['name']}</div>"         
            f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
            f"<div><b>Price/m²:</b> {row['price_m2']:,.0f} VND</div>"
            f"<div><b>Total Price:</b> {row['price_total']:,.0f} VND</div>"
            f"<div><b>District:</b> {row['district']}</div>"
            f"<div><b>Address:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
            f"<div><b>Cluster:</b> {cluster_name_mapping[row['cluster']]}</div>"
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

# Add all data layer
fg_all = folium.FeatureGroup(name='All', show=True)
for idx, row in df.iterrows():
    popup_content = (
        f"{popup_style}"
        f"<div><b>Name:</b> {row['name']}</div>"
        f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
        f"<div><b>Price/m²:</b> {row['price_m2']:,.0f} VND</div>"
        f"<div><b>Total Price:</b> {row['price_total']:,.0f} VND</div>"
        f"<div><b>District:</b> {row['district']}</div>"
        f"<div><b>Address:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
        f"<div><b>Cluster:</b> {cluster_name_mapping[row['cluster']]}</div>"
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

# If searching
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
                f"<div><b>Name:</b> {row['name']}</div>"
                f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
                f"<div><b>Price/m²:</b> {row['price_m2']:,.0f} VND</div>"
                f"<div><b>Total Price:</b> {row['price_total']:,.0f} VND</div>"
                f"<div><b>District:</b> {row['district']}</div>"
                f"<div><b>Address:</b> {row['street']}, {row['ward']}, {row['district']}</div>"
                f"<div><b>Cluster:</b> {cluster_name_mapping[row['cluster']]}</div>"
            )
            folium.Marker(
                location=[row['lat'], row['long']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='search')
            ).add_to(m)

        first_row = filtered_df.iloc[0]
        m.location = [first_row['lat'], first_row['long']]
        m.zoom_start = 16

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Show map
st_folium(m, width=700, height=500)
