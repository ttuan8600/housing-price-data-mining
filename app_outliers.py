import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import folium
from streamlit_folium import st_folium
from data_outlier.outliers_detection import detect_zscore_outliers, detect_iqr_outliers, IQR_method, z_scoremod_method

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

st.title("🏡 Real Estate Price Analysis - Outlier Detection")
df = load_data()
method = st.selectbox("**Outlier detection method:**", ["Z-score", "IQR"])
feature_list = ['price_m2', 'area', 'price_total']

# Slider for price_total
min_price, max_price = st.slider(
    "Choose total price range (VND)",
    min_value=int(df['price_total'].min()),
    max_value=int(df['price_total'].max()),
    value=(int(df['price_total'].min()), int(df['price_total'].max())),
    step=1000000,  # optional, to make sliding smoother for big numbers
    format='%d VND'
)
st.markdown(f"**Price from:** &nbsp; {min_price:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_price:,} VND")

# Slider for area
min_area, max_area = st.slider(
    "Choose area range (m²)",
    min_value=int(df['area'].min()),
    max_value=int(df['area'].max()),
    value=(int(df['area'].min()), int(df['area'].max())),
    format='%d m²'
)
st.markdown(f"**Area from:** &nbsp; {min_area:,} &nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp; {max_area:,} m²")

# Filter data based on selected price_total and area ranges
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price) &
                 (df['area'] >= min_area) & (df['area'] <= max_area)]
filtered_df = filtered_df[~filtered_df.eq("N/A").any(axis=1)]

# Outlier detection and adding Outlier_Reason
filtered_df['is_outlier'] = False
filtered_df['Outlier_Reason'] = ''
# Phát hiện outliers
if method == "IQR":
    outliers = filtered_df.groupby('property_type', group_keys=False).apply(
        lambda g: IQR_method(g, n=1, features=feature_list))

else:
    outliers = filtered_df.groupby('property_type', group_keys=False).apply(
        lambda g: z_scoremod_method(g, n=1, features=feature_list))

st.markdown(f"### 🔍 Total number of outliers detected by {method}: **{len(outliers)}**")
# Format display DataFrame for st.dataframe
display_df = outliers[['street', 'ward', 'district', 'price_total', 'price_m2', 'area']].copy()
# Rename columns for display
display_df = display_df.rename(columns={
    'street': 'Street',
    'ward': 'Ward',
    'district': 'District',
    'price_total': 'Total Price (VND)',
    'price_m2': 'Price per m² (VND)',
    'area': 'Area (m²)'
})
# Format numeric columns with thousand separators
display_df['Total Price (VND)'] = display_df['Total Price (VND)'].apply(lambda x: "{:,}".format(int(x)))
display_df['Price per m² (VND)'] = display_df['Price per m² (VND)'].apply(lambda x: "{:,}".format(int(x)))
display_df['Area (m²)'] = display_df['Area (m²)'].apply(lambda x: "{:,}".format(int(x)))

# Display formatted DataFrame
st.write("Outliers detected:")
st.dataframe(
    display_df,
    height=6 * 35,
    use_container_width=True
)

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
# Create a Folium map centered on the average coordinates of outliers
map_center = [outliers['lat'].mean(), outliers['long'].mean()]
m = folium.Map(location=map_center, zoom_start=11)
# Define colors for different property types
colors = {'apartment': 'blue', 'house': 'green', 'land': 'red'}
districts = sorted(outliers['district'].unique())

# Add FeatureGroup for each district
for district in districts:
    fg = folium.FeatureGroup(name=district, show=False)  # Default hidden
    district_df = outliers[outliers['district'] == district]
    for _, row in district_df.iterrows():
        popup_content = (
            f"{popup_style}"
            f"<div><b>Property:</b> {row['property_type']}</div>"
            f"<div><b>Street:</b> {row['street']}</div>"
            f"<div><b>District:</b> {row['district']}</div>"
            f"<div><b>Price/m²:</b> {row['price_m2']} VNĐ</div>"
            f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
            f"<div><b>Total Price:</b> {row['price_total']} VNĐ</div>"
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
# Add FeatureGroup for all outliers (default shown)
fg_all = folium.FeatureGroup(name='Tất cả', show=True)
for _, row in outliers.iterrows():
    popup_content = (
        f"{popup_style}"
        f"<div><b>Property:</b> {row['property_type']}</div>"
        f"<div><b>Street:</b> {row['street']}</div>"
        f"<div><b>District:</b> {row['district']}</div>"
        f"<div><b>Price/m²:</b> {row['price_m2']} VNĐ</div>"
        f"<div><b>Area:</b> {row['area']:,.0f} m²</div>"
        f"<div><b>Total Price:</b> {row['price_total']} VNĐ</div>"
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

# Add LayerControl
folium.LayerControl(collapsed=True).add_to(m)

# Display the map in Streamlit
st_folium(m, width=700, height=500)



