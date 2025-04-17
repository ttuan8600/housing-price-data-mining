import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import folium
from streamlit_folium import st_folium
from preprocessing import encode_ward, scale_features
from data_outlier.outliers_detection import detect_zscore_outliers, detect_iqr_outliers
from matplotlib import pyplot as plt
import seaborn as sns

# Load data từ SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query("SELECT property_type, street, ward, district, price_total, price_m2, area, long, lat FROM real_estate_processed", conn)
    df = df.dropna(subset=['property_type', 'street', 'ward', 'district', 'price_total', 'price_m2', 'area', 'long', 'lat']).copy()
    return df

st.title("🏡 Phân tích giá nhà đất - Phát hiện bất thường")
df = load_data()
# Slider for price_total
min_price, max_price = st.slider("Chọn phạm vi giá trị tổng (VND)",
                                 min_value=int(df['price_total'].min()),
                                 max_value=int(df['price_total'].max()),
                                 value=(int(df['price_total'].min()), int(df['price_total'].max())),
                                 format="%s VND")

# Slider for area
min_area, max_area = st.slider("Chọn phạm vi diện tích (m²)",
                               min_value=int(df['area'].min()),
                               max_value=int(df['area'].max()),
                               value=(int(df['area'].min()), int(df['area'].max())),
                               format="%s m²")

# Filter data based on selected price_total and area ranges
filtered_df = df[(df['price_total'] >= min_price) & (df['price_total'] <= max_price) &
                (df['area'] >= min_area) & (df['area'] <= max_area)]
# df['price_total'] = df['price_total'].apply(lambda x: "{:,}".format(int(x)))
st.subheader('Price per m² by Property Type')
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='property_type', y='price_m2', hue='property_type', ax=ax1)
plt.title('Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price per m² (VND)')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the first plot in Streamlit
st.pyplot(fig1)

st.subheader('Area vs Total Price')
fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=filtered_df,
    x='area',
    y='price_total',
    hue='property_type',
    alpha=0.7,
    edgecolor=None,
    ax=ax2
)
plt.title('Total Price vs Area by Property Type')
plt.xlabel('Area (m²)')
plt.ylabel('Total Price (VND)')
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Display the second plot in Streamlit
st.pyplot(fig2)
st.subheader('Average Pricer per m2 by District and Property type')
# Group by both district and property_type
avg_price_by_district_type = filtered_df.groupby(['district', 'property_type'])['price_m2'].mean().unstack()

fig3, ax3 = plt.subplots(figsize=(14, 8))
avg_price_by_district_type.plot(kind='barh', stacked=False, colormap='tab10', ax=ax3)

# Customize plot
plt.title('Average Price per m² by District and Property Type')
plt.xlabel('Average Price per m² (VND)')
plt.ylabel('District')
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the first plot in Streamlit
st.pyplot(fig3)

st.subheader('Average price per m2 by Property Type')
# Group by property_type
avg_price_by_type = filtered_df.groupby('property_type')['price_m2'].mean().reset_index()

# Sort values (optional)
avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)

# Create figure
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False, ax=ax4)

# Customize plot
plt.title('Average Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price per m²')
plt.xticks(rotation=15)
plt.tight_layout()

# Display the second plot in Streamlit
st.pyplot(fig4)

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
if method == "IQR":
    iqr_outliers = filtered_df.groupby('property_type', group_keys=False).apply(lambda g: detect_iqr_outliers(g))
    outliers = iqr_outliers
    st.markdown(f"### 🔍 Tổng số bất động sản bất thường bởi {method}: **{len(iqr_outliers)}**")
    iqr_outliers['price_total'] = iqr_outliers['price_total'].apply(lambda x: "{:,}".format(int(x)))
    iqr_outliers['price_m2'] = iqr_outliers['price_m2'].apply(lambda x: "{:,}".format(int(x)))
    st.dataframe(iqr_outliers[['street', 'ward', 'district', 'price_total', 'price_m2', 'area']], height=6*35, use_container_width=True)
    for prop_type, group in iqr_outliers.groupby('property_type'):
        st.write(f'Top 5 IQR outliers for property type: {prop_type}')
        st.write(group[['street', 'price_m2', 'area', 'price_total','district']]
              .sort_values('price_m2', ascending=False)
              .head())
    st.subheader(f'{method} Outliers for Price per m² by Property Type')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(x='property_type', y='price_m2', hue='property_type',
                  data=iqr_outliers, jitter=True, palette='Set2', legend=False, ax=ax)
    plt.title(f'{method} Outliers: Price per m² by Property Type')
    plt.xlabel('Property Type')
    plt.ylabel('Price per m² (VND)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
else:
    zscore_outliers = filtered_df.groupby('property_type', group_keys=False).apply(lambda g: detect_zscore_outliers(g))
    outliers = zscore_outliers
    st.markdown(f"### 🔍 Tổng số bất động sản bất thường bởi {method}: **{len(zscore_outliers)}**")
    zscore_outliers['price_total'] = zscore_outliers['price_total'].apply(lambda x: "{:,}".format(int(x)))
    zscore_outliers['price_m2'] = zscore_outliers['price_m2'].apply(lambda x: "{:,}".format(int(x)))
    st.dataframe(zscore_outliers[['street', 'ward', 'district', 'price_total', 'price_m2', 'area']], height=6*35, use_container_width=True)
    for prop_type, group in zscore_outliers.groupby('property_type'):
        st.write(f'Top 5 IQR outliers for property type: {prop_type}')
        st.write(group[['street', 'price_m2', 'area', 'price_total','district']]
              .sort_values('price_m2', ascending=False)
              .head())
    st.subheader(f'{method} Outliers for Price per m² by Property Type')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(x='property_type', y='price_m2', hue='property_type',
                  data=zscore_outliers, jitter=True, palette='Set2', legend=False, ax=ax)
    plt.title(f'{method} Outliers: Price per m² by Property Type')
    plt.xlabel('Property Type')
    plt.ylabel('Price per m² (VND)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

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
map_center = [outliers['long'].mean(), outliers['lat'].mean()]
m = folium.Map(location=map_center, zoom_start=13)
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
        )
        folium.Marker(
            location=[row['long'], row['lat']],
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
    )
    folium.Marker(
        location=[row['long'], row['lat']],
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

filtered_df = filtered_df.copy()

# Add is_outlier column: True if row is in outliers, False otherwise
filtered_df['is_outlier'] = filtered_df.index.isin(outliers.index)
filtered_df['price_total_formatted'] = filtered_df['price_total'].apply(lambda x: "{:,}".format(int(x)))
filtered_df['price_m2_formatted'] = filtered_df['price_m2'].apply(lambda x: "{:,}".format(int(x)))

# Selectbox for view options
view_option = st.selectbox(
    "Chọn dữ liệu để hiển thị",
    ["Cả hai", "Chỉ Outliers", "Chỉ Non-outliers"],
    index=0  # Default to "Both"
)

# Filter data based on view option
if view_option == "Chỉ Outliers":
    data_to_plot = filtered_df[filtered_df['is_outlier']]
elif view_option == "Chỉ Non-outliers":
    data_to_plot = filtered_df[~filtered_df['is_outlier']]
else:  # Cả hai (Both)
    data_to_plot = filtered_df

# Display filtered data for verification
st.write("Dữ liệu được chọn để vẽ:")
st.dataframe(
    data_to_plot[['property_type', 'ward', 'district', 'price_total_formatted', 'price_m2_formatted', 'area', 'is_outlier' ]],
    height=6 * 35,  # Approx. 35 pixels per row, 5 rows = 175 pixels
    use_container_width=True
)

# Plot: Area vs Total Price with outliers highlighted
st.write(f"### Biểu đồ phân bố - {view_option}")
chart_outlier = alt.Chart(data_to_plot).mark_circle(size=60).encode(
    x=alt.X('area:Q', title='Area (m²)'),
    y=alt.Y('price_total:Q', title='Total Price (VND)'),
    color=alt.condition(
        alt.datum.is_outlier,
        alt.value("red"),  # Outliers in red
        alt.value("steelblue")  # Non-outliers in steelblue
    ),
    tooltip=[
        alt.Tooltip('property_type:N', title='Type'),
        alt.Tooltip('area:Q', title='Area (m²)'),
        alt.Tooltip('price_total:Q', title='Total Price (VND)'),
        alt.Tooltip('district:N', title='District'),
        alt.Tooltip('ward:N', title='Ward')
    ]
).interactive()

# Display the chart in Streamlit
st.altair_chart(chart_outlier, use_container_width=True)

# Biểu đồ Bar Plot cho giá trung bình mỗi m² theo Quận
st.write("Giá trung bình mỗi m² theo Quận")
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
st.write("### Phân phối giá mỗi m²")
chart_price_m2 = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('price_m2', bin=alt.Bin(maxbins=30), title='Giá mỗi m² (VND)'),
    alt.Y('count()', title='Số lượng'),
    color=alt.Color('is_outlier:N', legend=alt.Legend(title="Outlier"))
).properties(
    title='Phân phối giá mỗi m² của bất động sản'
)
st.altair_chart(chart_price_m2, use_container_width=True)
