import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from data_outlier.outliers_detection import IQR_multivariate_spatial_method, lof_multivariate_spatial_method


# Load data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("data_real_estate.db")
    df = pd.read_sql_query(
        "SELECT property_type, street, ward, district, price_total, price_m2, area, long, lat FROM "
        "real_estate_processed",
        conn)
    df = df.dropna(
        subset=['property_type', 'street', 'ward', 'district', 'price_total', 'price_m2', 'area', 'long', 'lat']).copy()
    return df


df = load_data()


# Formatter function for comma-separated values
def comma_format(x, _):
    return f'{int(x):,}'


method = st.selectbox("Choose outlier detection method:", ["LOF", "IQR"])
st.title(f"🏡 Outliers Detection - {method} ")
if method == "LOF":
    st.info("""
        ## 📚Explanation of Z-Score Multivariate Spatial Outlier Detection

        **Overview**

        Local Outlier Factor (LOF) is a density-based algorithm used to identify outliers in a dataset. It measures how isolated a data point is compared to its neighbors by comparing the local density of the point to the densities of its surrounding points. Points that have a significantly lower density than their neighbors are considered outliers.

        **How It Detects Outliers**

        LOF operates by analyzing the local structure of the data. Here’s a step-by-step breakdown:

        1. **LOF Score:**

        The LOF score for a point is the average ratio of the LRD of its ( k ) neighbors to its own LRD.

        Interpretation:

        - LOF ≈ 1: The point has a similar density to its neighbors (not an outlier).
        - LOF > 1: The point has a lower density than its neighbors (potential outlier).
        - LOF < 1: The point is in a denser region than its neighbors (likely not an outlier).

        2. **Outlier Detection:**

        Points with high LOF scores (e.g., > 1.5) are flagged as outliers, indicating they are less dense (more isolated) than their neighbors.

        **Reasons for Outliers**

        In the context of real estate data, LOF is used to detect properties with unusual characteristics (e.g., `price_m2`, `price_total`, `area`) within spatial clusters defined by `long` and `lat`. For example:
        - A house with a `price_m2` of 500M VND in a neighborhood where others average 50M VND/m² will have a low local density (high LOF score) because its price is extreme compared to nearby properties.
        - LOF considers the multivariate combination of features, so a property with a reasonable `price_m2` but an unusual `price_total` and `area` combination can also be flagged.

        **Why This Approach?**

        - **Local Sensitivity:** Unlike global methods (e.g., Z-score), LOF focuses on local neighborhoods, making it effective for datasets with varying densities.
        - **Multivariate:** Handles multiple features (e.g., price_m2, price_total, area) simultaneously, capturing complex anomalies.
        - **No Distribution Assumption:** Does not assume the data follows a specific distribution, ideal for skewed real estate data.

    """)
    st.image("./img/lof.png", use_container_width=True)
    outliers = df.groupby('property_type', group_keys=False).apply(
        lambda g: lof_multivariate_spatial_method(g))
else:
    st.info("""
    ## 📚Explanation of IQR Multivariate Spatial Outlier Detection

    **Overview**

    The IQR_multivariate_spatial_method detects outliers in real estate data by combining spatial clustering (using long and lat) with a multivariate IQR-based approach. It identifies properties with anomalous prices (e.g., price_m2, price_total) compared to geographically close neighbors, addressing scenarios where a property’s price is unusually high within its locality.

    **How It Detects Outliers**

    - Spatial Clustering with DBSCAN by long and lat
    - Multivariate Outlier Detection using IQR method

    **Reasons for Outliers**

    - **High Price Discrepancy:** E.g., a house with price_m2 of 500M VND in a cluster averaging 50M VND/m².
    - **Unusual Feature Combinations:** A property’s mix of price_m2, price_total, and area is inconsistent with neighbors (e.g., high price for a small area).
    - **Spatial Context:** Anomalies are relative to nearby properties, ensuring location-specific comparisons.

    **Why This Approach?**

    - **Spatial Relevance:** Compares prices only among nearby properties, where similar market conditions apply.
    - **Multivariate Sensitivity:** Captures correlated feature anomalies via Mahalanobis Distance.
    - **Robustness:** MCD and IQR make the method resilient to outliers and skewed data.

    """)
    st.image("./img/iqr.png", caption="Overview of Property Data", use_container_width=True)
    outliers = df.groupby('property_type', group_keys=False).apply(
        lambda g: IQR_multivariate_spatial_method(g))
    # Set seaborn style
    sns.set(style="whitegrid")

    # Define features to plot
    features = {
        'price_total': 'Total Price (VND)',
        'area': 'Area (m²)',
        'price_m2': 'Price per m² (VND)'
    }

    # Create boxplots
    for feature, label in features.items():
        st.markdown(f"### 📦 Boxplot: {label} by Property Type")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='property_type', y=feature, ax=ax, palette="Set2")
        ax.set_xlabel("Property Type")
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(FuncFormatter(comma_format))

        st.pyplot(fig)


filtered_df = df.copy()

# Add is_outlier column: True if row is in outliers, False otherwise
filtered_df['is_outlier'] = filtered_df.index.isin(outliers.index)
filtered_df.loc[outliers.index, 'Outlier_Reason'] = outliers['Outlier_Reason']

# Selectbox for view options
view_option = st.selectbox(
    "Choose a filter",
    ["All", "Outliers", "Non-outliers"],
    index=0  # Default to "Both"
)

# Filter data based on view option
if view_option == "Outliers":
    data_to_plot = filtered_df[filtered_df['is_outlier']]
elif view_option == "Non-outliers":
    data_to_plot = filtered_df[~filtered_df['is_outlier']]
else:  # Cả hai (Both)
    data_to_plot = filtered_df

data = data_to_plot.copy()
display_data = data.rename(columns={
    'property_type': 'Property Type',
    'Outlier_Reason': 'Outlier Reason',
    'ward': 'Ward',
    'district': 'District',
    'price_total': 'Total Price (VND)',
    'price_m2': 'Price per m² (VND)',
    'area': 'Area (m²)'
})
# Format price columns with commas (e.g., 1,000,000)
display_data['Total Price (VND)'] = display_data['Total Price (VND)'].apply(lambda x: "{:,.0f}".format(x))
display_data['Price per m² (VND)'] = display_data['Price per m² (VND)'].apply(lambda x: "{:,.0f}".format(x))

# Display filtered data for verification
st.write("Data chosen:")
st.dataframe(
    display_data[
        ['Property Type', 'Ward', 'District', 'Total Price (VND)', 'Price per m² (VND)', 'Area (m²)',
         'Outlier Reason']],
    height=6 * 35,  # Approx. 35 pixels per row, 5 rows = 175 pixels
    use_container_width=True
)
# Plot: Area vs Total Price with outliers highlighted
st.write(f"### Scatter Plot of Total Price vs Area - {view_option}")
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
        alt.Tooltip('price_total:Q', title='Total Price (VND)', format=',.0f'),
        alt.Tooltip('price_m2:Q', title='Price per m² (VND)', format=',.0f'),
        alt.Tooltip('district:N', title='District'),
        alt.Tooltip('ward:N', title='Ward'),
        alt.Tooltip('Outlier_Reason:N', title='Reason')
    ]
).interactive()
# Display the chart in Streamlit
st.altair_chart(chart_outlier, use_container_width=True)
ptype = st.selectbox("**Choose property type:**", ["Land", "House", 'Apartment'])
# Filter data by selected property type
data_by_type = data_to_plot[data_to_plot['property_type'] == ptype.lower()]
# Create distribution chart for each feature
features = {
    'price_m2': "Price per m² (VND)",
    'price_total': "Total Price (VND)",
    'area': "Area (m²)"
}
for col, label in features.items():
    st.markdown(f"### Distribution of {label} for {ptype.capitalize()}s")
    chart = alt.Chart(data_by_type).mark_bar().encode(
        alt.X(col, bin=alt.Bin(maxbins=30), title=label),
        alt.Y('count()', title='Count'),
        color=alt.Color('is_outlier:N', legend=alt.Legend(title="Outlier"))
    ).properties(
        width='container'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

#
# area_plot = alt.Chart(df).mark_area(opacity=0.4).encode(
#     x=alt.X('area:Q', bin=alt.Bin(maxbins=40), title='Diện tích (m²)'),
#     y=alt.Y('count()', stack=None, title='Số lượng'),
#     color=alt.Color('property_type:N', title='Loại bất động sản')
# ).properties(
#     width=700,
#     height=400,
#     title='Biểu đồ phân phối diện tích theo loại bất động sản'
# )
#
# st.altair_chart(area_plot, use_container_width=True)
#
# price_plot = alt.Chart(df).mark_area(opacity=0.4).encode(
#     x=alt.X('price_total:Q', bin=alt.Bin(maxbins=40), title='Tổng giá (VND)'),
#     y=alt.Y('count()', stack=None, title='Số lượng'),
#     color=alt.Color('property_type:N', title='Loại bất động sản')
# ).properties(
#     width=700,
#     height=400,
#     title='Phân phối tổng giá theo loại bất động sản'
# )
# st.altair_chart(price_plot, use_container_width=True)
