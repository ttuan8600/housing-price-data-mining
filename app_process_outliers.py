import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from data_outlier.outliers_detection import detect_zscore_outliers, detect_iqr_outliers, IQR_method, z_scoremod_method


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
feature_list = ['price_m2', 'area', 'price_total']


# Formatter function for comma-separated values
def comma_format(x, _):
    return f'{int(x):,}'


method = st.selectbox("Choose outlier detection method:", ["Z-score", "IQR"])
st.title(f"🏡 Outliers Detection - {method} ")
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
    st.info("""
    Z-scores can be skewed by extreme values, which makes them less reliable for detecting outliers. A single outlier can distort the mean and affect all z-scores.
    A more robust method is the modified z-score, which is less sensitive to extreme values. It's calculated as:
    """)
    st.image("./img/z.png", caption="Modified Z-score", use_container_width=True)
    st.markdown("""
    Where:
    - xi: A single data value
    - x̃: The median of the dataset
    - MAD: The median absolute deviation of the dataset
    """)
    st.info("""
    - **The median absolute deviation (MAD)** measures how spread out data is, but unlike standard deviation, it's not affected much by outliers.
    - If your data is normal, use standard deviation. If not, MAD is a better choice for detecting spread.
    """)
    outliers = df.groupby('property_type', group_keys=False).apply(
        lambda g: z_scoremod_method(g, n=1, features=feature_list))
else:
    st.info("""
    ## 📚 Explanation of IQR
    **IQR (Interquartile Range)** is the range between the third quartile (Q3) and the first quartile (Q1). It helps identify points that lie outside the normal spread.
    - **Simple idea:** Imagine sorting exam scores from low to high and dividing them into 4 parts. IQR tells you the range between the middle 50%.
    - **Procedure:**
      1. Find the first quartile, Q1.
      2. Find the third quartile, Q3.
      3. Calculate the IQR. IQR = Q3-Q1.
      4. Define the normal data range with lower limit as Q1–1.5 IQR and upper limit as Q3+1.5 IQR.

    => **Any data point outside this range is considered as outlier.**
    - **Pros:** Doesn’t require normal distribution. Good for skewed data.
    - **Cons:** Might miss outliers that lie within the interquartile range.
    """)
    st.image("./img/iqr.png", caption="Overview of Property Data", use_container_width=True)
    outliers = df.groupby('property_type', group_keys=False).apply(
        lambda g: IQR_method(g, n=1, features=feature_list))
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
# st.markdown(f"### 🔍 Total number of outliers detected by {method}: **{len(outliers)}**")

# # Bar chart: Average price per m² by district
# avg_price_by_district = df.groupby('district')['price_m2'].mean().reset_index()
# chart_bar = alt.Chart(avg_price_by_district).mark_bar().encode(
#     x='district',
#     y='price_m2',
#     color='district:N',
#     tooltip=['district', 'price_m2']
# ).properties(
#     title='Average Price per m² by District'
# )
# st.altair_chart(chart_bar, use_container_width=True)
#
# # Plot: Average price per m² by district and property type
# st.subheader('Average Price per m² by District and Property Type')
# avg_price_by_district_type = df.groupby(['district', 'property_type'])['price_m2'].mean().unstack()
# fig3, ax3 = plt.subplots(figsize=(14, 8))
# avg_price_by_district_type.plot(kind='barh', stacked=False, colormap='tab10', ax=ax3)
# plt.title('Average Price per m² by District and Property Type')
# plt.xlabel('Average Price per m² (VND)')
# plt.ylabel('District')
# ax3.xaxis.set_major_formatter(FuncFormatter(comma_format))
# plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# st.pyplot(fig3)
#
# # Plot: Average price per m² by property type
# st.subheader('Average Price per m² by Property Type')
# avg_price_by_type = df.groupby('property_type')['price_m2'].mean().reset_index()
# avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)
# fig4, ax4 = plt.subplots(figsize=(8, 5))
# sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False, ax=ax4)
# plt.title('Average Price per m² by Property Type')
# plt.xlabel('Property Type')
# plt.ylabel('Average Price per m² (VND)')
# ax4.yaxis.set_major_formatter(FuncFormatter(comma_format))
# plt.xticks(rotation=15)
# plt.tight_layout()
# st.pyplot(fig4)


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
if view_option == "Chỉ Outliers":
    data_to_plot = filtered_df[filtered_df['is_outlier']]
elif view_option == "Chỉ Non-outliers":
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
