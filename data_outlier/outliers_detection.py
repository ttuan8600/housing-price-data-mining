# without using ML model
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import plotly.express as px
import folium

# conn = sqlite3.connect('data_real_estate.db')
#
# df = pd.read_sql_query ("SELECT property_type, street, ward, district, price_total, price_m2, area, long, lat FROM real_estate_processed", conn)
#
# conn.close()

# print(df.tail(5))
#
# print(df.describe())

# Plot: Price per m2 by Property Type
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df, x='property_type', y='price_m2',hue='property_type',)
# plt.title('Price per m² by Property Type')
# plt.xlabel('Property Type')
# plt.ylabel('Price per m² (VND)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# area vs total price
# plt.figure(figsize=(12, 7))
# sns.scatterplot(
#     data=df,
#     x='area',
#     y='price_total',
#     hue='property_type',
#     alpha=0.7,
#     edgecolor=None
# )

# plt.title('Total Price vs Area by Property Type')
# plt.xlabel('Area (m²)')
# plt.ylabel('Total Price (VND)')
# plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Group by both district and property_type
# avg_price_by_district_type = df.groupby(['district', 'property_type'])['price_m2'].mean().unstack()

# Plot
# plt.figure(figsize=(14, 8))
# avg_price_by_district_type.plot(kind='barh', stacked=False, colormap='tab10', figsize=(14, 8))
#
# plt.title('Average Price per m² by District and Property Type')
# plt.xlabel('Average Price per m² (VND)')
# plt.ylabel('District')
# plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# avg price per m2 by type
# avg_price_by_type = df.groupby('property_type')['price_m2'].mean().reset_index()

# Sort values (optional)
# avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)

# Plot
# plt.figure(figsize=(8, 5))
# sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False)
# plt.title('Average Price per m² by Property Type')
# plt.xlabel('Property Type')
# plt.ylabel('Average Price per m²')
# plt.xticks(rotation=15)
# plt.tight_layout()
# plt.show()

# using iqr
def detect_iqr_outliers(data):
    Q1 = data['price_m2'].quantile(0.25)
    Q3 = data['price_m2'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data['price_m2'] < lower_bound) | (data['price_m2'] > upper_bound)]

# detect outliers in price_m2
# print('-----------------------------------------------------')
# print('Outliers IQR')
# iqr_outliers = df.groupby('property_type', group_keys=False).apply(lambda g: detect_iqr_outliers(g))
# print(f'Outliers in price_m2 using IQR: {len(iqr_outliers)}')
# print(iqr_outliers.head())
# for prop_type, group in iqr_outliers.groupby('property_type'):
#     print(f'\nTop 5 IQR outliers for property type: {prop_type}')
#     print(group[['street', 'price_m2', 'area', 'district', 'long', 'lat']]
#           .sort_values('price_m2', ascending=False)
#           .head())
#
# outliers = iqr_outliers  # Replace with: outliers = iqr_outliers

# Create a Folium map centered on the average coordinates
# map_center = [outliers['long'].mean(), outliers['lat'].mean()]
# m = folium.Map(location=map_center, zoom_start=13)
#
# # Define colors for different property types
# colors = {'apartment': 'blue', 'house': 'green', 'land': 'red'}
#
# # Add markers for each outlier
# for _, row in outliers.iterrows():
#     popup_text = (f"Property: {row['property_type']}<br>"
#                   f"Street: {row['street']}<br>"
#                   f"District: {row['district']}<br>"
#                   f"Price/m²: {row['price_m2']}<br>"
#                   f"Area: {row['area']} m²")
#     folium.Marker(
#         location=[row['long'], row['lat']],
#         popup=popup_text,
#         icon=folium.Icon(color=colors.get(row['property_type'], 'gray'))
#     ).add_to(m)
#
# # Save the map to an HTML file
# m.save('iqr_outliers_map.html')
# print("Map saved as 'outliers_map.html'. Open it in a web browser to view.")
#
# # Plotting
# plt.figure(figsize=(12, 6))
# sns.stripplot(x='property_type', y='price_m2', hue='property_type',
#               data=iqr_outliers, jitter=True, palette='Set2', legend=False)
# plt.title('IQR Outliers: Price per m² by Property Type')
# plt.xlabel('Property Type')
# plt.ylabel('Price per m² (VND)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# print('-----------------------------------------------------')
# print('Outliers Z-Score')
# # using zscore
# df['zscore_price_m2'] = zscore(df['price_m2'])

# Set a threshold for outliers
def detect_zscore_outliers(data):
    data = data.copy()
    data['z_score'] = zscore(data['price_m2'])
    return data[(data['z_score'] > 2.5) | (data['z_score'] < -2.5)]

# zscore_outliers = df.groupby('property_type', group_keys=False).apply(
#     lambda g: detect_zscore_outliers(g)
# )
# print(f'Outliers in price_m2 using Z-score: {len(zscore_outliers)}')
#
# for prop_type, group in zscore_outliers.groupby('property_type'):
#     print(f'\nTop 5 Z-score outliers for property type: {prop_type}')
#     print(group[['street', 'price_m2', 'area', 'district', 'zscore_price_m2', 'long', 'lat']]
#           .sort_values('zscore_price_m2', ascending=False)
#           .head())
# # Z-score Outliers
# plt.figure(figsize=(12, 6))
# sns.stripplot(x='property_type', y='price_m2', hue='property_type', data=zscore_outliers, jitter=True, palette='Set1', legend=False)
# plt.title('Z-Score Outliers: Price per m² by Property Type')
# plt.xlabel('Property Type')
# plt.ylabel('Price per m² (VND)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# outliers = zscore_outliers  # Replace with: outliers = iqr_outliers
#
# # Create a Folium map centered on the average coordinates
# map_center = [outliers['long'].mean(), outliers['lat'].mean()]
# m = folium.Map(location=map_center, zoom_start=13)
#
# # Define colors for different property types
# colors = {'apartment': 'blue', 'house': 'green', 'land': 'red'}
#
# # Add markers for each outlier
# for _, row in outliers.iterrows():
#     popup_text = (f"Property: {row['property_type']}<br>"
#                   f"Street: {row['street']}<br>"
#                   f"District: {row['district']}<br>"
#                   f"Price/m²: {row['price_m2']}<br>"
#                   f"Area: {row['area']} m²")
#     folium.Marker(
#         location=[row['long'], row['lat']],
#         popup=popup_text,
#         icon=folium.Icon(color=colors.get(row['property_type'], 'gray'))
#     ).add_to(m)
#
# # Save the map to an HTML file
# m.save('zscore_outliers_map.html')
# print("Map saved as 'zscore_outliers_map.html'. Open it in a web browser to view.")
# print(zscore_outliers.head())
