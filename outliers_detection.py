# without using ML model
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


conn = sqlite3.connect('data_real_estate.db')

df = pd.read_sql_query ("SELECT property_type, street, ward, district, price_total, price_m2, area FROM real_estate_processed", conn)

conn.close()

# print(df.tail(5))
#
# print(df.describe())

# price per m2
plt.figure(figsize=(10, 6))
sns.histplot(df['price_m2'], bins=30, kde=True)
plt.title('Distribution of Price per m²')
plt.xlabel('Price per m² (VND)')
plt.ylabel('Frequency')
plt.show()

# area vs total price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price_total', hue='property_type', data=df)
plt.title('Total Price vs Area')
plt.xlabel('Area (m²)')
plt.ylabel('Total Price (VND)')
plt.legend(title='Property Type')
plt.show()

# avg price per district
avg_price_by_district = df.groupby('district')['price_m2'].mean().sort_values()

plt.figure(figsize=(12, 6))
avg_price_by_district.plot(kind='barh', color='skyblue')
plt.title('Average Price per m² by District')
plt.xlabel('Average Price per m² (VND)')
plt.ylabel('District')
plt.show()

# avg price per m2 by type
avg_price_by_type = df.groupby('property_type')['price_m2'].mean().reset_index()

# Sort values (optional)
avg_price_by_type = avg_price_by_type.sort_values(by='price_m2', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_price_by_type, x='property_type', y='price_m2', hue='property_type', legend=False)
plt.title('Average Price per m² by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price per m²')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
# using iqr
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# detect outliers in price_m2
print('-----------------------------------------------------')
print('Outliers IQR')
outliers_price_m2 = detect_outliers_iqr(df, 'price_m2')
print(f'Number of outliers in price_m2: {len(outliers_price_m2)}')

# view outliers
print(outliers_price_m2[['street', 'price_m2', 'area', 'district']].head())

# Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='area', y='price_m2', label='Normal', alpha=0.6)
sns.scatterplot(data=outliers_price_m2, x='area', y='price_m2', color='orange', label='IQR Outlier', s=100, edgecolor='black')
plt.title('Outlier Detection in Price per m² (IQR Method)')
plt.xlabel('Area (m²)')
plt.ylabel('Price per m² (VND)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print('-----------------------------------------------------')
print('Outliers Z-Score')
# using zscore
df['zscore_price_m2'] = zscore(df['price_m2'])

# Set a threshold for outliers
threshold = 2.5
outliers_zscore = df[abs(df['zscore_price_m2']) > threshold]

print(f'Outliers in price_m2 using Z-score: {len(outliers_zscore)}')
print(outliers_zscore[['street', 'price_m2', 'area', 'district', 'zscore_price_m2']].sort_values('zscore_price_m2', ascending=False).head())

# Plotting: Area vs Price_m2 with Outliers Highlighted
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='area', y='price_m2', label='Normal', alpha=0.5)
sns.scatterplot(data=outliers_zscore, x='area', y='price_m2', color='red', label='Outlier', s=100, edgecolor='black')
plt.title('Outlier Detection in Price per m² (Z-score Method)')
plt.xlabel('Area (m²)')
plt.ylabel('Price per m² (VND)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()