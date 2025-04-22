# without using ML model
import sqlite3
import pandas as pd
from scipy.stats import zscore
from scipy.stats import median_abs_deviation
import numpy as np


# conn = sqlite3.connect('../data_real_estate.db')
#
# df = pd.read_sql_query(
#     "SELECT property_type, street, ward, district, price_total, price_m2, area, long, lat FROM real_estate_processed",
#     conn)
#
# conn.close()
#
# feature_list = ['price_m2', 'area', 'price_total']

# using iqr
def detect_iqr_outliers(data):
    Q1 = data['price_m2'].quantile(0.25)
    Q3 = data['price_m2'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data['price_m2'] < lower_bound) | (data['price_m2'] > upper_bound)]


# Set a threshold for outliers
def detect_zscore_outliers(data):
    data = data.copy()
    threshold = 2
    data['z_score'] = zscore(data['price_m2'])
    return data[(data['z_score'] > threshold) | (data['z_score'] < -threshold)]


def IQR_method(df, n=1, features=None):
    """
    Detects outliers using the IQR method, returning the outlier rows from the input DataFrame.
    Parameters:
    - df: DataFrame or group.
    - n: Minimum number of features an observation must be an outlier in.
    - features: List of column names to check.
    Returns:
    - DataFrame containing outlier rows with original columns plus 'is_outlier' and 'Outlier_Reason'.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_dict = {}  # Store index: [reasons]

    for column in features:
        if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
            continue
        Q1 = np.percentile(df[column], 20)
        Q3 = np.percentile(df[column], 80)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        lower_bound = Q1 - outlier_step
        upper_bound = Q3 + outlier_step
        # Check for low and high outliers separately
        low_outliers = df[df[column] < lower_bound].index
        high_outliers = df[df[column] > upper_bound].index
        for idx in low_outliers:
            if idx not in outlier_dict:
                outlier_dict[idx] = []
            outlier_dict[idx].append(f"Low {column}")
        for idx in high_outliers:
            if idx not in outlier_dict:
                outlier_dict[idx] = []
            outlier_dict[idx].append(f"High {column}")

    # Filter observations with >= n outlier features
    outlier_indices = [
        idx for idx, cols in outlier_dict.items() if len(cols) >= n
    ]

    # Create the output DataFrame with outlier rows
    if not outlier_indices:
        # Return empty DataFrame with same columns plus is_outlier and Outlier_Reason
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
        return out_df

    out_df = df.loc[outlier_indices].copy()
    out_df['is_outlier'] = True
    out_df['Outlier_Reason'] = [
        ', '.join(outlier_dict[idx]) for idx in out_df.index
    ]

    return out_df


def z_scoremod_method(df, n=1, features=None):
    """
    Detects outliers using the modified Z-score method, returning the outlier rows from the input DataFrame.
    Parameters:
    - df: DataFrame or group.
    - n: Minimum number of features an observation must be an outlier in.
    - features: List of column names to check.
    Returns:
    - DataFrame containing outlier rows with original columns plus 'is_outlier' and 'Outlier_Reason'.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_dict = {}  # Store index: [reasons]
    threshold = 6

    for column in features:
        if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
            continue
        median = df[column].median()
        mad = median_abs_deviation(df[column])
        if mad == 0:
            continue  # Skip if MAD is zero to avoid division by zero
        mod_z_score = 0.6745 * (df[column] - median) / mad
        # Check for low (negative) and high (positive) outliers
        low_outliers = df[mod_z_score < -threshold].index
        high_outliers = df[mod_z_score > threshold].index
        for idx in low_outliers:
            if idx not in outlier_dict:
                outlier_dict[idx] = []
            if column == 'price_total':
                outlier_dict[idx].append(f"Low total price")
            if column == 'price_m2':
                outlier_dict[idx].append(f"Low price per m²")
            if column == 'area':
                outlier_dict[idx].append(f"Low area")
        for idx in high_outliers:
            if idx not in outlier_dict:
                outlier_dict[idx] = []
            if column == 'price_total':
                outlier_dict[idx].append(f"High total price")
            if column == 'price_m2':
                outlier_dict[idx].append(f"High price per m²")
            if column == 'area':
                outlier_dict[idx].append(f"High area")

    # Filter observations with >= n outlier features
    outlier_indices = [
        idx for idx, cols in outlier_dict.items() if len(cols) >= n
    ]

    # Create the output DataFrame with outlier rows
    if not outlier_indices:
        # Return empty DataFrame with same columns plus is_outlier and Outlier_Reason
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
        return out_df

    out_df = df.loc[outlier_indices].copy()
    out_df['is_outlier'] = True
    out_df['Outlier_Reason'] = [
        ', '.join(outlier_dict[idx]) for idx in out_df.index
    ]

    return out_df

# Outliers_IQR = IQR_method(df, 1, feature_list)
# print(Outliers_IQR)
# Outliers_z_score = z_scoremod_method(df, 1, feature_list)
# print(Outliers_z_score)
