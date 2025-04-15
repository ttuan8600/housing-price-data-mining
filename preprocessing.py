import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_ward(df, ward_dtype=None):
    if ward_dtype is None:
        ward_dtype = pd.api.types.CategoricalDtype(categories=sorted(df['ward'].unique()))
    df['ward'] = df['ward'].astype(ward_dtype)
    df['ward_encoded'] = df['ward'].cat.codes
    return df, ward_dtype

def scale_features(df):
    features = df[['area', 'price_m2', 'ward_encoded']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return scaler, X_scaled