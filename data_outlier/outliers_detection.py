import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor


def IQR_multivariate_spatial_method(df, features=None, eps=0.05, min_samples=40, threshold=1.5):
    """
    Detects multivariate outliers within spatial clusters using a robust covariance-based IQR-like method.
    Parameters:
    - df: DataFrame or group, must include 'long' and 'lat' columns.
    - features: List of column names to check for outliers (e.g., ['price_m2', 'price_total']).
               If None, uses all numeric columns except 'long' and 'lat'.
    - eps: Maximum distance (in degrees) for DBSCAN clustering (default 0.01, ~1.1km).
    - min_samples: Minimum number of points to form a cluster (default 5).
    - threshold: Multiplier for IQR bounds (default 1.5).
    Returns:
    - DataFrame containing outlier rows with original columns plus 'is_outlier' and 'Outlier_Reason'.
    """
    if features is None:
        features = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in ['long', 'lat']]

    # Validate features and ensure 'long', 'lat' exist
    features = [col for col in features if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    if not features or 'long' not in df.columns or 'lat' not in df.columns:
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
        return out_df

    # Extract spatial coordinates
    coords = df[['long', 'lat']].values

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
    df['cluster'] = db.labels_

    # Initialize output DataFrame
    out_dfs = []

    # Process each cluster (excluding noise points, label -1)
    for cluster in set(df['cluster']) - {-1}:
        cluster_df = df[df['cluster'] == cluster].copy()
        if len(cluster_df) < 3:  # Skip small clusters
            continue

        # Extract feature data for the cluster
        X = cluster_df[features].copy()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use Minimum Covariance Determinant for robust covariance
        try:
            mcd = MinCovDet(random_state=42).fit(X_scaled)
            center = mcd.location_
            cov = mcd.covariance_
        except ValueError:
            continue

        # Compute Mahalanobis distance
        distances = np.array([np.sqrt(np.dot((x - center), np.linalg.solve(cov, (x - center))))
                             for x in X_scaled])

        # Apply IQR-like method on distances
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify outliers
        outlier_mask = (distances < lower_bound) | (distances > upper_bound)
        outlier_indices = cluster_df.index[outlier_mask].tolist()

        if outlier_indices:
            cluster_out_df = cluster_df.loc[outlier_indices].copy()
            cluster_out_df['is_outlier'] = True
            cluster_out_df['Outlier_Reason'] = [
                f"Spatial outlier in cluster {cluster} (distance={distances[cluster_df.index.get_loc(idx)]:.2f})"
                for idx in cluster_out_df.index
            ]
            out_dfs.append(cluster_out_df)

    # Combine results
    if not out_dfs:
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
    else:
        out_df = pd.concat(out_dfs)
        out_df['is_outlier'] = True

    # Drop temporary cluster column
    df = df.drop(columns=['cluster'], errors='ignore')
    out_df = out_df.drop(columns=['cluster'], errors='ignore')

    return out_df

def lof_multivariate_spatial_method(df, features=None, eps=0.05, min_samples=40, n_neighbors=20, lof_threshold=1.5):
    """
    Detects multivariate outliers within spatial clusters using Local Outlier Factor (LOF).
    Parameters:
    - df: DataFrame or group, must include 'long' and 'lat' columns.
    - features: List of column names to check for outliers (e.g., ['price_m2', 'price_total']).
               If None, uses all numeric columns except 'long' and 'lat'.
    - eps: Maximum distance (in degrees) for DBSCAN clustering (default 0.05, ~5.5km).
    - min_samples: Minimum number of points to form a cluster (default 40).
    - n_neighbors: Number of neighbors for LOF density estimation (default 20).
    - lof_threshold: Threshold for LOF score to identify outliers (default 1.5).
    Returns:
    - DataFrame containing outlier rows with original columns plus 'is_outlier' and 'Outlier_Reason'.
    """
    if features is None:
        features = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in ['long', 'lat']]

    # Validate features and ensure 'long', 'lat' exist
    features = [col for col in features if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    if not features or 'long' not in df.columns or 'lat' not in df.columns:
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
        return out_df

    # Extract spatial coordinates
    coords = df[['long', 'lat']].values

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
    df['cluster'] = db.labels_

    # Initialize output DataFrame
    out_dfs = []

    # Process each cluster (excluding noise points, label -1)
    for cluster in set(df['cluster']) - {-1}:
        cluster_df = df[df['cluster'] == cluster].copy()
        if len(cluster_df) < max(n_neighbors, 3):  # Ensure enough points for LOF
            continue

        # Extract feature data for the cluster
        X = cluster_df[features].copy()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean')
        lof.fit(X_scaled)
        # LOF scores are negative; higher (less negative) values indicate outliers
        lof_scores = -lof.negative_outlier_factor_

        # Identify outliers based on LOF score threshold
        outlier_mask = lof_scores > lof_threshold
        outlier_indices = cluster_df.index[outlier_mask].tolist()

        if outlier_indices:
            cluster_out_df = cluster_df.loc[outlier_indices].copy()
            cluster_out_df['is_outlier'] = True
            cluster_out_df['Outlier_Reason'] = [
                f"Spatial outlier in cluster {cluster} (LOF score={lof_scores[cluster_df.index.get_loc(idx)]:.2f})"
                for idx in cluster_out_df.index
            ]
            out_dfs.append(cluster_out_df)

    # Combine results
    if not out_dfs:
        out_df = df.iloc[0:0].copy()
        out_df['is_outlier'] = pd.Series(dtype=bool)
        out_df['Outlier_Reason'] = pd.Series(dtype=str)
    else:
        out_df = pd.concat(out_dfs)
        out_df['is_outlier'] = True

    # Drop temporary cluster column
    df = df.drop(columns=['cluster'], errors='ignore')
    out_df = out_df.drop(columns=['cluster'], errors='ignore')

    return out_df