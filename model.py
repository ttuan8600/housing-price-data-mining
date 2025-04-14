from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy import stats
import numpy as np
from scipy.stats import zscore
def train_kmeans(X_scaled, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X_scaled)
    return model, clusters, 'kmeans'

def train_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)
    return model, clusters, 'dbscan'

# def detect_outliers_zscore(df):
#     z_scores = np.abs(stats.zscore(df[['area', 'price_total']]))
#     outliers = (z_scores > 3).any(axis=1)
#     return outliers

def detect_outliers_zscore(df):
    df['zscore_price_m2'] = zscore(df['price_m2'])
    threshold = 2.5
    outliers = abs(df['zscore_price_m2']) > threshold
    return outliers

def detect_outliers_iqr(df):
    Q1 = df[['area', 'price_total']].quantile(0.25)
    Q3 = df[['area', 'price_total']].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[['area', 'price_total']] < (Q1 - 1.5 * IQR)) |
                (df[['area', 'price_total']] > (Q3 + 1.5 * IQR))).any(axis=1)
    return outliers