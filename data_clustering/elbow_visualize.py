import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def check_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return tables

def load_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    print("Available tables:", check_tables(db_path))
    query = """
    SELECT * FROM real_estate_processed WHERE price_total > 0 AND area > 0 AND long IS NOT NULL AND lat IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df

def get_elbow(df):
    df.dropna()

    # Chọn các cột số để clustering
    data = df[['price_total', 'price_m2', 'area', 'long', 'lat']].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    return data_scaled, scaler, inertia