def standardize_data(df):
    # Định nghĩa phạm vi tọa độ của Đà Nẵng
    long_min, long_max = 108.0, 108.3
    lat_min, lat_max = 15.9, 16.2

    # Lọc các bản ghi nằm trong phạm vi tọa độ của Đà Nẵng
    df = df[
        (df['long'].between(long_min, long_max)) &
        (df['lat'].between(lat_min, lat_max))
        ]

    return df