from autogluon.tabular import TabularPredictor
import pandas as pd
import sqlite3

conn = sqlite3.connect('data_real_estate.db')

df = pd.read_sql_query ("SELECT property_type, street, ward, district, price_total, price_m2, area FROM real_estate_processed", conn)

conn.close()
predictor = TabularPredictor(label='price_total').fit(train_data=df.head(10))
predictor.predict(df.tail(10))
