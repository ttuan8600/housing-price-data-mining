import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from db_initializing import DBInitializing
import plotly.express as px

class DataVisualization:
    def __init__(self, db_initializer: DBInitializing):
        self.db_initializer = db_initializer

    def fetch_processed_data(self):
        query = "SELECT * FROM real_estate_processed"
        df = pd.read_sql_query(query, self.db_initializer.conn)
        return df

    def plot_area(self):
        df = self.fetch_processed_data()

        # Thresholds
        small_area_threshold = 20
        large_area_threshold = 500

        # Label area category
        def categorize_area(area):
            if area < small_area_threshold:
                return 'Small Area'
            elif area > large_area_threshold:
                return 'Large Area'
            else:
                return 'Normal Area'

        df['area_category'] = df['area'].apply(categorize_area)

        # Plot
        fig = px.scatter(
            df,
            x='id',
            y='area',
            color='area_category',
            symbol='area_category',
            title='Area Distribution by ID',
            hover_data=['id', 'area']
        )

        fig.update_traces(marker=dict(size=8))
        fig.update_layout(xaxis_title="Property ID", yaxis_title="Area (m²)")
        fig.show()

    def plot_price_total_vs_estimated(self):
        df = self.fetch_processed_data()
        df['estimated_price_total'] = df['price_m2'] * df['area']

        # Melt dataframe to long format for Plotly
        df_melted = df.melt(
            id_vars=['id'],
            value_vars=['price_total', 'estimated_price_total'],
            var_name='Price Type',
            value_name='Price'
        )

        fig = px.scatter(
            df_melted,
            x='id',
            y='Price',
            color='Price Type',
            symbol='Price Type',
            title="Actual vs Estimated Total Price by Property ID",
            hover_data=['id', 'Price Type', 'Price']
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(xaxis_title="Property ID", yaxis_title="Total Price")
        fig.show()

    def plot_property_type_distribution(self):
        df = self.fetch_processed_data()
        sns.countplot(x='property_type', data=df)
        plt.title("Property Type Distribution")
        plt.xlabel("Property Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_district_counts(self):
        df = self.fetch_processed_data()
        df['district'].value_counts().plot(kind='bar')
        plt.title("Property Counts by District")
        plt.xlabel("District")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    db_i_obj = DBInitializing()
    da_vi_obj = DataVisualization(db_i_obj)
    # da_vi_obj.plot_area()
    da_vi_obj.plot_price_total_vs_estimated()
    # da_vi_obj.plot_property_type_distribution()
    # da_vi_obj.plot_district_counts()