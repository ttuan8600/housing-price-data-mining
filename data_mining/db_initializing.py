import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
import sqlite3
from data_mining.real_estate_processed import RealEstateProcessed
from data_mining.real_estate_collected import RealStateCollected

class DBInitializing:
    def __init__(self, db_name=Config.DB_NAME):
        self.db_name = db_name
        self.conn, self.cursor = self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_estate_collected (
                name TEXT,
                url TEXT,
                address TEXT,
                price_total TEXT,
                price_m2 TEXT,
                area TEXT,
                map_data_src TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_estate_processed (
                name TEXT,
                url TEXT,
                property_type TEXT CHECK(property_type IN ('apartment', 'house', 'land')),
                street TEXT,
                ward TEXT,
                district TEXT,
                price_total FLOAT,
                price_m2 FLOAT,
                area FLOAT,
                long FLOAT,
                lat FLOAT
            )
        ''')
        conn.commit()
        return conn, cursor
    
    def insert_real_estate_collected(self, estate: RealStateCollected):
        self.cursor.execute('''
            INSERT INTO real_estate_collected (
                name, url, address, price_total, price_m2, area, map_data_src
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            estate.name,
            estate.url,
            estate.address,
            estate.price_total,
            estate.price_m2,
            estate.area,
            estate.map_data_src
        ))
        self.conn.commit()
    
    def insert_real_estate_processed(self, estate: RealEstateProcessed):
        self.cursor.execute('''
            INSERT INTO real_estate_processed (
                name, url, property_type, street, district, ward,
                price_total, price_m2, area, long, lat
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            estate.name, estate.url, estate.property_type, estate.street,
            estate.district, estate.ward, estate.price_total,
            estate.price_m2, estate.area, estate.long, estate.lat
        ))
        self.conn.commit()

    def clear_real_estate_collected(self):
        try:
            self.cursor.execute('DELETE FROM real_estate_collected')
            self.conn.commit()
            print("[real_estate_collected cleared]")
        except sqlite3.Error as e:
            print(f"[Error clearing real_estate_collected]: {e}")

    def clear_real_estate_processed(self):
        try:
            self.cursor.execute('DELETE FROM real_estate_processed')
            self.conn.commit()
            print("[real_estate_processed cleared]")
        except sqlite3.Error as e:
            print(f"[Error clearing real_estate_processed]: {e}")

    def close(self):
        self.conn.close()
        print("[Database connection closed]")

if __name__ == "__main__":
    db_init_obj = DBInitializing()
    db_init_obj.clear_real_estate_collected()
    db_init_obj.clear_real_estate_processed()
    db_init_obj.close()

