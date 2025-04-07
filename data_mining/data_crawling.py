from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from utils import ParserUtils
from real_estate_collected import RealStateCollected
from real_estate_processed import RealEstateProcessed
from db_initializing import DBInitializing

class DataCrawling:
    def __init__(self, url):
        self.url = url
        self.driver = self.init_driver()

    def init_driver(self):
        chrome_options = Options()
        # chrome_options.add_argument("--headless=new")  # Disable for dev
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--start-maximized")
        service = Service(Config.CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        return driver

    def crawl(self):
        self.driver.get(self.url)
        time.sleep(2)

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "js__product-link-for-product-id"))
            )
            product_links = self.driver.find_elements(By.CLASS_NAME, "js__product-link-for-product-id")
            hrefs = [link.get_attribute("href") for link in product_links if link.get_attribute("href")]

        except Exception as e:
            print(f"[Error while getting links]: {type(e).__name__}: {e}")
            return []

        self.driver.quit()
        results = []

        for href in hrefs:
            try:
                self.driver = self.init_driver()
                self.driver.get(href)
                time.sleep(1.5)

                try:
                    name = self.driver.find_element(By.CSS_SELECTOR, "h1.re__pr-title.pr-title.js__pr-title").text
                except Exception as e:
                    print(f"[Missing name]: {type(e).__name__}: {e}")
                    name = None

                try:
                    url = self.driver.current_url
                except Exception as e:
                    print(f"[Missing URL]: {type(e).__name__}: {e}")
                    url = None

                try:
                    address = self.driver.find_element(By.CSS_SELECTOR, "span.re__pr-short-description.js__pr-address").text
                except Exception as e:
                    print(f"[Missing address]: {type(e).__name__}: {e}")
                    address = None

                try:
                    short_info_items = self.driver.find_elements(By.CSS_SELECTOR, "div.re__pr-short-info-item.js__pr-short-info-item")
                    price_total = short_info_items[0].find_element(By.CSS_SELECTOR, "span.value").text
                except Exception as e:
                    print(f"[Missing price_total]: {type(e).__name__}: {e}")
                    price_total = None

                try:
                    price_m2 = short_info_items[0].find_element(By.CSS_SELECTOR, "span.ext").text
                except Exception as e:
                    print(f"[Missing price_m2]: {type(e).__name__}: {e}")
                    price_m2 = None

                try:
                    area = short_info_items[1].find_element(By.CSS_SELECTOR, "span.value").text
                except Exception as e:
                    print(f"[Missing area]: {type(e).__name__}: {e}")
                    area = None

                # Get iframe and extract coordinates from data-src
                try:
                    # Wait for the map section to appear
                    map_section = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((
                            By.CSS_SELECTOR, "div.re__section.re__pr-map.js__section.js__li-other"
                        ))
                    )

                    # Wait for iframe inside map section
                    iframe = WebDriverWait(map_section, 10).until(
                        EC.presence_of_element_located((
                            By.CSS_SELECTOR, "div.re__section-body.js__section-body iframe"
                        ))
                    )

                    map_data_src = iframe.get_attribute("data-src")
                    # lat, long = None, None
                    # if data_src and "q=" in data_src:
                    #     coords = data_src.split("q=")[1].split("&")[0]
                    #     lat, long = coords.split(",")

                except Exception as e:
                    print(f"[Error extracting coordinates]: {type(e).__name__}: {e}")

                results.append({
                    "name": name,
                    "url": url,
                    "address": address,
                    "price_total": price_total,
                    "price_m2": price_m2,
                    "area": area,
                    "map_data_src": map_data_src
                })

            except Exception as e:
                print(f"[Error scraping {href}]: {type(e).__name__}: {e}")
                continue

            self.driver.quit()
            # break
            
        estates = []
        for item in results:
            estate = RealStateCollected(
                name=item["name"],
                url=item["url"],
                address=item["address"],
                price_total=item["price_total"],
                price_m2=item["price_m2"],
                area=item["area"],
                map_data_src=item["map_data_src"]
            )
            estates.append(estate)

        self.store_estate(estates)

        return estates
    
    def store_estate(self, estates: list[RealStateCollected]):
        db = DBInitializing()
        for estate in estates:
            db.insert_real_estate_collected(estate)

    def convert_to_processed(self, estate: RealStateCollected) -> RealEstateProcessed:
        property_type = ParserUtils.parse_property_type(estate.name)
        street, ward, district = ParserUtils.parse_address_components(estate.address)
        price_total = ParserUtils.parse_price(estate.price_total)
        price_m2 = ParserUtils.parse_price(estate.price_m2)
        area = ParserUtils.parse_area(estate.area)
        long, lat = ParserUtils.parse_coordinates(estate.map_data_src)

        return RealEstateProcessed(
            name=estate.name,
            url=estate.url,
            property_type=property_type,
            street=street,
            district=district,
            ward=ward,
            price_total=price_total,
            price_m2=price_m2,
            area=area,
            long=long,
            lat=lat
        )
    
    def convert_collected_to_processed(self):
        db = DBInitializing()
        db.cursor.execute("SELECT * FROM real_estate_collected")
        rows = db.cursor.fetchall()

        estates_processed = []

        for row in rows:
            estate_collected = RealStateCollected(
                name=row[0],
                url=row[1],
                address=row[2],
                price_total=row[3],
                price_m2=row[4],
                area=row[5],
                map_data_src=row[6]
            )
            estate_processed = self.convert_to_processed(estate_collected)
            estates_processed.append(estate_processed)

        for estate in estates_processed:
            db.insert_real_estate_processed(estate)

        print(f"[Converted and inserted {len(estates_processed)} processed records]")
        db.close()


if __name__ == "__main__":
    url = "https://batdongsan.com.vn/nha-dat-ban-da-nang?vrs=1&sortValue=1"
    crawler = DataCrawling(url)
    data = crawler.crawl()

    for item in data:
        print(item)

    crawler.convert_collected_to_processed()
