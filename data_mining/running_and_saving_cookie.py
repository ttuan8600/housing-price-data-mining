from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pickle
from config import Config

chrome_options = Options()
chrome_options.add_argument("--start-maximized")

service = Service(Config.CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.get("https://batdongsan.com.vn/")  # hoặc link cụ thể bạn cần
print("🕵️ Hãy hoàn thành xác minh CAPTCHA nếu có và nhấn Enter khi xong...")
input()

# Lưu cookie sau khi đã xác minh
with open("cookies.pkl", "wb") as f:
    pickle.dump(driver.get_cookies(), f)

driver.quit()
