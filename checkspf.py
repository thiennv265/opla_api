import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
from io import BytesIO
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from selenium.webdriver.common.by import By
# Nếu file chưa tồn tại, tạo mới
import os

from datetime import datetime

def current_timestamp():
    """
    Trả về chuỗi thời gian hiện tại dạng yyyymmdd_hhmmss
    Ví dụ: 20250914_160512
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
    
file_path = f"spf_results_{current_timestamp()}.xlsx"
fp = st.empty()
if not os.path.exists(file_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ShopeeFood"
    # tạo header
    headers = ["url", "crawl_time", "name_restaurant", "status", "brand_title", "brand_url", 
               "danh_hieu", "danh_muc", "chi_nhanh", "date_time", "notification"]
    ws.append(headers)
    wb.save(file_path)
fp = st.write(f"Resutl in {file_path}")
# Tùy chọn chạy ẩn
chrome_options = Options()
chrome_options.add_argument("--headless")  # bỏ dòng này nếu muốn nhìn thấy Chrome chạy
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-logging")
chrome_options.add_argument("--log-level=3")
service = Service("E:/cd/chromedriver.exe")  # đường dẫn ChromeDriver
driver = webdriver.Chrome(service=service, options=chrome_options)
    
def scrape_shopeefood(url, driver):
    try:
        driver.get(url)
        time.sleep(5)  # hoặc dùng WebDriverWait để chính xác

        soup = BeautifulSoup(driver.page_source, "html.parser")

        data = {}

        # Trạng thái mở cửa lấy từ opentime-status
        status_tag = soup.select_one("div.opentime-status span.stt")
        if status_tag:
            data["status"] = status_tag.get("title")  # có thể "Mở cửa" hoặc "Đã đóng"
        else:
            data["status"] = None


        # Tên nhà hàng
        name_restaurant = soup.select_one("h1.name-restaurant")
        data["name_restaurant"] = name_restaurant.get_text(strip=True) if name_restaurant else None

        # kind-restaurant
        kind_restaurant = soup.select_one("div.kind-restaurant")

        # Khởi tạo mặc định
        data["danh_hieu"] = None
        data["danh_muc"] = None
        data["chi_nhanh"] = None
        data["notification"] = None
        if kind_restaurant:
            parts = [t.strip() for t in kind_restaurant.stripped_strings if t.strip() not in ["-", "–"]]

            # Danh hiệu: thường nằm trong <div class="tag-preferred">
            tag_preferred = kind_restaurant.select_one("div.tag-preferred")
            if tag_preferred:
                data["danh_hieu"] = tag_preferred.get_text(strip=True)

            # Danh mục: text đầu tiên ngoài tag-preferred
            danh_muc_candidates = [p for p in parts if p != data["danh_hieu"] and "Chi nhánh" not in p]
            if danh_muc_candidates:
                data["danh_muc"] = danh_muc_candidates[0]

            # Chi nhánh: lấy từ text của a.link-brand
            chi_nhanh_tag = kind_restaurant.select_one("a.link-brand")
            if chi_nhanh_tag:
                data["chi_nhanh"] = chi_nhanh_tag.get_text(strip=True)
        
        # lấy element modal-body
        # tìm modal-body
        # tìm tất cả modal-content
        modals = soup.select("div.modal-content")

        for modal in modals:
            header = modal.select_one("div.txt-bold.font13 span.txt-red")
            if header and header.text.strip() == "ShopeeFood":
                # lấy text modal-body
                modal_body = modal.select_one("div.modal-body")
                if modal_body:
                    text = ". ".join(modal_body.get_text(strip=True, separator="\n").splitlines())
                    print(text)
                    data["notification"] = text

        # Brand: lấy cả title và href
        brand_tag = soup.select_one("a.link-brand")
        if brand_tag:
            data["brand_title"] = brand_tag.get("title")
            data["brand_url"] = brand_tag.get("href")
        else:
            data["brand_title"] = None
            data["brand_url"] = None
        return data
    except Exception as e:
        print(e)

# --- Giao diện Streamlit ---
st.title("ShopeeFood Scraper 🍕")
link_input = st.text_area("Nhập list link ShopeeFood (mỗi link 1 dòng):")

col1, col2 = st.columns(2)  # chia 2 cột

# Nút download ở cột 1
with col2:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Tải Excel hiện tại",
                data=f,
                file_name=file_path.split("/")[-1],  # chỉ lấy tên file
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

with col1:   
    crawl_clicked = st.button("DO CRAWL", use_container_width=True) 
if crawl_clicked:
    urls = [u.strip() for u in link_input.split("\n") if u.strip()]
    results = []

    progress_placeholder = st.empty()  # hiển thị tiến độ
    current_item = st.empty()          # link đang crawl
    table_placeholder = st.empty()     # bảng kết quả

    total = len(urls)
    overall_start = time.time()

    for idx, url in enumerate(urls, start=1):
        # cập nhật tiến độ
        pro_text = f"⏳ Crawling {idx}/{total} ({round(idx/total*100, 1)}%) - {url}"
        progress_placeholder.write(pro_text)
        print(pro_text)
        # current_item.write(f"🔍 Đang crawl: {url}")

        start = time.time()
        
        try:
            ct = current_timestamp()
            data = scrape_shopeefood(url, driver)
            data["url"] = url
            data["crawl_time"] = round(time.time() - start, 2)
            # lưu 
            row = [
                data.get("url", None),
                data.get("crawl_time", None),
                data.get("name_restaurant", None),
                data.get("status", None),
                data.get("brand_title", None),
                data.get("brand_url", None),
                data.get("danh_hieu", None),
                data.get("danh_muc", None),
                data.get("chi_nhanh", None),
                ct,
                data.get("notification", None)
                
            ]

            wb = openpyxl.load_workbook(file_path)
            ws = wb.active
            ws.append(row)
            wb.save(file_path)

            results.append(data)

            # cập nhật bảng ngay sau mỗi link
            df = pd.DataFrame(results)
            table_placeholder.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error {url}: {e}")

    driver.quit()

    if results:
        st.success(f"✅ Hoàn thành {total}/{total} - Tổng thời gian: {round((time.time()-overall_start)/60,0)}m")

