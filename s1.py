import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
from io import BytesIO
import openpyxl
from datetime import datetime
import os
from webdriver_manager.chrome import ChromeDriverManager

# ==== 🕒 Utility ====
def current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
st.set_page_config(
    page_title=f"M{current_timestamp()}",  # 🏷️ Title hiển thị trên tab
    page_icon="🍜",                        # 📌 Icon trên tab (favicon)
    layout="wide"                          # ✅ Wide mode mặc định
)
FILENAME_PREFIX = "spf_results_1_"  # ✅ đổi khác nhau giữa script 1 và 2
if "file_path" not in st.session_state:
    tt = current_timestamp()
    st.session_state.file_path = f"{FILENAME_PREFIX}{tt}.xlsx"

file_path = st.session_state.file_path
# ==== 📄 File Excel ====

if not os.path.exists(file_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ShopeeFood"
    headers = ["url", "crawl_time", "name_restaurant", "status", "brand_title", "brand_url", 
               "danh_hieu", "danh_muc", "chi_nhanh", "date_time", "notification"]
    ws.append(headers)
    wb.save(file_path)

# ==== ⚙️ Selenium Setup ====
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

# service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# ==== 🕷️ Crawler ====
def scrape_shopeefood(url, driver):
    try:
        driver.get(url)
        time.sleep(4)
        print("Page title:", driver.title)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        data = {}

        status_tag = soup.select_one("div.opentime-status span.stt")
        data["status"] = status_tag.get("title") if status_tag else None

        name_restaurant = soup.select_one("h1.name-restaurant")
        data["name_restaurant"] = name_restaurant.get_text(strip=True) if name_restaurant else None
        data["dt"] = current_timestamp()
        kind_restaurant = soup.select_one("div.kind-restaurant")
        data["danh_hieu"] = data["danh_muc"] = data["chi_nhanh"] = data["notification"] = None

        if kind_restaurant:
            parts = [t.strip() for t in kind_restaurant.stripped_strings if t.strip() not in ["-", "–"]]
            tag_preferred = kind_restaurant.select_one("div.tag-preferred")
            if tag_preferred:
                data["danh_hieu"] = tag_preferred.get_text(strip=True)
            danh_muc_candidates = [p for p in parts if p != data["danh_hieu"] and "Chi nhánh" not in p]
            if danh_muc_candidates:
                data["danh_muc"] = danh_muc_candidates[0]
            chi_nhanh_tag = kind_restaurant.select_one("a.link-brand")
            if chi_nhanh_tag:
                data["chi_nhanh"] = chi_nhanh_tag.get_text(strip=True)

        modals = soup.select("div.modal-content")
        for modal in modals:
            header = modal.select_one("div.txt-bold.font13 span.txt-red")
            if header and header.text.strip() == "ShopeeFood":
                modal_body = modal.select_one("div.modal-body")
                if modal_body:
                    text = ". ".join(modal_body.get_text(strip=True, separator="\n").splitlines())
                    data["notification"] = text

        brand_tag = soup.select_one("a.link-brand")
        if brand_tag:
            data["brand_title"] = brand_tag.get("title")
            data["brand_url"] = brand_tag.get("href")
        else:
            data["brand_title"] = data["brand_url"] = None

        return data
    except Exception as e:
        print(e)
        return {}

# ==== 🌐 Streamlit UI ====
st.title(f"M{current_timestamp()}")
link_input = st.text_area("Nhập list link ShopeeFood (mỗi dòng 1 link):")

col1, col2 = st.columns(2)

with col2:
    if os.path.exists(file_path):  # có nội dung
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Tải Excel kết quả",
                data=f.read(),  # .read() để chắc chắn không cache
                file_name=os.path.basename(file_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )    

with col1:
    crawl_clicked = st.button("DO CRAWL", use_container_width=True)

if crawl_clicked:
    urls = [u.strip() for u in link_input.split("\n") if u.strip()]
    results = []

    progress_placeholder = st.empty()
    table_placeholder = st.empty()
    total = len(urls)
    overall_start = time.time()

    for idx, url in enumerate(urls, start=1):
        pro_text = f"⏳ Crawling {idx}/{total} ({round(idx/total*100, 1)}%) - {url}"
        print(pro_text)
        progress_placeholder.write(pro_text)

        start = time.time()
        try:
            ct = current_timestamp()
            data = scrape_shopeefood(url, driver)
            data["url"] = url
            data["crawl_time"] = round(time.time() - start, 2)

            row = [
                data.get("url"),
                data.get("crawl_time"),
                data.get("name_restaurant"),
                data.get("status"),
                data.get("brand_title"),
                data.get("brand_url"),
                data.get("danh_hieu"),
                data.get("danh_muc"),
                data.get("chi_nhanh"),
                data.get("dt"),
                data.get("notification")
            ]

            wb = openpyxl.load_workbook(file_path)
            ws = wb.active
            ws.append(row)
            wb.save(file_path)

            results.append(data)
            df = pd.DataFrame(results)
            table_placeholder.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error {url}: {e}")

    driver.quit()
    if results:
        st.success(f"✅ Crawl xong {total}/{total} link - {round((time.time()-overall_start)/60, 1)} phút")
