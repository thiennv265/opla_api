import subprocess, io, sys, time, traceback

def install_if_missing(package):
  try:
    __import__(package)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])
# Cần kiểm tra và cài các thư viện ngoài
for pkg in ["requests", "pandas", "fastapi", "cachetools", "urllib3", "openpyxl", "numpy", "simplejson", "asyncio", "rapidfuzz", "aiohttp"]:
  install_if_missing(pkg)
from typing import List
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from rapidfuzz import fuzz, process
import requests
import numpy as np
import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
from fastapi import FastAPI, Query, Response, Request, HTTPException
from datetime import datetime, timezone, timedelta
from threading import Lock
from fastapi.responses import JSONResponse
import simplejson as json
import logging, re, aiohttp, asyncio, random, openpyxl, json
telegram_token = "7069011696:AAHTEO8CmfHKebxAh8TBjMb73wKZt6nbDFg"
app = FastAPI()
cache = {}
lock = Lock()
# Danh sách skip ban đầu (có thể lớn đến 30000)
skips = list(range(0, 30000, 150))

# Biến cờ dừng toàn cục
stop_flag = asyncio.Event()

raw_rows = []
raw_logs = []
MAX_RETRIES = 10
from starlette.exceptions import HTTPException as StarletteHTTPException
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request, exc):
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={"Here the error":"Stupid! 404"})
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Có thể log traceback tại đây
    return JSONResponse(
        status_code=500,
        content={"Here the error": "Internal server error 500"}
    )
    
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"Here the error": "Double check. Please!"}
    )

class CustomFormatter(logging.Formatter):
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    def formatTime(self, record, datefmt=None):
        return f"{self.YELLOW}{get_current_time_str()}{self.RESET}"

# Logger riêng (hoặc dùng uvicorn.access nếu muốn)
logger = logging.getLogger("masked.access")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = CustomFormatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def mask_query_params(url: str, keys=("token", "secrets")) -> str:
    for key in keys:
        pattern = rf"({key}=)([^&\s]+)"
        url = re.sub(pattern, lambda m: m.group(1) + mask_half(m.group(2)), url)
    return url

def mask_half(value: str) -> str:
    if len(value) < 4:
        return "***"
    half = len(value) // 2
    return value[:half] + "*" * (len(value) - half)

def json_size(data):
    json_bytes = json.dumps(data).encode("utf-8")
    size_mb = len(json_bytes) / (1024 * 1024)
    return f"{size_mb:.1f} MB"

def color_status(status_code: int) -> str:
    if 200 <= status_code < 300:
        return f"\033[92m{status_code}\033[0m"  # xanh lá
    elif 300 <= status_code < 400:
        return f"\033[96m{status_code}\033[0m"  # cyan
    elif 400 <= status_code < 500:
        return f"\033[93m{status_code}\033[0m"  # vàng
    else:
        return f"\033[91m{status_code}\033[0m"  # đỏ

@app.middleware("http")
async def log_masked_requests(request: Request, call_next):
    start = time.time()
    # client_ip = request.client.host
    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()
    method = request.method
    url = str(request.url)
    masked_url = mask_query_params(url)
    response = await call_next(request)
    status_code = response.status_code
    duration_ms = time.time() - start
    msg_print = f"\033[91m{client_ip}\033[0m - {method} {masked_url} - {color_status(status_code)} - {duration_ms:.2f}s"
    logger.info(msg_print)
    msg_sendlog = f"{client_ip} - {method} {masked_url} - {status_code} - {duration_ms:.2f}s"
    if status_code != 404:
        send_log(msg_sendlog,"main")
    return response

def total_time(start, stop):
    elapsed = stop - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"

def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y-%m-%d_%H-%M-%S')
  
def appendToRow(myDict: dict, key: str, value: str):
  myDict[key] = value
 
def convert_utc_to_gmt7(dt_str: str) -> str:
    """
    Chuyển chuỗi thời gian ISO UTC (có định dạng: 2025-06-26T05:09:58.660403+00:00)
    sang định dạng tương ứng trong GMT+7.

    Trả về chuỗi ISO format theo GMT+7.
    """
    try:
        # Parse từ chuỗi có timezone UTC
        dt_utc = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        # Chuyển sang GMT+7
        gmt7 = dt_utc.astimezone(timezone(timedelta(hours=7)))
        return str(gmt7)
    except Exception as e:
        print(e)
        raise ValueError(f"Không thể parse thời gian: {e}")

async def dedup_dicts_smart(data: list[dict]) -> list[dict]:
    try:
        # return pd.DataFrame(data).drop_duplicates().to_dict(orient="records")
        df = pd.DataFrame(data).drop_duplicates()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.where(pd.notnull(df), None)
        return df
    except Exception as e:
        traceback.print_exc()
        print(e)
        
async def fetch_worker(worker_id: int, queue: asyncio.Queue, session, stats: dict, token):
    while not queue.empty() and not stop_flag.is_set():
        url, skipp = await queue.get()
        await fetch_url_with_retry(worker_id, url, session, stats, token)

def send_excel_to_telegram(file_bytes: bytes, filename: str, chat_id: str, bot_token: str):
    files = {
        'document': (filename, file_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    }
    data = {
        'chat_id': chat_id,
        'caption': f'📦 File: {filename}'
    }

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    response = requests.post(url, data=data, files=files)
    return response
    
def send_log(msg, sender_name=None, telegram_token = telegram_token):
    if telegram_token:
      try:
          requests.get(
              f"https://api.telegram.org/bot{telegram_token}/sendMessage",
              params={"chat_id": "716085753", "text": "🅾️" + get_current_time_str() + " " + f"{sender_name}\n{msg}" if sender_name else "\n" + msg,"disable_web_page_preview": True}
          )
      except Exception as e:
          print(e)
    else:
      print("No TELE TOKEN ")

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; SM-G996B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.131 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36 Edg/114.0.1823.82",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/113.0.5672.130 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; Pixel 6 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.134 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:114.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Linux; Android 10; SM-A505F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.130 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (iPad; CPU OS 15_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.162 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.91 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.61 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (Windows NT 10.0; ARM64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.102 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:114.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_7_4) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Brave/1.52.129 Chrome/114.0.5735.110 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Samsung Galaxy S23) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.61 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 16_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.61 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"]

BATCH_SIZE = 1000
MAX_WORKERS = 3
MAX_RETRIES = 10
RETRY_DELAY = 10
# --------- BATCH FETCH WITH RETRY ---------
async def fetch_batch(session, url, headers, skip, batch_size, w_num):
    full_url = f"{url}&take={batch_size}&skip={skip}"
    retries = 0

    while retries < MAX_RETRIES:
        try:
            async with session.get(full_url, headers=headers) as response:
                if response.status == 200:
                    r = await response.json()
                    print(f"[W-{w_num}] 🚀 Fetching {full_url} +{len(r)} ({retries+1}/{MAX_RETRIES})")
                    return r
                else:
                    text = await response.text()
                    print(f"[[W-{w_num}]⚠️ Retry {retries+1}] Non-200 ({response.status}) for skip={skip}: {text}")
        except Exception as e:
            print(f"[[W-{w_num}]❌ Retry {retries+1}] Exception for skip={skip}: {e}")

        retries += 1
        await asyncio.sleep(RETRY_DELAY)

    print(f"[W-{w_num}]🚫 Failed after {MAX_RETRIES} retries for skip={skip}")
    return []

# --------- MAIN ASYNC FUNCTION ---------
async def getleads(token: str):
    try:
        start = time.time()
        url_base = "https://api-admin.oplacrm.com/api/public/leads?"
        headers = {
            "Authorization": token,
            "User-Agent": random.choice(user_agents),
            "Connection": "keep-alive",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br"
        }

        sta = get_current_time_str()
        raw_rows = []
        total_bytes = 0
        skip = 0
        should_stop = False

        connector = aiohttp.TCPConnector(limit=MAX_WORKERS)
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while not should_stop:
                tasks = [
                    fetch_batch(session, url_base, headers, skip + i * BATCH_SIZE, BATCH_SIZE, i+1)
                    for i in range(MAX_WORKERS)
                ]
                results = await asyncio.gather(*tasks)

                for batch_data in results:
                    if not batch_data or len(batch_data) < BATCH_SIZE:
                        should_stop = True
                    total_bytes += len(json.dumps(batch_data).encode('utf-8'))

                    for item in batch_data:
                        row = {}
                        for key, value in item.items():
                            if key == "custom_field_lead_values":
                                for i in value:
                                    appendToRow(row, f'{i["custom_field"]["name"]}', i["value"])
                            elif key == "account_name":
                                appendToRow(row, 'store_lead', value)
                            elif key == "name":
                                appendToRow(row, 'contact_name', value)
                            elif key == "phone":
                                appendToRow(row, 'phone', f'0{value["phone"]}')
                            elif key == "id":
                                appendToRow(row, 'lead_id', value)
                            elif key == "created_at":
                                appendToRow(row, 'created_at', value[:10])
                            elif key == "owner":
                                appendToRow(row, 'owner', value['full_name'])
                                appendToRow(row, 'owner_id', value['external_id'])
                        raw_rows.append(row)

                skip += BATCH_SIZE * MAX_WORKERS

        # Dedupe in thread
        dunique = await dedup_dicts_smart(raw_rows)

        sto = get_current_time_str()
        async with asyncio.Lock():
            if len(dunique) > 0:
                if token not in cache:
                    cache[token] = {}
                cache[token]["leads"] = dunique
                cache[token]["updated_leads"] = get_current_time_str()
        stop = time.time()
        msgg = f"   {sta} -> {sto}: {total_bytes / (1024 * 1024):.2f} MB - {len(dunique)} lead records - {total_time(start, stop)}"
        send_log(msgg, "get leads")
        print(msgg)
        return dunique

    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

async def processing_logs(logs_df, current_df):
    try:
        # Xử lý datetime và chuẩn hóa stage
        logs_df['datetime'] = pd.to_datetime(logs_df['datetime'])
        logs_df['stage'] = logs_df['stage'].str.strip()

        # Lấy ngày Chờ duyệt (mới nhất)
        cho_duyet = (
            logs_df[logs_df['stage'] == 'Chờ Duyệt']
            .drop_duplicates('store_id', keep='first')[['store_id', 'datetime']]
            .rename(columns={'datetime': 'logs Ngày Chờ duyệt'})
        )
        cho_duyet['logs Ngày Chờ duyệt'] = cho_duyet['logs Ngày Chờ duyệt'].dt.strftime('%Y-%m-%d')

        # Lấy ngày Phê duyệt (mới nhất từ "Cần điều chỉnh", "Đủ thông tin")
        phe_duyet = (
            logs_df[logs_df['stage'].isin(['Cần điều chỉnh', 'Đủ thông tin'])]
            .drop_duplicates('store_id', keep='first')[['store_id', 'datetime']]
            .rename(columns={'datetime': 'logs Ngày Phê duyệt'})
        )
        phe_duyet['logs Ngày Phê duyệt'] = phe_duyet['logs Ngày Phê duyệt'].dt.strftime('%Y-%m-%d')

        # Gộp logs lại
        logs_summary = cho_duyet.merge(phe_duyet, on='store_id', how='outer')

        # 🔧 Dùng .copy() để tránh SettingWithCopyWarning
        current_df = current_df.copy()

        # Chuyển ngày trong current về dạng chuỗi (để so sánh và export)
        current_df['store_Ngày Chờ duyệt'] = pd.to_datetime(
            current_df['store_Ngày Chờ duyệt'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')

        current_df['store_Ngày Phê duyệt'] = pd.to_datetime(
            current_df['store_Ngày Phê duyệt'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')

        # Chọn các cột cần từ current và gộp với logs
        merged = current_df[[
            'store_id', 'store_short_id',
            'store_Ngày Chờ duyệt', 'store_Ngày Phê duyệt'
        ]].merge(logs_summary, on='store_id', how='left')

        # So sánh ngày
        merged['Check Chờ Duyệt'] = merged['store_Ngày Chờ duyệt'] == merged['logs Ngày Chờ duyệt']
        merged['Check Phê Duyệt'] = merged['store_Ngày Phê duyệt'] == merged['logs Ngày Phê duyệt']

        # Tạo cột correct nếu lệch
        merged['correct Ngày Chờ duyệt'] = merged.apply(
            lambda row: row['logs Ngày Chờ duyệt']
            if pd.notna(row['logs Ngày Chờ duyệt']) and row['logs Ngày Chờ duyệt'] != row['store_Ngày Chờ duyệt']
            else '',
            axis=1
        )
        merged['correct Ngày Phê duyệt'] = merged.apply(
            lambda row: row['logs Ngày Phê duyệt']
            if pd.notna(row['logs Ngày Phê duyệt']) and row['logs Ngày Phê duyệt'] != row['store_Ngày Phê duyệt']
            else '',
            axis=1
        )

        # Kết quả cuối cùng
        final_result = merged[[
            'store_id', 'store_short_id',
            'logs Ngày Chờ duyệt', 'store_Ngày Chờ duyệt', 'Check Chờ Duyệt', 'correct Ngày Chờ duyệt',
            'logs Ngày Phê duyệt', 'store_Ngày Phê duyệt', 'Check Phê Duyệt', 'correct Ngày Phê duyệt'
        ]]

        return final_result

    except Exception as e:
        traceback.print_exc()
        send_log(f"Lỗi {e}", "main")
        return None
        
async def fetch_url_with_retry(worker_id: int, url: str, session, stats: dict, token):
    headers = {"Authorization": token, "User-Agent": random.choice(user_agents), "Connection": "keep-alive", "Accept":"*/*", "Accept-Encoding":"gzip, deflate, br"}
    retries = 0
    start = time.time()
    while retries < MAX_RETRIES and not stop_flag.is_set():
        print(f"[W-{worker_id}] 🚀 Fetching {url} (try {retries + 1}/{MAX_RETRIES})")
        await asyncio.sleep(5)

        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 401:
                    print(f"[W-{worker_id}] 🔐 Token sai tại {url}")
                    stop_flag.set()
                    return

                # Retry được với lỗi 5xx
                if 500 <= response.status < 600:
                    print(f"[W-{worker_id}] 🔁 HTTP {response.status} - thử lại {url}")
                    retries += 1
                    await asyncio.sleep(10)
                    continue

                # Không retry với lỗi khác (404, 403...)
                if response.status != 200:
                    print(f"[W-{worker_id}] ❌ HTTP {response.status} tại {url} - không retry")
                    break

                data = await response.json()
                sources = data
                if not isinstance(data, list):
                    print(f"[W-{worker_id}] ⚠️ Dữ liệu không phải list: {data}")
                    break

                if not data:
                    print(f"[W-{worker_id}] 🛑 Dừng lại: {url} trả về rỗng")
                    stop_flag.set()
                    return

                # Nếu thành công
                count = len(data)
                size = len(json.dumps(data).encode("utf-8"))
                stats["total_items"] += count
                stats["total_bytes"] += size
                excluded_keys = ["weight","area","google_map_address","description","stage_compact","amount", "invoice", "invoices", "opportunity_process",
                               "opportunity_process_stage_id","tax_inclusive_amount","forecast","opportunities_joint","opportunities_products","date_closed",
                               "locked","date_closed_actual","discussions","is_parent","source","opportunity_status","project_type","opportunities_contacts",
                               "Error","notes","parent_opportunity_id","parent_opportunity","opportunities_children","opportunity_type_id","activities","date_open"]
                special_keys = ["custom_field_opportunity_values","opportunity_process_stage","owner","users_opportunities","accounts_opportunities","created_at","stage_logs"]
                for index, item in enumerate(sources):
                    row = {}
                    for key, value in item.items():
                      # print(key)
                      if key not in excluded_keys:
                        if key not in special_keys:
                          appendToRow(row, f'store_{key}',value)
                        elif key == "created_at":
                          appendToRow(row, f'store_{key}',value[:10])
                        elif key == "stage_logs":
                          # print(len(value))
                          for i in value:
                            row_log = {}
                            appendToRow(row_log, f'store_id',item["id"])
                            appendToRow(row_log, f'store_short_id',item["short_id"])
                            appendToRow(row_log, f'creator',i["creator"]["email"])
                            appendToRow(row_log, f'datetime', convert_utc_to_gmt7(i["created_at"]))
                            appendToRow(row_log, f'stage',i["new_stage"])
                            raw_logs.append(row_log)
                        elif key =="custom_field_opportunity_values":
                          for i in value:
                            # print(i["custom_field"])
                            if i["custom_field"]["name"] not in ["20. ADO","23. Giá món trung bình *","18. Vĩ độ","19. Kinh độ","21. Quận *","Quận (cũ)",
                                                                 "24. Phần mềm bán hàng *","23. Khung giờ hoạt động 2","25. Ghi Chú Riêng"]:
                                appendToRow(row, f'store_{i["custom_field"]["name"]}',i["value"])
                        elif key == "opportunity_process_stage":
                          appendToRow(row, f'store_{key}',value["opportunity_stage"]["name"])
                        elif key == "owner":
                          appendToRow(row, f'store_{key}',value["email"])
                        elif key == "users_opportunities":
                          appendToRow(row, f'store_{key}',value[0]["user"]["email"])
                        elif key == "accounts_opportunities":
                          for k, v in value[0]["account"].items():
                            included_keys = ["id","name","short_id","account_type","owner","custom_field_account_values","tax_identification_number"]
                            if k in included_keys:
                              if k == "account_type":
                                appendToRow(row, f'mex_{k}',v)
                              elif k == "tax_identification_number":
                                appendToRow(row, f'mex_tax_id',v)
                              elif k == "owner":
                                appendToRow(row, f'mex_{k}',v["email"])
                              elif k == "custom_field_account_values":
                                for k1 in v:
                                  excluded_keys_mex = ["34. Link ảnh","21. Phần mềm bán hàng *","28. Ghi chú trạng thái","27. Trạng thái ký kết *","29. Lý do Không Hợp Lệ",
                                                       "24. Phần mềm bán hàng *","23. Khung giờ hoạt động 2","25. Ghi Chú Riêng","20. ADO","23. Giá món trung bình *","24. Link gian hàng SPF",
                                                      "25. Link gian hàng GF", "27. Link gian hàng Google Review","33. Link gian hàng BeF","account_type"]
                                  n = k1["custom_field"]["name"]
                                  vl = k1["value"]
                                  if k1["custom_field"]["master_data_custom_fields"]:
                                    for i in k1["custom_field"]["master_data_custom_fields"]:
                                      if i["id"] == vl:
                                        vl = i["value"]
                                  if n not in excluded_keys_mex:
                                    appendToRow(row, f'mex_{n}',vl)
                              else:
                                appendToRow(row, f'mex_{k}',v)
                        else:
                            appendToRow(row, f'store_{key}',value)
                    raw_rows.append(row)
                stop = time.time()
                print(f"[W-{worker_id}] ✅ {count} item từ {url} - {total_time(start, stop)}")
                return  # kết thúc thành công

        except Exception as e:
            stop = time.time()
            print(f"[W-{worker_id}] ❗ Lỗi: {e} tại {url} - {total_time(start, stop)}")
            retries += 1

    print(f"[W-{worker_id}] ❌ Bỏ qua {url} sau {MAX_RETRIES} lần thử")

async def fetch_opportunities_queue(token):
    sta = get_current_time_str()
    start = time.time()
    # Tạo danh sách URL
    urls = [
        (f"https://api-admin.oplacrm.com/api/public/opportunities?take=150&skip={skipp - 30 if skipp > 0 else 0}", skipp)
        for skipp in skips
    ]
    stats = {"total_items":0, "total_bytes":0}
    queue = asyncio.Queue()
    for item in urls:
        await queue.put(item)
    connector = aiohttp.TCPConnector(limit=3)
    timeout = aiohttp.ClientTimeout(total= 240, connect =60, sock_read= 240)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Tạo 3 worker chạy song song
        tasks = [
            fetch_worker(i + 1, queue, session, stats, token)
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
    store_records = await dedup_dicts_smart(raw_rows)
    store_logs = await dedup_dicts_smart(raw_logs)
    sto = get_current_time_str()
    stop = time.time()
    stop_flag.clear()
    msgg = f"   {sta} -> {sto}: {stats['total_bytes'] / (1024 * 1024):.2f} MB - {len(store_records)} store records + {len(store_logs)} log records - {total_time(start, stop)}"   
    print (msgg)
    send_log(msgg,"main")
    async with asyncio.Lock():
      if len(store_records) > 0:
        if token not in cache: cache[token] = {}
        cache[token]["stores"] = store_records
        cache[token]["stage_logs"] = store_logs
        cache[token]["updated_stores_and_stage_logs"] = get_current_time_str()
    return [store_records, store_logs]

def convert_all_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép toàn bộ các cột trong DataFrame về kiểu str, xử lý cả NaN.
    """
    return df.applymap(lambda x: '' if pd.isna(x) else str(x))
    
async def tele_logs(df, df_current, token):
    try:
        expected_cols = ["store_id", "store_short_id", "store_Ngày Chờ duyệt", "store_Ngày Phê duyệt"]
        available_cols = [col for col in expected_cols if col in df_current.columns]
        df_current = df_current[available_cols]

        df_processing = await processing_logs(df, df_current)

        df_date_update_result = await update_dates_from_df(df_processing, token)

        # Export = 1: xuất file Excel và gửi về Telegram
        file_bytes = io.BytesIO()
        with pd.ExcelWriter(file_bytes, engine="openpyxl") as writer:
            convert_all_columns_to_str(df).to_excel (writer, index=False, sheet_name="logs")
            convert_all_columns_to_str(df_current).to_excel(writer, index=False, sheet_name="current")
            if df_processing is not None and not df_processing.empty:
                convert_all_columns_to_str(df_processing).to_excel(writer, index=False, sheet_name="processing")
            if df_date_update_result is not None and not df_date_update_result.empty:
                convert_all_columns_to_str(df_date_update_result).to_excel(writer, index=False, sheet_name="update_result")
                
        file_bytes.seek(0)

        send_excel_to_telegram(
            file_bytes=file_bytes,
            filename=f"api_logs_{cache[token]['updated_stores_and_stage_logs']}.xlsx",
            chat_id="716085753",
            bot_token=telegram_token
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


async def tele_stores(df, token):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            convert_all_columns_to_str(df).to_excel(writer, index=False, sheet_name="stores")

        output.seek(0)
        file_content = output.read()
        file_bytes = io.BytesIO(file_content)
        file_bytes.seek(0)

        send_excel_to_telegram(
            file_bytes=file_bytes,
            filename=f"api_stores_{cache[token]['updated_stores_and_stage_logs']}.xlsx",
            chat_id="716085753",
            bot_token=telegram_token
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

async def put_opportunity(session, url, payload, headers):
    try:
        async with session.put(url, json=payload, headers=headers) as response:
            text = await response.text()
            return response.status, text
    except Exception as e:
        return -1, str(e)

async def update_dates_from_df(df: pd.DataFrame, token: str) -> pd.DataFrame:
    df = df.copy()
    df["PUT Chờ duyệt"] = ""
    df["PUT Phê duyệt"] = ""

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    total_put = 0
    total_success = 0
    total_failed = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        task_refs = []
        
        for idx, row in df.iterrows():
            store_id = row["store_id"]
            put_url = f"https://api-admin.oplacrm.com/api/public/opportunities/{store_id}"

            if row.get("Check Chờ Duyệt") is False and pd.notna(row.get("logs Ngày Chờ duyệt")) and str(row.get("logs Ngày Chờ duyệt")).strip() != "":
                payload = {
                    "custom_fields": [{
                        "custom_field_id": "Ngày Chờ duyệt",
                        "value": row.get("logs Ngày Chờ duyệt")
                    }]
                }
                tasks.append(put_opportunity(session, put_url, payload, headers))
                task_refs.append((idx, "PUT Chờ duyệt"))

            if row.get("Check Phê Duyệt") is False and pd.notna(row.get("logs Ngày Phê duyệt")) and str(row.get("logs Ngày Phê duyệt")).strip() != "":
                payload = {
                    "custom_fields": [{
                        "custom_field_id": "Ngày Phê duyệt",
                        "value": row.get("logs Ngày Phê duyệt")
                    }]
                }
                tasks.append(put_opportunity(session, put_url, payload, headers))
                task_refs.append((idx, "PUT Phê duyệt"))

        total_put = len(tasks)
        if total_put == 0:
            msgg = "✅ Không có bản ghi nào cần PUT"
            print(msgg)
            send_log(msgg, "update log")
            return df
        results = await asyncio.gather(*tasks)

        failed_tasks = []
        for (idx, col), (status, msg) in zip(task_refs, results):
            df.at[idx, col] = f"{status} - {msg[:100]}"
            if status == 200:
                total_success += 1
            else:
                total_failed += 1
                failed_tasks.append((idx, col, df.at[idx, "store_id"], msg))

        # Retry các PUT lỗi
        MAX_RETRIES = 3
        for retry in range(1, MAX_RETRIES + 1):
            if not failed_tasks:
                break
            print(f"🔁 Retry lần {retry} với {len(failed_tasks)} PUT lỗi")

            retry_tasks = []
            retry_refs = []

            for idx, col, store_id, _ in failed_tasks:
                put_url = f"https://api-admin.oplacrm.com/api/public/opportunities/{store_id}"

                if col == "PUT Chờ duyệt":
                    payload = {
                        "custom_fields": [{
                            "custom_field_id": "Ngày Chờ duyệt",
                            "value": df.at[idx, "logs Ngày Chờ duyệt"]
                        }]
                    }
                else:
                    payload = {
                        "custom_fields": [{
                            "custom_field_id": "Ngày Phê duyệt",
                            "value": df.at[idx, "logs Ngày Phê duyệt"]
                        }]
                    }

                retry_tasks.append(put_opportunity(session, put_url, payload, headers))
                retry_refs.append((idx, col))

            results = await asyncio.gather(*retry_tasks)
            new_failed_tasks = []

            for (idx, col), (status, msg) in zip(retry_refs, results):
                df.at[idx, col] += f" | Retry {retry}: {status}"
                if status == 200:
                    total_success += 1
                    total_failed -= 1  # giảm thất bại vì đã thành công
                else:
                    new_failed_tasks.append((idx, col, df.at[idx, "store_id"], msg))

            failed_tasks = new_failed_tasks

    end_time = time.time()
    elapsed = end_time - start_time
    msgg = f"Tổng PUT: {total_put} - 🟢: {total_success} - 🔴: {total_failed} - {elapsed:.2f}s\n"
    print(msgg)
    send_log(msgg,"update log")
    return df

@app.get("/")
async def home():
  return {"hello":":)"}

@app.get("/opla/")
async def api_opla(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    export: int = Query(None)
):
    try:
        if secrets == 'chucm@ym@n8686':
            if cache.get(token, {}).get("stores") is None:
                await fetch_opportunities_queue(token)
            df = pd.DataFrame(cache[token]["stores"])
            if limit:
                df = df.iloc[:limit]
            if fields:
                valid_fields = [col for col in fields if col in df.columns]
                df = df[valid_fields]
            if not export or export != 1:
                safe_json = json.dumps(
                    df.to_dict(orient="records"),
                    ignore_nan=True
                )
                return Response(content=safe_json, media_type="application/json")
            elif export == 1:                   
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name=cache[token]["updated_stores_and_stage_logs"])

                output.seek(0)
                file_content = output.read()
                
                file_bytes = io.BytesIO(file_content)
                file_bytes.seek(0)
                # Gửi tới Telegram
                send_excel_to_telegram(
                    file_bytes= file_bytes,
                    filename=f"api_store_{cache[token]['updated_stores_and_stage_logs']}.xlsx",
                    chat_id="716085753",
                    bot_token=telegram_token
                )

                # Trả response
                headers = {
                    "Content-Disposition": f"attachment; filename=api_store_{cache[token]['updated_stores_and_stage_logs']}.xlsx"
                }
                return Response(
                    content=file_content,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=headers
                )
        else:
            return {"Lỗi": "Sai secrets :("}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
        
        
@app.get("/leads/")
async def api_lead(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    export: int = Query(None)
):
    try:
        if secrets != 'chucm@ym@n8686':
            return {}

        if cache.get(token, {}).get("leads") is None:
            # Chuyển getleads thành async nếu có I/O
            data_leads = await getleads(token)

        df = cache[token]["leads"]
        if fields:
            df = df[[col for col in fields if col in df.columns]]
        if limit:
            df = df.iloc[:limit]

        if not export or export != 1:
                safe_json = json.dumps(
                    df.to_dict(orient="records"),
                    ignore_nan=True
                )

        elif export == 1:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name=cache[token]["updated_leads"])
            output.seek(0)
            file_content = output.read()
            file_bytes = io.BytesIO(file_content)
            file_bytes.seek(0)

            # Giả sử hàm này async
            send_excel_to_telegram(
                file_bytes=file_bytes,
                filename=f"api_leads_{cache[token]['updated_leads']}.xlsx",
                chat_id="716085753",
                bot_token=telegram_token
            )

            headers = {
                "Content-Disposition": f"attachment; filename=api_leads_{cache[token]['updated_leads']}.xlsx"
            }
            return Response(
                content=file_content,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers=headers
            )

    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@app.get("/f5/")
async def api_f5(token: str = Query(...), secrets: str = Query(...)):
    if secrets != 'chucm@ym@n8686':
        return {"Lỗi": "Sai secrets :("}

    if cache.get(token, {}).get("data") is not None:
        del cache[token]

    await fetch_opportunities_queue(token)  # async function
    await getleads(token)
    await tele_logs(cache[token]["stage_logs"], cache[token]["stores"], token)
    await tele_stores(cache[token]["stores"], token)
    send_log("Refreshed!", "f5")
    return {'result': 'OK :)'}


@app.get("/f5/clear-all")
async def api_clear(secrets: str = Query(...)):
    if secrets != 'chucm@ym@n8686':
        return {"Lỗi": "Sai secrets :("}
    
    global cache
    cache = {}
    send_log("Empty!", "clear all")
    return {'result': 'OK :)'}


@app.get("/updated/")
async def last_update(token: str = Query(...)):
    try:
        return {
            "updated_stores_and_stage_logs": cache.get(token, {}).get("updated_stores_and_stage_logs"),
            "updated_leads": cache.get(token, {}).get("updated_leads")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.get("/logs/")
async def api_logs(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    export: int = Query(None)
):
    try:
        if secrets != 'chucm@ym@n8686':
            return {"Lỗi": "Sai secrets :("}

        if cache.get(token, {}).get("stage_logs") is None:
            await fetch_opportunities_queue(token)  # đảm bảo getdata là async

        df = pd.DataFrame(cache[token]["stage_logs"])
        df_current = pd.DataFrame(cache[token]["stores"])

        expected_cols = [
            "store_id",
            "store_short_id",
            "store_Ngày Chờ duyệt",
            "store_Ngày Phê duyệt"
        ]
        available_cols = [col for col in expected_cols if col in df_current.columns]
        df_current = df_current[available_cols]

        # xử lý đồng bộ -> nên để chạy trong thread pool
        df_processing = await processing_logs(df, df_current)
        
        df_date_update_result = await update_dates_from_df(df_processing, token)

        if limit:
            df = df.iloc[:limit]

        if fields:
            valid_fields = [col for col in fields if col in df.columns]
            df = df[valid_fields]

        if not export or export != 1:
            return {"Thông báo": "Không xem được :)"}

        # Export = 1: xuất file Excel và gửi về Telegram
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            convert_all_columns_to_str(df).to_excel (writer, index=False, sheet_name="logs")
            convert_all_columns_to_str(df_current).to_excel(writer, index=False, sheet_name="current")
            if df_processing is not None and not df_processing.empty:
                convert_all_columns_to_str(df_processing).to_excel(writer, index=False, sheet_name="processing")
            if df_date_update_result is not None and not df_date_update_result.empty:
                convert_all_columns_to_str(df_date_update_result).to_excel(writer, index=False, sheet_name="update_result")

        output.seek(0)
        file_content = output.read()
        file_bytes = io.BytesIO(file_content)
        file_bytes.seek(0)

        # Gửi file tới Telegram (async)
        send_excel_to_telegram(
            file_bytes=file_bytes,
            filename=f"api_logs_{cache[token]['updated_stores_and_stage_logs']}.xlsx",
            chat_id="716085753",
            bot_token=telegram_token
        )

        headers = {
            "Content-Disposition": f"attachment; filename=api_logs_{cache[token]['updated_stores_and_stage_logs']}.xlsx"
        }

        return Response(
            content=file_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )

    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
