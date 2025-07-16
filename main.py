import subprocess, io, sys, time, traceback

def install_if_missing(package):
  try:
    __import__(package)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])
# Cần kiểm tra và cài các thư viện ngoài
for pkg in ["requests", "pandas", "fastapi", "cachetools", "urllib3", "openpyxl", "numpy", "simplejson", "openpyxl", "rapidfuzz", "tqdm"]:
  install_if_missing(pkg)
from typing import List
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import openpyxl
from rapidfuzz import fuzz, process
import requests
import json
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from fastapi import FastAPI, Query, Response, Request, HTTPException
from datetime import datetime, timezone, timedelta
from threading import Lock
from fastapi.responses import JSONResponse
import simplejson as json
import logging
import re

from starlette.exceptions import HTTPException as StarletteHTTPException
telegram_token = "7069011696:AAHTEO8CmfHKebxAh8TBjMb73wKZt6nbDFg"
app = FastAPI(docs_url = "/docs/guide", redoc_url = None, openapi_url="/openapi.json")
cache = {}
lock = Lock()

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
    
# Lấy giờ GMT+7
def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y-%m-%d_%H-%M-%S')

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
    send_log(msg_sendlog,"main")
    return response

def dedup_dicts_smart(data: list[dict]) -> list[dict]:
    try:
        # return pd.DataFrame(data).drop_duplicates().to_dict(orient="records")
        df = pd.DataFrame(data).drop_duplicates()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.where(pd.notnull(df), None)
        return df
    except Exception as e:
        print(e)
        # return dedup_large_dict_list(data)
      
def appendToRow(myDict: dict, key: str, value: str):
  myDict[key] = value
    
def getdata(token: str):
    try:
        sta = get_current_time_str()
        raw_rows = []
        raw_logs = []
        total_bytes = 0
        for skipp in range(0,30001,150):
            url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=170&skip={skipp - 20 if skipp > 0 else 0}"
            # url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=10"
            headers = {"Authorization": token}
            response = requests.get(url, headers=headers, verify = False)
            if response.status_code == 200:
                sources = response.json()
                size = len(json.dumps(sources).encode('utf-8'))  # tính size theo byte
                total_bytes += size
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
                            print(i["custom_field"])
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
            else:
              return f'Error: {response.status_code} {response.text}'
            if len(sources) < 170: break
        store_records = dedup_dicts_smart(raw_rows)
        store_logs = dedup_dicts_smart(raw_logs)
        sto = get_current_time_str()
        msgg = f"   {sta} -> {sto}: {total_bytes / (1024 * 1024):.2f} MB - {len(store_records)} store records + {len(store_logs)} log records"
        print (msgg)
        send_log(msgg,"main")
        return [store_records, store_logs]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

def getleads(token: str):
    try:
        sta = get_current_time_str()
        raw_rows = []
        total_bytes = 0
        url = f"https://api-admin.oplacrm.com/api/public/leads?take=100000"
        headers = {
            "Authorization": token
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            sources = response.json()
            size = len(json.dumps(sources).encode('utf-8'))  # tính size theo byte
            total_bytes += size
            included_keys = ["account_name","name","id","created_at","custom_field_lead_values","owner","phone"]
            for index, item in enumerate(sources):
                row = {}
                for key, value in item.items():
                    if key =="custom_field_lead_values":
                        for i in value:
                            appendToRow(row, f'{i["custom_field"]["name"]}',i["value"])
                    elif key == "account_name":
                        # print(value)
                        appendToRow(row, 'store_lead',value)
                    elif key == "name":
                        appendToRow(row, 'contact_name',value)
                    elif key == "phone":
                        appendToRow(row, 'phone',f'0{value["phone"]}')
                    elif key == "id":
                        appendToRow(row, 'lead_id',value)
                    elif key == "created_at":
                        appendToRow(row, 'created_at',value[:10])
                    elif key == "owner":
                        appendToRow(row, 'owner',value['full_name'])
                        appendToRow(row, 'owner_id',value['external_id'])
                raw_rows.append(row)
        else:
            return f'Error: {response.status_code} {response.text}'
        dunique = dedup_dicts_smart(raw_rows)
        sto = get_current_time_str()
        msgg = f"   {sta} -> {sto}: {total_bytes / (1024 * 1024):.2f} MB - {len(dunique)} lead records"
        # print (msgg)
        send_log(msgg,"main")
        return dunique
    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

def convert_all_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép toàn bộ các cột trong DataFrame về kiểu str, xử lý cả NaN.
    """
    return df.applymap(lambda x: '' if pd.isna(x) else str(x))
    
def processing_logs(logs_df, current_df):
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
        cho_duyet['logs Ngày Chờ duyệt'] = cho_duyet['logs Ngày Chờ duyệt'].dt.strftime('%Y-%m-%d')  # -> chuỗi

        # Lấy ngày Phê duyệt (mới nhất từ "Cần điều chỉnh", "Đủ thông tin")
        phe_duyet = (
            logs_df[logs_df['stage'].isin(['Cần điều chỉnh', 'Đủ thông tin'])]
            .drop_duplicates('store_id', keep='first')[['store_id', 'datetime']]
            .rename(columns={'datetime': 'logs Ngày Phê duyệt'})
        )
        phe_duyet['logs Ngày Phê duyệt'] = phe_duyet['logs Ngày Phê duyệt'].dt.strftime('%Y-%m-%d')  # -> chuỗi

        # Gộp logs lại
        logs_summary = cho_duyet.merge(phe_duyet, on='store_id', how='outer')

        # Chuyển ngày trong current về dạng chuỗi (để so sánh và export)
        current_df['store_Ngày Chờ duyệt'] = pd.to_datetime(current_df['store_Ngày Chờ duyệt']).dt.strftime('%Y-%m-%d')
        current_df['store_Ngày Phê duyệt'] = pd.to_datetime(current_df['store_Ngày Phê duyệt']).dt.strftime('%Y-%m-%d')

        # Chọn các cột cần từ current và gộp với logs
        merged = current_df[['store_id', 'store_short_id', 'store_Ngày Chờ duyệt', 'store_Ngày Phê duyệt']].merge(
            logs_summary, on='store_id', how='left'
        )

        # So sánh ngày
        merged['Check Chờ Duyệt'] = merged['store_Ngày Chờ duyệt'] == merged['logs Ngày Chờ duyệt']
        merged['Check Phê Duyệt'] = merged['store_Ngày Phê duyệt'] == merged['logs Ngày Phê duyệt']

        # Tạo cột correct nếu lệch
        merged['correct Ngày Chờ duyệt'] = merged.apply(
            lambda row: row['logs Ngày Chờ duyệt'] if pd.notna(row['logs Ngày Chờ duyệt']) and row['logs Ngày Chờ duyệt'] != row['store_Ngày Chờ duyệt'] else '',
            axis=1
        )
        merged['correct Ngày Phê duyệt'] = merged.apply(
            lambda row: row['logs Ngày Phê duyệt'] if pd.notna(row['logs Ngày Phê duyệt']) and row['logs Ngày Phê duyệt'] != row['store_Ngày Phê duyệt'] else '',
            axis=1
        )

        # Kết quả cuối cùng
        final_result = merged[[
            'store_id', 'store_short_id',
            'logs Ngày Chờ duyệt', 'store_Ngày Chờ duyệt', 'Check Chờ Duyệt', 'correct Ngày Chờ duyệt',
            'logs Ngày Phê duyệt', 'store_Ngày Phê duyệt', 'Check Phê Duyệt', 'correct Ngày Phê duyệt'
        ]]
        return final_result
        # # Xuất file Excel
        # final_result.to_excel("ket_qua_so_sanh.xlsx", index=False)

        # # In thử kết quả
        # print(final_result)
    except Exception as e:
        traceback.print_exc()
        send_log(f"Lỗi {e}", "main")
        return None
@app.get("/")
def home():
  return {"hello":":)"}

@app.get("/opla/")
def api_opla(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    export: int = Query(None)
):
    try:
        if secrets == 'chucm@ym@n8686':
            if cache.get(token, {}).get("stores") is None:
                data_stores = getdata(token)
                with lock:
                    if len(data_stores[0]) > 0:
                      if token not in cache: cache[token] = {}
                      cache[token]["stores"] = data_stores[0]
                      cache[token]["stage_logs"] = data_stores[1]
                      cache[token]["updated_stores_and_stage_logs"] = get_current_time_str()
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

@app.get("/logs/")
def api_logs(token: str = Query(...),secrets: str = Query(...),fields: List[str] = Query(None),limit: int = Query(None),export: int = Query(None)):
    try:
        if secrets == 'chucm@ym@n8686':
            if cache.get(token, {}).get("stage_logs") is None:
                data_stores = getdata(token)
                with lock:
                    if len(data_stores[1]) > 0:
                      if token not in cache: cache[token] = {}
                      cache[token]["stores"] = data_stores[0]
                      cache[token]["stage_logs"] = data_stores[1]
                      cache[token]["updated_stores_and_stage_logs"] = get_current_time_str()

            df = pd.DataFrame(cache[token]["stage_logs"])
            df_current = pd.DataFrame(cache[token]["stores"])
            expected_cols = ["store_id","store_short_id","store_Ngày Chờ duyệt","store_Ngày Phê duyệt"]
            available_cols = [col for col in expected_cols if col in df_current.columns]
            df_current = df_current[available_cols]
            df_processing = processing_logs(df,df_current)
            # print(df)
            # print(df_current)
            if limit:
                df = df.iloc[:limit]
            if fields:
                valid_fields = [col for col in fields if col in df.columns]
                df = df[valid_fields]

            if not export or export != 1:
                return {"Thông báo":"Không xem được :)"}
            elif export == 1:                   
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    convert_all_columns_to_str(df).to_excel(writer, index=False, sheet_name="logs")
                    convert_all_columns_to_str(df_current).to_excel(writer, index=False, sheet_name="current")
                    if df_processing:
                        convert_all_columns_to_str(df_processing).to_excel(writer, index=False, sheet_name="processing")

                output.seek(0)
                file_content = output.read()
                file_bytes = io.BytesIO(file_content)
                file_bytes.seek(0)
                # Gửi tới Telegram
                send_excel_to_telegram(
                    file_bytes= file_bytes,
                    filename=f"api_logs_{cache[token]['updated_stores_and_stage_logs']}.xlsx",
                    chat_id="716085753",
                    bot_token=telegram_token
                )

                # Trả response
                headers = {
                    "Content-Disposition": f"attachment; filename=api_logs_{cache[token]['updated_stores_and_stage_logs']}.xlsx"
                }
                return Response(
                    content=file_content,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=headers
                )
        return {"Lỗi": "Sai secrets :("}
    except Exception as e:
      print(traceback.print_exc())
      raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
        
@app.get("/leads/")
def api_lead(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    export: int = Query(None)
):
    try:
        if secrets != 'chucm@ym@n8686': return {}
        if cache.get(token, {}).get("leads") is None:
            data_leads = getleads(token)
            with lock:
                if len(data_leads) > 0:
                  if token not in cache: cache[token] = {}
                  cache[token]["leads"] = data_leads,
                  cache[token]["updated_leads"] = get_current_time_str()
        df = pd.DataFrame(cache[token]["leads"])
        if fields: df = df[[col for col in fields if col in df.columns]]
        if limit: df = df.iloc[:limit]
        if not export or export != 1:
            return Response(
                content=json.dumps(df.to_dict(orient="records"), ignore_nan=True),
                media_type="application/json"
            )
        elif export == 1:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name=cache[token]["updated_leads"])

            output.seek(0)
            file_content = output.read()

            file_bytes = io.BytesIO(file_content)
            file_bytes.seek(0)
            # Gửi tới Telegram
            send_excel_to_telegram(
                file_bytes= file_bytes,
                filename=f"api_leads_{cache[token]['updated_leads']}.xlsx",
                chat_id="716085753",
                bot_token=telegram_token
            )

            # Trả response
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
def api_f5(token: str = Query(...), secrets: str = Query(...)):
  if secrets == 'chucm@ym@n8686':
    if cache.get(token, {}).get("data") is not None:
        del cache[token]
    data_stores = getdata(token)
    with lock:
        if len(data_stores[0]) > 0:
          if token not in cache: cache[token] = {}
          cache[token]["stores"] = data_stores[0]
          cache[token]["stage_logs"] = data_stores[1]
          cache[token]["updated_stores_and_stage_logs"] = get_current_time_str()
    send_log("Refreshed!", "main")
    return {'result': 'OK :)'}
  else:
    return {"Lỗi": "Sai secrets :("}
    
@app.get("/f5/clear-all")
def api_clear(secrets: str = Query(...)):
  if secrets == 'chucm@ym@n8686':
    cache = {}
    send_log("Empty!", "main")
    return {'result': 'OK :)'}
  else:
    return {"Lỗi": "Sai secrets :("}

@app.get("/updated/")
def last_update(token: str = Query(...)):
  try:
    return {
    "updated_stores_and_stage_logs": cache.get(token, {}).get("updated_stores_and_stage_logs"),
    "updated_leads": cache.get(token, {}).get("updated_leads")}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
