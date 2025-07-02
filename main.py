import subprocess
import io
import sys, time

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
from tqdm import tqdm
import requests
import json
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from fastapi import FastAPI, Query, Response, Request, HTTPException
from datetime import datetime, timezone, timedelta
from cachetools import TTLCache
from threading import Lock
from fastapi.responses import JSONResponse
import simplejson as json
import logging
import re
app = FastAPI(docs_url = None, redoc_url = None, openapi_url = None)
cache = TTLCache(maxsize=1000, ttl=3000)  # 2 tiếng
lock = Lock()

# Lấy giờ GMT+7
def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y%m%d_%H%M%S')
  
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

def analyze_duplicates(df, store_col="store_name", addr_col="store_6. Địa chỉ gian hàng *", threshold=99):
    df = df.copy()
    df[store_col] = df[store_col].astype(str)
    df[addr_col] = df[addr_col].astype(str)

    so_luong_trung_ten = []
    so_luong_trung_diachi = []
    chi_tiet_trung_ten = []
    chi_tiet_trung_diachi = []

    for i, row in df.iterrows():
        current_id = row["store_short_id"]
        current_store = row[store_col].lower()
        current_addr = row[addr_col].lower()

        matches_ten = []
        matches_diachi = []

        for j, cmp_row in df.iterrows():
            if cmp_row["store_short_id"] == current_id:
                continue

            cmp_store = cmp_row[store_col].lower()
            cmp_addr = cmp_row[addr_col].lower()

            # Trùng tên
            score_store = fuzz.token_sort_ratio(current_store, cmp_store)
            if score_store >= threshold:
                matches_ten.append(
                    f"{{{cmp_row['store_owner']}, {cmp_row['store_created_at']}, {cmp_row['store_short_id']}, {cmp_row[store_col]}}}"
                )

            # Trùng địa chỉ
            score_addr = fuzz.token_sort_ratio(current_addr, cmp_addr)
            if score_addr >= threshold:
                matches_diachi.append(
                    f"{{{cmp_row['store_owner']}, {cmp_row['store_created_at']}, {cmp_row['store_short_id']}, {cmp_row[store_col]}, {cmp_row[addr_col]}}}"
                )

        so_luong_trung_ten.append(len(matches_ten))
        chi_tiet_trung_ten.append(", ".join(matches_ten))

        so_luong_trung_diachi.append(len(matches_diachi))
        chi_tiet_trung_diachi.append(", ".join(matches_diachi))

    # Gộp vào df gốc
    df["Dup_Name"] = so_luong_trung_ten
    df["Dup_Address"] = so_luong_trung_diachi
    df["Detail_Address"] = chi_tiet_trung_diachi
    df["Detail_Name"] = chi_tiet_trung_ten

    return df

def analyze_duplicates_optimized(df, store_col="store_name", addr_col="store_6. Địa chỉ gian hàng *", threshold=99):
    df = df.copy()
    df[store_col] = df[store_col].astype(str).str.lower()
    df[addr_col] = df[addr_col].astype(str).str.lower()
    df["store_short_id"] = df["store_short_id"].astype(str)

    # Khởi tạo kết quả
    df["Dup_Name"] = 0
    df["Dup_Address"] = 0
    df["Detail_Name"] = ""
    df["Detail_Address"] = ""

    store_list = df[store_col].tolist()
    addr_list = df[addr_col].tolist()

    tqdm.pandas(desc="🔍 Processing duplicates")

    def get_matches(index, current_store, current_addr, current_id):
        matches_ten = []
        matches_diachi = []

        for j in range(len(df)):
            if j == index:
                continue

            cmp_id = df.iloc[j]["store_short_id"]
            cmp_store = store_list[j]
            cmp_addr = addr_list[j]

            score_store = fuzz.token_sort_ratio(current_store, cmp_store)
            if score_store >= threshold:
                matches_ten.append(
                    f"{{{df.iloc[j]['store_owner']}, {df.iloc[j]['store_created_at']}, {cmp_id}, {df.iloc[j][store_col]}}}"
                )

            score_addr = fuzz.token_sort_ratio(current_addr, cmp_addr)
            if score_addr >= threshold:
                matches_diachi.append(
                    f"{{{df.iloc[j]['store_owner']}, {df.iloc[j]['store_created_at']}, {cmp_id}, {df.iloc[j][store_col]}, {df.iloc[j][addr_col]}}}"
                )

        return pd.Series([
            len(matches_ten),
            len(matches_diachi),
            ", ".join(matches_ten),
            ", ".join(matches_diachi)
        ], index=["Dup_Name", "Dup_Address", "Detail_Name", "Detail_Address"])

    # Apply có tiến độ
    df[["Dup_Name", "Dup_Address", "Detail_Name", "Detail_Address"]] = df.progress_apply(
        lambda row: get_matches(
            row.name,
            row[store_col],
            row[addr_col],
            row["store_short_id"]
        ), axis=1
    )

    return df    

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
    logger.info(f"\033[91m{client_ip}\033[0m - {method} {masked_url} - {color_status(status_code)} - {duration_ms:.2f}s")
    return response

def dedup_large_dict_list(data: list[dict]) -> list[dict]:
  seen = set()
  unique = []
  for d in data:
# Dùng tuple(sorted(...)) nếu có thể, fallback sang str(...) nếu lỗi
    try:
      key = tuple(sorted((k, str(v)) for k, v in d.items()))
    except Exception:
      key = str(sorted(d.items()))
    if key not in seen:
      seen.add(key)    
      unique.append(d)
  return unique

def appendToRow(myDict: dict, key: str, value: str):
  myDict[key] = value
    
def getdata(token: str):
    try:
        sta = get_current_time_str()
        raw_rows = []
        total_bytes = 0
        for skipp in range(0,30001,150):
            url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=160&skip={skipp - 10 if skipp > 0 else 0}"
            # url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=10"
            headers = {"Authorization": token}
            response = requests.get(url, headers=headers, verify = False)
            if response.status_code == 200:
                sources = response.json()
                size = len(json.dumps(sources).encode('utf-8'))  # tính size theo byte
                total_bytes += size
                excluded_keys = ["weight","area","google_map_address","description","stage_compact","amount", "invoice", "invoices", "opportunity_process",
                               "opportunity_process_stage_id","tax_inclusive_amount","forecast","opportunities_joint","owner_id","opportunities_products",
                               "locked","date_closed_actual","discussions","is_parent","source","opportunity_status","project_type","opportunities_contacts",
                               "Error","notes","parent_opportunity_id","parent_opportunity","opportunities_children","opportunity_type_id","activities",
                               "stage_logs","date_open"]
                special_keys = ["custom_field_opportunity_values","opportunity_process_stage","owner","users_opportunities","accounts_opportunities","created_at"]
                for index, item in enumerate(sources):
                    row = {}
                    for key, value in item.items():
                      if key not in excluded_keys:
                        if key not in special_keys:
                          appendToRow(row, f'store_{key}',value)
                        elif key == "created_at":
                          appendToRow(row, f'store_{key}',value[:10])
                        elif key =="custom_field_opportunity_values":
                          for i in value:
                            if i["custom_field"]["name"] not in ["20. ADO","23. Giá món trung bình *","18. Vĩ độ","19. Kinh độ","21. Quận *"]:
                              appendToRow(row, f'store_{i["custom_field"]["name"]}',i["value"])
                        elif key == "opportunity_process_stage":
                          appendToRow(row, f'store_{key}',value["opportunity_stage"]["name"])
                        elif key == "owner":
                          appendToRow(row, f'store_{key}',value["email"])
                        elif key == "users_opportunities":
                          appendToRow(row, f'store_{key}',value[0]["user"]["email"])
                        elif key == "accounts_opportunities":
                          for k, v in value[0]["account"].items():
                            included_keys = ["id","name","short_id","account_type","owner","custom_field_account_values"]
                            if k in included_keys:
                              if k == "account_type":
                                appendToRow(row, f'mex_info_{k}',v)
                              elif k == "owner":
                                appendToRow(row, f'mex_info_{k}',v["email"])
                              elif k == "custom_field_account_values":
                                for k1 in v:
                                  excluded_keys_mex = ["34. Link ảnh"]
                                  n = k1["custom_field"]["name"]
                                  vl = k1["value"]
                                  if k1["custom_field"]["master_data_custom_fields"]:
                                    for i in k1["custom_field"]["master_data_custom_fields"]:
                                      if i["id"] == vl:
                                        vl = i["value"]
                                  if n not in excluded_keys_mex:
                                    appendToRow(row, f'mex_info_{n}',vl)
                              else:
                                appendToRow(row, f'mex_{k}',v)
                    raw_rows.append(row)
            else:
              return f'Error: {response.status_code} {response.text}'
            if len(sources) < 160: break
        dunique = dedup_large_dict_list(raw_rows)
        sto = get_current_time_str()
        print (f"   {sta} -> {sto}: {total_bytes / (1024 * 1024):.2f} MB - {len(dunique)} store records") 
        return dunique
    except Exception as e:
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
            included_keys = ["account_name","name","id","created_at","custom_field_lead_values","owner"]
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
        dunique = dedup_large_dict_list(raw_rows)
        sto = get_current_time_str()
        print (f"   {sta} -> {sto}: {total_bytes / (1024 * 1024):.2f} MB - {len(dunique)} lead records") 
        return dunique
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
        
@app.get("/")
def home():
  return {"hello":":)"}

@app.get("/opla/")
def api_opla(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    excel: int = Query(None)
):
    try:
        if secrets == 'chucm@ym@n8686':
            with lock:
                if token not in cache:
                    cache[token] = {
                        "data": getdata(token),
                        "updated": get_current_time_str(),
                        "leads": getleads(token)
                    }

                df = pd.DataFrame(cache[token]["data"])

                if limit:
                    df = df.iloc[:limit]
                if fields:
                    valid_fields = [col for col in fields if col in df.columns]
                    df = df[valid_fields]

                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.where(pd.notnull(df), None)
                if not excel or excel != 1:
                    safe_json = json.dumps(
                        df.to_dict(orient="records"),
                        ignore_nan=True
                    )
                    return Response(content=safe_json, media_type="application/json")
                elif excel == 1:                   
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name=cache[token]["updated"])

                    output.seek(0)
                    headers = {
                        "Content-Disposition": f"attachment; filename=store_{get_current_time_str()}.xlsx"
                    }

                    return Response(
                        content=output.read(),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers=headers
                    )
        else:
            return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
        
@app.get("/leads/")
def api_lead(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None),
    limit: int = Query(None),
    excel: int = Query(None)
):
    try:
        if secrets != 'chucm@ym@n8686': return {}
        with lock:
            if token not in cache:
                cache[token] = {
                    "data": getdata(token),
                    "updated": get_current_time_str(),
                    "leads": getleads(token)
                }
            df = pd.DataFrame(cache[token]["leads"])
            if fields: df = df[[col for col in fields if col in df.columns]]
            if limit: df = df.iloc[:limit]
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.where(pd.notnull(df), None)
            if not excel or excel != 1:
                return Response(
                    content=json.dumps(df.to_dict(orient="records"), ignore_nan=True),
                    media_type="application/json"
                )
            elif excel == 1:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name=cache[token]["updated"])

                output.seek(0)
                headers = {
                    "Content-Disposition": f"attachment; filename=leads_{get_current_time_str()}.xlsx"
                }
                return Response(
                    content=output.read(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=headers
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.get("/checkdup/")
def api_checkdup(
    token: str = Query(...),
    secrets: str = Query(...),
    rate: int = Query(None),
    excel: int = Query(None),
    filter0: int = Query(None)
):
    try:
        if secrets != 'chucm@ym@n8686': return {}
        with lock:
            if token not in cache:
                cache[token] = {
                    "data": getdata(token),
                    "updated": get_current_time_str(),
                    "leads": getleads(token)
                }
            if not cache[token]["checkdup"]:
              df = pd.DataFrame(cache[token]["data"])
              df = df[["store_id","store_short_id","store_name","store_owner","store_created_at","store_6. Địa chỉ gian hàng *", "mex_short_id", "mex_name"]]
              if rate:
                  df = analyze_duplicates(df,threshold=rate)
              else:
                  df = analyze_duplicates(df)
              df.replace([np.inf, -np.inf], np.nan, inplace=True)
              df = df.where(pd.notnull(df), None)
              if filter0 == 1: df = df[(df["Dup_Name"] > 0) | (df["Dup_Address"] > 0)]
              cache[token]["checkup"] = df
          
            if not excel or excel != 1:
                return Response(
                    content=json.dumps(cache[token]["checkup"].to_dict(orient="records"), ignore_nan=True),
                    media_type="application/json"
                )
            elif excel == 1:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    cache[token]["checkup"].to_excel(writer, index=False, sheet_name=get_current_time_str())

                output.seek(0)
                headers = {
                    "Content-Disposition": f"attachment; filename=checkdup_{get_current_time_str()}.xlsx"
                }

                return Response(
                    content=output.read(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=headers
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.get("/clear/")
def api_clear(token: str = Query(...), secrets: str = Query(...)):
  if secrets == 'chucm@ym@n8686':
    with lock:
      if token in cache:
        del cache[token]
        return {'result': 'OK :)'}
  else:
    return {"Lỗi": "Sai secrets :("}

@app.get("/updated/")
def last_update(token: str = Query(...)):
  with lock:
    if token in cache:
      return {"updated": cache[token]["updated"]}
    else:
      try:
          cache[token] = {
                            "data": getdata(token),
                            "updated": get_current_time_str(),
                            "leads": getleads(token)
                        }
          return {"updated": cache[token]["updated"]}
      except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")



@app.get("/opla/excel/")
def download_excel(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None)  # 👈 Thêm vào đây
):
    if secrets != 'chucm@ym@n8686':
        return {"Lỗi": "Sai secrets :("}

    with lock:
        if token not in cache:
            cache[token] = {
                "data": getdata(token),
                "updated": get_current_time_str(),
                "leads": getleads(token)
            }

        df = pd.DataFrame(cache[token]["data"])

        # 👇 Lọc theo fields nếu được cung cấp
        if fields:
            valid_fields = [col for col in fields if col in df.columns]
            df = df[valid_fields]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=cache[token]["updated"])

        output.seek(0)
        headers = {
            "Content-Disposition": f"attachment; filename=store_{get_current_time_str()}.xlsx"
        }

        return Response(
            content=output.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )

@app.get("/leads/excel/")
def download_excel(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None)  # 👈 Thêm param lọc cột
):
    if secrets != 'chucm@ym@n8686':
        return {"Lỗi": "Sai secrets :("}

    with lock:
        if token not in cache:
            cache[token] = {
                "data": getdata(token),
                "updated": get_current_time_str(),
                "leads": getleads(token)
            }

        df = pd.DataFrame(cache[token]["leads"])

        # 👇 Lọc cột nếu có fields
        if fields:
            valid_fields = [col for col in fields if col in df.columns]
            df = df[valid_fields]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=cache[token]["updated"])

        output.seek(0)

        headers = {
            "Content-Disposition": f"attachment; filename=leads_{get_current_time_str()}.xlsx"
        }

        return Response(
            content=output.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
