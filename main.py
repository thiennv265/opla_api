import subprocess
import io
import sys

def install_if_missing(package):
  try:
    __import__(package)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# Cần kiểm tra và cài các thư viện ngoài
for pkg in ["requests", "pandas", "fastapi", "cachetools", "urllib3", "openpyxl", "numpy" ]:
  install_if_missing(pkg)
from typing import List
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import openpyxl
import requests
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Response, Request
from datetime import datetime, timezone, timedelta
from cachetools import TTLCache
from threading import Lock

app = FastAPI(docs_url = None, redoc_url = None, openapi_url = None)
cache = TTLCache(maxsize=1000, ttl=3000)  # 2 tiếng
lock = Lock()

# Lấy giờ GMT+7
def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y%m%d_%H%M%S')

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
        print(f'#{get_current_time_str()} Getting Stores')
        raw_rows = []
        for skipp in range(0,30001,150):
            url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=160&skip={skipp - 10 if skipp > 0 else 0}"
            # url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=10"
            headers = {"Authorization": token}
            response = requests.get(url, headers=headers, verify = False)
            if response.status_code == 200:
                sources = response.json()
                excluded_keys = ["weight","area","google_map_address","description","stage_compact","amount", "invoice", "invoices", "opportunity_process",
                               "opportunity_process_stage_id","tax_inclusive_amount","forecast","opportunities_joint","owner_id","opportunities_products",
                               "locked","date_closed_actual","discussions","is_parent","source","opportunity_status","project_type","opportunities_contacts",
                               "Error","notes","parent_opportunity_id","parent_opportunity","opportunities_children","opportunity_type_id","activities",
                               "stage_logs"]
                special_keys = ["custom_field_opportunity_values","opportunity_process_stage","owner","users_opportunities","accounts_opportunities"]
                for index, item in enumerate(sources):
                    row = {}
                    for key, value in item.items():
                      if key not in excluded_keys:
                        if key not in special_keys:
                          appendToRow(row, f'store_{key}',value)
                        elif key =="custom_field_opportunity_values":
                          for i in value:
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
                                  # else:
                                    # appendToRow(row, f'mex_{k}',v)
                    raw_rows.append(row)
            else:
              return f'Error: {response.status_code} {response.text}'
            if len(sources) < 160: break
        dunique = dedup_large_dict_list(raw_rows)
        print (f"#{get_current_time_str()} Got {len(dunique)} Rows") 
        return dunique
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

def getleads(token: str):
    try:
        print (f"#{get_current_time_str()} Getting Leads") 
        raw_rows = []
        url = f"https://api-admin.oplacrm.com/api/public/leads?take=100000"
        headers = {
            "Authorization": token
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            sources = response.json()
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
        print (f"#{get_current_time_str()} Got {len(dunique)} Rows") 
        return dunique
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
        
@app.get("/")
def home():
  return {"hello":"Chúc sếp ngày mới vui vẻ :)"}

@app.get("/opla/")
def api_opla(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None)
):
    if secrets == 'chucm@ym@n8686':
        with lock:
            if token not in cache:
                cache[token] = {
                    "data": getdata(token),
                    "updated": get_current_time_str(),
                    "leads": getleads(token)
                }
            df = pd.DataFrame(cache[token]["data"])
            if fields:
                valid_fields = [col for col in fields if col in df.columns]
                df = df[valid_fields]
            return df.to_dict(orient="records")
    else:
        return {}  # Không báo lỗi, trả rỗng

@app.get("/leads/")
def api_lead(
    token: str = Query(...),
    secrets: str = Query(...),
    fields: List[str] = Query(None)
):
    if secrets == 'chucm@ym@n8686':
        with lock:
            if token not in cache:
                cache[token] = {
                    "data": getdata(token),
                    "updated": get_current_time_str(),
                    "leads": getleads(token)
                }
            df = pd.DataFrame(cache[token]["leads"])
            
            if fields:
                valid_fields = [col for col in fields if col in df.columns]
                if valid_fields:
                    df = df[valid_fields]
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.where(pd.notnull(df), None)
            return df.to_dict(orient="records")
    else:
        return {}

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
async def last_update(token: str = Query(...)):
  with lock:
    if token in cache:
      return {"updated": cache[token]["updated"]}
    else:
      return {"updated": "N/A"}

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
