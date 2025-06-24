import subprocess
import io
import sys

def install_if_missing(package):
  try:
    __import__(package)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# Cần kiểm tra và cài các thư viện ngoài
for pkg in ["requests", "pandas", "fastapi", "cachetools", "urllib3", "openpyxl" ]:
  install_if_missing(pkg)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import openpyxl
import requests
import json
import pandas as pd
from fastapi import FastAPI, Query, Response
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
  print(f'#{get_current_time_str()}')
  raw_rows = []
  for skipp in range(0,30001,150):
    url = f"https://api-admin.oplacrm.com/api/public/opportunities?take=160&skip={skipp - 10 if skipp > 0 else 0}"
    headers = {"Authorization": token}
    response = requests.get(url, headers=headers, verify = False)
    if response.status_code == 200:
      sources = response.json()
      excluded_keys = ["weight","area","google_map_address","description","stage_compact","amount", "invoice", "invoices", "opportunity_process",
                       "opportunity_process_stage_id","tax_inclusive_amount","forecast","opportunities_joint","owner_id","opportunities_products",
                       "locked","date_closed_actual","discussions","is_parent","source","opportunity_status","project_type","opportunities_contacts",
                       "Error","notes","parent_opportunity_id","parent_opportunity","opportunities_children","opportunity_type_id","activities"]
      special_keys = ["custom_field_opportunity_values","opportunity_process_stage","owner","users_opportunities","accounts__opportunities"]
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
                      else:
                        appendToRow(row, f'mex_{k}',v)
        raw_rows.append(row)
        if len(sources) < 160: break
    else:
      return f'Error: {response.status_code} {response.text}'
  dunique = dedup_large_dict_list(raw_rows)
  print (f"Total {len(dunique)} rows") 
  return dunique

@app.get("/")
def home():
  return {"hello":"Chúc sếp ngày mới vui vẻ :)"}

@app.get("/opla/")
def api_opla(token: str = Query(...), secrets: str = Query(...)):
if secrets == 'chucm@ym@n8686':
  with lock:
    if token not in cache:
      cache[token] = {"data": getdata(token), "updated": get_current_time_str()}
    return cache[token]["data"]
else:
  return {"Lỗi": "Sai secrets :("}

@app.get("/now/")
def api_now(token: str = Query(...), secrets: str = Query(...)):
if secrets == 'chucm@ym@n8686':
  with lock:
    cache[token] = {"data": getdata(token), "updated": get_current_time_str()}
  return cache[token]["data"]
else:
  return {"Lỗi": "Sai secrets :("}

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
  return {"updated": "N/A F5"}

@app.get("/opla/excel")
def download_excel(token: str = Query(...), secrets: str = Query(...)):
  if secrets != 'chucm@ym@n8686':
    return {"Lỗi": "Sai secrets :("}
  with lock:
    if token not in cache:
      cache[token] = {"data": getdata(token), "updated": get_current_time_str()}
      df = pd.DataFrame(cache[token]["data"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
      df.to_excel(writer, index=False, sheet_name=cache[token]["updated"])
      output.seek(0)
      headers = {"Content-Disposition": f"attachment; filename={get_current_time_str()}.xlsx",
      "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    return Response(content=output.read(), media_type=headers["Content-Type"], headers=headers)
