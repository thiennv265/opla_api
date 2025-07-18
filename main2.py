token = "Bearer $2a$10$47WeoE0UFrhlxluqV5oVReAgjkJLiMQ4PtZLmsHbSv5M5unVgicFq"
from fastapi import FastAPI
import aiohttp
import asyncio
import json
import time, random
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

app = FastAPI()
token = token

# Danh sách skip ban đầu (có thể lớn đến 30000)
skips = list(range(0, 30001, 170))

# Biến cờ dừng toàn cục
stop_flag = asyncio.Event()

raw_rows = []
raw_logs = []
MAX_RETRIES = 3
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

def dedup_dicts_smart(data: list[dict]) -> list[dict]:
    try:
        # return pd.DataFrame(data).drop_duplicates().to_dict(orient="records")
        df = pd.DataFrame(data).drop_duplicates()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.where(pd.notnull(df), None)
        return df
    except Exception as e:
        print(e)
        
async def fetch_worker(worker_id: int, queue: asyncio.Queue, session, stats: dict):
    while not queue.empty() and not stop_flag.is_set():
        url, skipp = await queue.get()
        await fetch_url_with_retry(worker_id, url, session, stats)

async def fetch_url_with_retry(worker_id: int, url: str, session, stats: dict):
    headers = {"Authorization": token}
    retries = 0

    while retries < MAX_RETRIES and not stop_flag.is_set():
        print(f"[Worker-{worker_id}] 🚀 Fetching {url} (try {retries + 1}/{MAX_RETRIES})")
        await asyncio.sleep(random.uniform(1, 3))

        try:
            async with session.get(url, headers=headers, ssl=False) as response:
                if response.status == 401:
                    print(f"[Worker-{worker_id}] 🔐 Token sai tại {url}")
                    stop_flag.set()
                    return

                if response.status != 200:
                    print(f"[Worker-{worker_id}] ❌ HTTP {response.status} tại {url}")
                    break  # không retry nếu không phải lỗi mạng

                data = await response.json()
                sources = data
                if not isinstance(data, list):
                    print(f"[Worker-{worker_id}] ⚠️ Dữ liệu không phải list: {data}")
                    break

                if not data:
                    print(f"[Worker-{worker_id}] 🛑 Dừng lại: {url} trả về rỗng")
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

                print(f"[Worker-{worker_id}] ✅ {count} phần tử từ {url}")
                return  # kết thúc thành công

        except Exception as e:
            print(f"[Worker-{worker_id}] ❗ Lỗi: {e} tại {url}")
            retries += 1

    print(f"[Worker-{worker_id}] ❌ Bỏ qua {url} sau {MAX_RETRIES} lần thử")



@app.get("/fetch-opportunities-queue")
async def fetch_opportunities_queue():
    start_time = time.time()

    # Tạo danh sách URL
    urls = [
        (f"https://api-admin.oplacrm.com/api/public/opportunities?take=180&skip={skipp - 20 if skipp > 0 else 0}", skipp)
        for skipp in skips
    ]

    queue = asyncio.Queue()
    for item in urls:
        await queue.put(item)

    stats = {"total_items": 0, "total_bytes": 0}

    async with aiohttp.ClientSession() as session:
        # Tạo 3 worker chạy song song
        tasks = [
            fetch_worker(i + 1, queue, session, stats)
            for i in range(6)
        ]
        await asyncio.gather(*tasks)

    duration = round(time.time() - start_time, 2)
    print(f"✅ Hoàn thành trong {duration} giây")
    # print(f"📦 Tổng phần tử: {stats['total_items']}")
    print(f"📦 Tổng phần tử: {len(dedup_dicts_smart(raw_rows))} stores & {len(dedup_dicts_smart(raw_logs))} logs")
    print(f"💾 Tổng dung lượng: {stats['total_bytes']} bytes")

    return {
        "total_time_seconds": duration,
        "total_items": stats["total_items"],
        "total_bytes": stats["total_bytes"],
        "stopped_at_skip": None if not stop_flag.is_set() else True
    }
