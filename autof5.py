import getpass, time, requests, datetime
from datetime import datetime, timezone, timedelta

API_URL = getpass.getpass("URL: ")
telegram_token = getpass.getpass("TELE TOKEN: ")
noti = f"Saved: URL: {API_URL[:10]}** ({len(API_URL)})\nTELE TOKEN: {telegram_token[:10]}** ({len(telegram_token)})"
print(noti)

def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y-%m-%d_%H-%M-%S')
  
def send_log(msg, sender_name=None):
    try:
        requests.get(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            params={"chat_id": "716085753", "text": get_current_time_str() + " " + f"{sender_name}\n{msg}" if sender_name else "\n" + msg,"disable_web_page_preview": True}
        )
    except Exception as e:
        print(e)
send_log(noti,"f5")
print("Script đang chạy. Sẽ gọi API mỗi giờ :30.")
send_log(f"✅ Started.","f5")

while True:
    now = datetime.now()
    if now.minute == 30 and now.second == 0 and now.hour % 2 == 0:
        try:
            response = requests.get(API_URL)
            msg = f"✅ API: {response.status_code} {API_URL[:10]}**"
        except Exception as e:
            msg = f"❌ API: {e} {response.status_code} {API_URL[:10]}**"
        print(msg)
        send_log(msg,"f5")
        time.sleep(61)
    else:
        time.sleep(1)
