import getpass, time, requests, datetime
from datetime import datetime, timezone, timedelta

API_URL = getpass.getpass("URL: ")
telegram_token = getpass.getpass("TELE TOKEN: ")
noti_secret = f"Saved: URL: {secret[:10]}** ({len(secret)})\nTELE TOKEN: {telegram_token[:10]}** ({len(telegram_token)})"
print(noti_secret)

def get_current_time_str():
  tz_gmt7 = timezone(timedelta(hours=7))
  now = datetime.now(tz_gmt7)
  return now.strftime('%Y-%m-%d_%H:%M:%S')
  
def send_log(msg, sender_name=None):
    try:
        requests.get(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            params={"chat_id": "716085753", "text": get_current_time_str() + " " + f"{sender_name}\n{msg}" if sender_name else "\n" + msg}
        )
    except Exception as e:
        print(e)
send_log(noti_secret,"f5")
print("Script đang chạy. Sẽ gọi API mỗi giờ :30.")
send_log(f"✅ Started.","f5")

while True:
    now = datetime.now()
    if now.minute == 30 and now.second == 0 and now.hour < 21 and now.hour > 5:
        try:
            response = requests.get(API_URL)
            msg = f"✅ API: {response.status_code}"
        except Exception as e:
            msg = f"❌ API: {e}"
        print(msg)
        send_log(msg,"f5")
        time.sleep(61)
    else:
        time.sleep(1)
