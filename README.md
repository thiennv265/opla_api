# opla_api
wget -O main.py https://raw.githubusercontent.com/user/repo/branch/path/to/file.py
sudo apt update
sudo apt install python3 python3-pip nginx certbot python3-certbot-nginx python3-fastapi python3-uvicorn tmux python3-venv -y && python3 -m venv venv && source venv/bin/activate && pip3 install fastapi uvicorn requests pandas fastapi cachetools urllib3 openpyxl numpy simplejson asyncio rapidfuzz aiohttp
tmux new -s fastapi
uvicorn main:app --host 0.0.0.0 --port 8764 --reload --timeout-keep-alive 1800 --no-access-log
tmux a -t fastapi
sudo nano /etc/nginx/sites-available/opla.2tr.top
sudo ln -s /etc/nginx/sites-available/opla.2tr.top /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
chỉnh ip vps
sudo certbot --nginx -d opla.2tr.top

#swap-ram
swapon --show
free -h
sudo fallocate -l 2G /swapfile
# Nếu không có fallocate, dùng lệnh thay thế:
# sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
