#!/bin/bash
echo "🔧 Đang cập nhật hệ thống..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Cài đặt Python & pip..."
sudo apt install python3 python3-pip -y

echo "📦 Cài đặt thư viện Python..."
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install streamlit selenium beautifulsoup4 pandas openpyxl webdriver-manager 

echo "📦 Cài đặt Tmux..."
sudo apt install tmux

echo "🌐 Tải và cài đặt Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y
rm google-chrome-stable_current_amd64.deb

echo "🌐 Tải và cài đặt thư viện chạy Google Chrome VPS..."
echo "sudo apt-get update"
sudo apt-get install -y fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 xdg-utils

echo "✅ Cài đặt hoàn tất!"
echo "📍 Kiểm tra phiên bản:"
google-chrome --version
chromedriver --version
python3 --version

echo "📥 Đang tải các file scraper..."

wget -nc https://raw.githubusercontent.com/thiennv265/opla_api/refs/heads/main/s1.py
wget -nc https://raw.githubusercontent.com/thiennv265/opla_api/refs/heads/main/s2.py

echo "✅ Tải xong."

echo "🚀 Bây giờ bạn có thể chạy ứng dụng Streamlit như sau:"
echo "streamlit run s1.py --server.port=8501 --server.address=0.0.0.0"
echo "streamlit run s2.py --server.port=8502 --server.address=0.0.0.0"
