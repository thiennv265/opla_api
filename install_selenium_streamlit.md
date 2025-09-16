nano install_selenium_streamlit.sh

-----
#!/bin/bash

echo "🔧 Đang cập nhật hệ thống..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Cài đặt Python & pip..."
sudo apt install python3 python3-pip -y

echo "📦 Cài đặt thư viện Python..."
pip3 install --upgrade pip
pip3 install streamlit selenium beautifulsoup4 pandas openpyxl webdriver-manager

echo "🌐 Tải và cài đặt Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y
rm google-chrome-stable_current_amd64.deb

echo "🔍 Xác định version Chrome..."
CHROME_VERSION=$(google-chrome --version | grep -oP '[0-9.]+' | head -1 | cut -d '.' -f 1)
echo "→ Chrome major version: $CHROME_VERSION"

echo "⚙️ Tải ChromeDriver tương thích..."

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

-----
chmod +x install_selenium_streamlit.sh
./install_selenium_streamlit.sh
