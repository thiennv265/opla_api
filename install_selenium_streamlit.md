nano install_selenium_streamlit.sh

-----
#!/bin/bash

echo "🔧 Đang cập nhật hệ thống..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Cài đặt Python & pip..."
sudo apt install python3 python3-pip -y

echo "📦 Cài đặt thư viện Python..."
pip3 install --upgrade pip
pip3 install streamlit selenium beautifulsoup4 pandas openpyxl

echo "🌐 Tải và cài đặt Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O chrome.deb
sudo apt install ./chrome.deb -y
rm chrome.deb

echo "🔍 Xác định version Chrome..."
CHROME_VERSION=$(google-chrome --version | grep -oP '[0-9.]+' | head -1 | cut -d '.' -f 1)
echo "→ Chrome major version: $CHROME_VERSION"

echo "⚙️ Tải ChromeDriver tương thích..."
LATEST_DRIVER=$(wget -qO- "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION}")
wget https://chromedriver.storage.googleapis.com/${LATEST_DRIVER}/chromedriver_linux64.zip -O chromedriver.zip
unzip chromedriver.zip
sudo mv chromedriver /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver
rm chromedriver.zip

echo "✅ Cài đặt hoàn tất!"
echo "📍 Kiểm tra phiên bản:"
google-chrome --version
chromedriver --version
python3 --version

echo "📥 Đang tải các file scraper..."

wget -nc https://raw.githubusercontent.com/yourname/yourrepo/main/scraper1.py
wget -nc https://raw.githubusercontent.com/yourname/yourrepo/main/scraper2.py

echo "✅ Tải xong."

echo "chmod +x install_selenium_streamlit.sh"
echo "./install_selenium_streamlit.sh"
echo "🚀 Bây giờ bạn có thể chạy ứng dụng Streamlit như sau:"
echo "streamlit run scraper1.py --server.port=8501 --server.address=0.0.0.0"
echo "streamlit run scraper2.py --server.port=8502 --server.address=0.0.0.0"

-----
