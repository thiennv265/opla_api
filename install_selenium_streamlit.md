sudo sed -i 's/vn.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list && sudo apt clean && sudo apt update
wget -nc https://raw.githubusercontent.com/thiennv265/opla_api/refs/heads/main/install_checkspf.sh && chmod +x install_checkspf.sh && ./install_checkspf.sh

tmux
tmux new -s streamlit_app
tmux attach -t ten_phien
tmux kill-session -t ten_phien
