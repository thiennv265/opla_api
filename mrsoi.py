import importlib
import subprocess
import sys
from datetime import datetime

def install_if_missing(package):
  try:
    __import__(package)
  except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])
# Cần kiểm tra và cài các thư viện ngoài
for pkg in ["rapidfuzz", "unidecode", "openpyxl", "pandas"]:
  install_if_missing(pkg)

# ======= Import sau khi đã đảm bảo cài =======
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import unidecode
from io import BytesIO

# ======= Hàm xử lý chuỗi =======
def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.strip().lower()
    text = unidecode.unidecode(text)  # bỏ dấu tiếng Việt
    return text

st.set_page_config(layout="wide")
# ======= Tiêu đề App =======
st.title("🔍 Mr. Súp Pờ Soi")
st.write("Check trùng Store theo Name và Address với % ngưỡng trùng :))\nChung tay nói không với mafia")

# ======= Upload file =======
uploaded_file = st.file_uploader("📂 Tải lên file Excel có cột ID, Name, Address, Stage", type=["xlsx"])

# ======= Chọn ngưỡng fuzzy match =======
col1, col2 = st.columns(2)
with col1:
    name_threshold = st.slider("Ngưỡng trùng Name (%)", min_value=0, max_value=100, value=90)
with col2:
    address_threshold = st.slider("Ngưỡng trùng Address (%)", min_value=0, max_value=100, value=90)

if uploaded_file:
    df = pd.read_excel(uploaded_file, dtype={"ID": str})
    
    if "Name" not in df.columns or "Address" not in df.columns or "ID" not in df.columns or "Stage" not in df.columns:
        st.error("❌ File Excel phải có cột: Name, Address, ID, Stage")
    else:
        if st.button("🚀 Chạy kiểm tra trùng"):
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            # Tiền xử lý
            df['Name_clean'] = df['Name'].apply(preprocess)
            df['Address_clean'] = df['Address'].apply(preprocess)
            df['Name_first'] = df['Name_clean'].str[0]

            results = []
            match_count = 0

            groups = list(df.groupby('Name_first'))
            total_groups = len(groups)

            progress_bar = st.progress(0)
            log_box = st.empty()

            for g_idx, (group_key, group_df) in enumerate(groups, start=1):
                rows = group_df.index.tolist()
                for i in range(len(rows)):
                    for j in range(i+1, len(rows)):
                        idx1, idx2 = rows[i], rows[j]
                        name_score = fuzz.ratio(df.loc[idx1, 'Name_clean'], df.loc[idx2, 'Name_clean'])
                        address_score = fuzz.ratio(df.loc[idx1, 'Address_clean'], df.loc[idx2, 'Address_clean'])

                        if name_score >= name_threshold and address_score >= address_threshold:
                            match_count += 1
                            results.append({
                                "ID": df.loc[idx1, 'ID'],
                                "Store": df.loc[idx1, 'Name'],
                                "Address": df.loc[idx1, 'Address'],
                                "Stage": df.loc[idx1, 'Stage'],
                                "short_id_trung": df.loc[idx2, 'ID'],
                                "Store_trung": df.loc[idx2, 'Name'],
                                "Address_trung": df.loc[idx2, 'Address'],
                                "Stage_trung": df.loc[idx2, 'Stage'],
                                "NameMatch(%)": name_score,
                                "AddressMatch(%)": address_score
                            })
                            log_box.write(f"🔍 Tìm thấy {match_count} cặp trùng (Nhóm '{group_key}')")

                # Cập nhật tiến độ
                progress_bar.progress(g_idx / total_groups)
            log_box = st.empty()
            st.success(f"✅ Hoàn tất! Tìm thấy {match_count} cặp trùng.")

            if results:
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)

                # Nút export
                output = BytesIO()
                result_df.to_excel(output, index=False)
                output.seek(0)

                st.download_button(
                    label="💾 Tải xuống kết quả",
                    data=output,
                    file_name="matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
