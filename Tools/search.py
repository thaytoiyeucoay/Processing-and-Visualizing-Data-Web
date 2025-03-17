import streamlit as st
import pandas as pd
import os
import json
import tempfile
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
import time
import numpy as np
import subprocess
import sys

# Tìm kiếm và hiển thị dataset từ Kaggle
def searchAndDisplayKaggleDatasets(credentials_ok):
    st.subheader(" Hoặc tìm kiếm bộ dữ liệu trên Kaggle")
    query = st.text_input("Nhập từ khóa tìm kiếm")
    #max_results = st.slider("Số lượng kết quả tối đa", 5, 50, 20)
    
    if query and credentials_ok:
        with st.spinner('Đang tìm kiếm...'):
            datasets = search_kaggle_datasets(query)
        
        if not datasets:
            st.warning("Không tìm thấy kết quả nào!")
        else:
            # Hiển thị kết quả tìm kiếm để debug
            #st.write("Kết quả tìm kiếm:", datasets)
            selected_dataset_idx = st.selectbox(
                "Chọn một bộ dữ liệu để xem thêm thông tin:",
                range(len(datasets)),
                format_func=lambda i: datasets[i].title if hasattr(datasets[i], 'title') else f"Dataset {i+1}"
            )
            selected_dataset = datasets[selected_dataset_idx]
            display_dataset_info(selected_dataset)
            
            if st.button("Tải về và hiển thị dữ liệu"):
                with st.spinner('Đang tải dữ liệu...'):
                    temp_dir, csv_files = download_kaggle_dataset(selected_dataset.ref)
                
                if not csv_files:
                    st.warning("Không tìm thấy file CSV nào trong bộ dữ liệu này!")
                else:
                    if len(csv_files) > 1:
                        selected_file = st.selectbox(
                            "Chọn file để hiển thị:",
                            csv_files,
                            format_func=lambda x: os.path.basename(x)
                        )
                    else:
                        selected_file = csv_files[0]
                    
                    try:
                        df = pd.read_csv(selected_file)
                        st.subheader("Thông tin bộ dữ liệu")
                        st.write(f"Số hàng: {df.shape[0]}")
                        st.write(f"Số cột: {df.shape[1]}")
                        st.subheader("Mẫu dữ liệu")
                        st.dataframe(df.head(10))
                        with st.expander("Xem thống kê mô tả"):
                            st.write(df.describe())
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Tải xuống dữ liệu dưới dạng CSV",
                            data=csv,
                            file_name=os.path.basename(selected_file),
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file CSV: {e}")



def install_kaggle():
    """Install the Kaggle package if not already installed."""
    try:
        # Try importing the module to check if it's installed
        import kaggle
        print("Kaggle package is already installed.")
    except ImportError:
        print("Installing Kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("Kaggle package installed successfully.")

def setupKaggleCredentials():
    #install_kaggle()

    try:
        # Read credentials from kaggle.json file
        #with open('C:/Users/BUI KHANH DUY/.kaggle/kaggle.json', 'r') as f:
        #    credentials = json.load(f)
        
        # Set environment variables from the JSON file
        os.environ['KAGGLE_USERNAME'] = "duybuii"
        os.environ['KAGGLE_KEY'] = os.getenv("STREAMLIT_API_KEY")
        
        #st.success("Đã thiết lập thông tin xác thực Kaggle từ file JSON.")
        return True
    except FileNotFoundError:
        #st.error("Không tìm thấy file kaggle.json. Vui lòng tải file từ tài khoản Kaggle của bạn.")
        return False
    except Exception as e:
        #st.error(f"Lỗi khi thiết lập thông tin xác thực Kaggle: {e}")
        return False    

# Hàm khởi tạo và authenticate Kaggle API
def get_kaggle_api():
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        st.error(f"Lỗi authenticate Kaggle API: {e}")
    return api


# Hàm hỗ trợ: Tìm kiếm dataset trên Kaggle
def search_kaggle_datasets(query, max_results=20):
    try:
        api = get_kaggle_api()
        #st.info("Đang gọi hàm tìm kiếm từ Kaggle API...")
        datasets = api.dataset_list(search=query)
        #st.info(f"Đã nhận được {len(datasets)} dataset từ Kaggle.")
        return datasets
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm: {e}")
        return []

# Hàm hỗ trợ: Tải dataset từ Kaggle
def download_kaggle_dataset(dataset_ref):
    try:
        api = get_kaggle_api()
        temp_dir = tempfile.mkdtemp()
        api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        st.info(f"Số file CSV trong dataset: {len(csv_files)}")
        
        if not csv_files:
            for root, _, _ in os.walk(temp_dir):
                csv_files.extend(glob.glob(os.path.join(root, "*.csv")))
                
        return temp_dir, csv_files
    except Exception as e:
        st.error(f"Lỗi khi tải về dataset: {e}")
        return None, []

# Hàm hỗ trợ: Hiển thị thông tin dataset
def display_dataset_info(dataset):
    title = getattr(dataset, 'title', 'Không xác định')
    owner = getattr(dataset, 'ownerName', 'Không xác định')
    size = getattr(dataset, 'size', 'Không xác định')
    last_updated = getattr(dataset, 'lastUpdated', 'Không xác định')
    download_count = getattr(dataset, 'downloadCount', 'Không xác định')
    vote_count = getattr(dataset, 'voteCount', 'Không xác định')
    
    st.subheader(title)
    st.write(f"**Người tạo:** {owner}")
    st.write(f"**Kích thước:** {size}")
    st.write(f"**Cập nhật:** {last_updated}")
    st.write(f"**Lượt tải:** {download_count}")
    st.write(f"**Đánh giá:** {vote_count} 👍")
    
    
    if hasattr(dataset, 'description') and dataset.description:
        with st.expander("Xem mô tả"):
            st.markdown(dataset.description)
    
    st.markdown(f"[Xem trên Kaggle](https://www.kaggle.com/datasets/{dataset.ref})")
    
credentials_ok = setupKaggleCredentials()
searchAndDisplayKaggleDatasets(credentials_ok)

