import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, TargetEncoder, HashingEncoder
from sklearn.model_selection import KFold
import io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Thiết kế trang
def designPage():
    st.set_page_config(
        page_title="Mã hóa dữ liệu cho Machine Learning",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://example.com/help',
            'Report a bug': 'https://example.com/bug',
            'About': 'Công cụ mã hóa dữ liệu cho Machine Learning'
        }
    )

# Thiết kế tiêu đề
def designTitle():
    st.markdown("""
    <style>
    @keyframes gradient {
      0% {
        background-position: 0% 50%;
      }
      50% {
        background-position: 100% 50%;
      }
      100% {
        background-position: 0% 50%;
      }
    }
    .animated-gradient-title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(-45deg, #3498db, #9b59b6, #1abc9c, #f1c40f);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 6s ease infinite;
        margin-bottom: 20px;
    }
    .encoding-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .encoding-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #2196F3;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        font-weight: 500;
    }
    .info-box h3 {
        color: #0c63e4;
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
    }
    .info-box p {
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .info-box strong {
        color: #000000;
        font-weight: 700;
    }
    .info-box ul {
        margin-left: 20px;
        padding-left: 0;
    }
    .info-box li {
        margin-bottom: 5px;
    }
    .feature-icon {
        font-size: 24px;
        margin-right: 10px;
        color: #3498db;
    }
    .method-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .highlight-text {
        background-color: #f1c40f;
        padding: 2px 5px;
        border-radius: 3px;
    }
    </style>
    <p class="animated-gradient-title">Mã hóa dữ liệu (Encoding) cho Machine Learning</p>
    """, unsafe_allow_html=True)

# Hàm tạo link tải xuống
def get_table_download_link(df, filename="data.csv", text="Tải xuống dữ liệu"):
    """Tạo link tải xuống cho DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# Các hàm mã hóa
def one_hot_encode(df, columns):
    """Thực hiện One-Hot Encoding"""
    return pd.get_dummies(df, columns=columns, drop_first=False)

def label_encode(df, columns):
    """Thực hiện Label Encoding"""
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[f"{col}_label_encoded"] = le.fit_transform(df[col])
        encoders[col] = {value: i for i, value in enumerate(le.classes_)}
        
    return df_encoded, encoders

def binary_encode(df, columns, return_mappings=False):
    """Thực hiện Binary Encoding"""
    df_encoded = df.copy()
    encoder = BinaryEncoder(cols=columns)
    binary_encoded = encoder.fit_transform(df[columns])
    
    # Kết hợp kết quả với DataFrame gốc
    df_encoded = pd.concat([df_encoded, binary_encoded], axis=1)
    
    # Tạo ánh xạ categories -> binary
    if return_mappings:
        mappings = {}
        for col in columns:
            uniques = df[col].unique()
            binary_vals = {}
            for val in uniques:
                # Lấy giá trị binary cho mỗi giá trị duy nhất
                temp_df = pd.DataFrame({col: [val]})
                encoded = encoder.transform(temp_df)
                binary_cols = [c for c in encoded.columns if c.startswith(f"{col}_")]
                binary_vals[val] = encoded[binary_cols].values[0].tolist()
            mappings[col] = binary_vals
        return df_encoded, mappings
    
    return df_encoded

def target_encode(df, columns, target_col, cv=5, add_noise=False, noise_level=0.01):
    """Thực hiện Target Encoding với K-fold"""
    df_encoded = df.copy()
    encoders = {}
    
    if target_col not in df.columns:
        st.error(f"Cột mục tiêu {target_col} không tồn tại trong dữ liệu!")
        return df
    
    for col in columns:
        # Tạo cột kết quả
        encoded_col_name = f"{col}_target_encoded"
        df_encoded[encoded_col_name] = np.nan
        
        # Thực hiện K-fold target encoding
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        encoding_map = {}
        
        for train_idx, test_idx in kf.split(df):
            # Chia dữ liệu thành train và test
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Tính target mean cho mỗi category trong tập train
            target_means = train_df.groupby(col)[target_col].mean()
            encoding_map.update(target_means.to_dict())
            
            # Áp dụng encoding vào tập test
            for category, mean_val in target_means.items():
                mask = test_df[col] == category
                # Thêm nhiễu nếu cần
                if add_noise:
                    noise = np.random.normal(0, noise_level, size=sum(mask))
                    df_encoded.loc[df.index[test_idx][mask], encoded_col_name] = mean_val + noise
                else:
                    df_encoded.loc[df.index[test_idx][mask], encoded_col_name] = mean_val
            
            # Xử lý giá trị chưa thấy trong train
            global_mean = train_df[target_col].mean()
            mask_unknown = ~test_df[col].isin(target_means.index)
            df_encoded.loc[df.index[test_idx][mask_unknown], encoded_col_name] = global_mean
        
        encoders[col] = encoding_map
    
    return df_encoded, encoders

def hash_encode(df, columns, n_components=8):
    """Thực hiện Hash Encoding"""
    df_encoded = df.copy()
    encoder = HashingEncoder(cols=columns, n_components=n_components)
    hash_encoded = encoder.fit_transform(df[columns])
    
    # Kết hợp kết quả với DataFrame gốc
    df_encoded = pd.concat([df_encoded, hash_encoded], axis=1)
    return df_encoded

# Hiển thị biểu đồ so sánh (before/after encoding)
def show_encoding_comparison(original_df, encoded_df, category_col, encoded_col, target_col=None):
    fig = plt.figure(figsize=(12, 6))
    
    if target_col:
        # Tạo scatter plot cho dữ liệu trước và sau khi encoding
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=original_df[category_col].astype(str), y=original_df[target_col])
        plt.title(f"{target_col} vs {category_col} (Before Encoding)")
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=encoded_df[encoded_col], y=encoded_df[target_col])
        plt.title(f"{target_col} vs {encoded_col} (After Encoding)")
    else:
        # Tạo barplot cho dữ liệu trước và sau khi encoding
        plt.subplot(1, 2, 1)
        value_counts = original_df[category_col].value_counts().sort_index()
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
        plt.title(f"Distribution of {category_col} (Before Encoding)")
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # Chỉ hiển thị khi encoded_col là một cột duy nhất (như label encoding)
        if encoded_col in encoded_df.columns:
            value_counts = encoded_df[encoded_col].value_counts().sort_index()
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
            plt.title(f"Distribution of {encoded_col} (After Encoding)")
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def app():
    # Thiết kế trang
    designPage()
    
    # Hiển thị tiêu đề
    designTitle()
    
    # Giới thiệu về encoding
    with st.expander("💡 Giới thiệu về Encoding trong Machine Learning", expanded=False):
        st.markdown("""
        <div class="info-box">
        <h3>Tại sao cần Encoding?</h3>
        <p>Các thuật toán máy học thường yêu cầu dữ liệu đầu vào ở dạng số. Encoding giúp chuyển đổi dữ liệu phân loại thành dạng số để máy học có thể hiểu và xử lý.</p>
        
        <h3>Các phương pháp Encoding</h3>
        <ul>
            <li><strong>One-Hot Encoding:</strong> Tạo cột mới cho mỗi giá trị phân loại, phù hợp với các mô hình tuyến tính và neural networks.</li>
            <li><strong>Label Encoding:</strong> Gán số nguyên cho mỗi danh mục, phù hợp với mô hình cây quyết định.</li>
            <li><strong>Binary Encoding:</strong> Chuyển đổi nhãn danh mục thành biểu diễn nhị phân, giảm kích thước so với one-hot.</li>
            <li><strong>Target Encoding:</strong> Thay thế mỗi giá trị phân loại bằng giá trị trung bình của biến mục tiêu tương ứng.</li>
            <li><strong>Hash Encoding:</strong> Sử dụng hàm băm để chuyển đổi giá trị phân loại, hữu ích cho dữ liệu lớn với nhiều danh mục.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload dữ liệu
    st.markdown('<h2 style="color:#3498db;"><span class="feature-icon">📤</span> Tải lên dữ liệu</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu cần mã hóa", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Đọc dữ liệu
            df = pd.read_csv(uploaded_file)
            
            # Hiển thị thông tin dữ liệu
            st.markdown('<h2 style="color:#3498db;"><span class="feature-icon">📊</span> Thông tin dữ liệu</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Số hàng:** {df.shape[0]}")
                st.write(f"**Số cột:** {df.shape[1]}")
            
            with col2:
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                st.write(f"**Số cột phân loại:** {len(categorical_columns)}")
                st.write(f"**Các cột phân loại:** {', '.join(categorical_columns) if categorical_columns else 'Không có'}")
                
            # Hiển thị dữ liệu
            st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
            st.write("### Dữ liệu đã tải lên")
            st.dataframe(df.head(10))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Phần mã hóa dữ liệu
            if categorical_columns:
                st.markdown('<h2 style="color:#3498db;"><span class="feature-icon">🔄</span> Mã hóa dữ liệu</h2>', unsafe_allow_html=True)
                
                # Chọn cột cần encoding
                selected_columns = st.multiselect(
                    "Chọn các cột cần mã hóa:",
                    categorical_columns,
                    default=categorical_columns[:min(3, len(categorical_columns))]
                )
                
                if selected_columns:
                    # Tạo tabs cho các phương pháp encoding
                    method_tabs = st.tabs([
                        "One-Hot Encoding", 
                        "Label Encoding", 
                        "Binary Encoding", 
                        "Target Encoding", 
                        "Hash Encoding",
                        "Kết hợp nhiều phương pháp"
                    ])
                    
                    # TAB 1: ONE-HOT ENCODING
                    with method_tabs[0]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">One-Hot Encoding</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này chuyển đổi mỗi giá trị phân loại thành một cột mới với giá trị 0 hoặc 1.</p>
                        <p><strong>Phù hợp với:</strong> Mô hình tuyến tính, Neural Networks, khi số lượng giá trị phân loại ít.</p>
                        <p><strong>Ưu điểm:</strong> Không áp đặt thứ tự, dễ hiểu, tương thích với hầu hết các thuật toán.</p>
                        <p><strong>Nhược điểm:</strong> Tạo ra nhiều cột khi có nhiều giá trị phân loại.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tùy chọn
                        drop_first = st.checkbox("Loại bỏ cột đầu tiên (tránh đa cộng tuyến)", value=False)
                        
                        if st.button("Áp dụng One-Hot Encoding", key="btn_onehot"):
                            with st.spinner("Đang thực hiện One-Hot Encoding..."):
                                # Thực hiện encoding
                                df_encoded = pd.get_dummies(df, columns=selected_columns, drop_first=drop_first)
                                
                                # Hiển thị kết quả
                                st.success(f"Đã mã hóa One-Hot thành công cho {len(selected_columns)} cột!")
                                st.write("### Dữ liệu sau khi mã hóa")
                                st.dataframe(df_encoded.head(10))
                                
                                # Thông tin về kích thước dữ liệu
                                orig_cols = len(df.columns)
                                new_cols = len(df_encoded.columns)
                                st.info(f"Kích thước dữ liệu ban đầu: {orig_cols} cột → Kích thước sau mã hóa: {new_cols} cột")
                                
                                # Hiển thị ma trận 0-1 một cách trực quan
                                st.write("### Ma trận One-Hot (0 và 1)")
                                
                                # Lấy tất cả các cột đã được one-hot encode
                                onehot_cols = []
                                for col in selected_columns:
                                    onehot_cols.extend([c for c in df_encoded.columns if c.startswith(f"{col}_")])
                                
                                if onehot_cols:
                                    # Hiển thị ma trận với màu sắc
                                    num_rows = min(20, df_encoded.shape[0])  # Giới hạn số hàng hiển thị
                                    
                                    # Tạo heatmap với plotly
                                    fig = px.imshow(
                                        df_encoded[onehot_cols].head(num_rows),
                                        labels=dict(x="Cột One-Hot", y="Dòng dữ liệu", color="Giá trị"),
                                        x=onehot_cols,
                                        y=[f"Dòng {i+1}" for i in range(num_rows)],
                                        color_continuous_scale=[[0, '#ffffff'], [1, '#2196F3']],
                                        title=f"Ma trận One-Hot (20 mẫu đầu tiên)",
                                        aspect="auto",
                                        height=500
                                    )
                                    
                                    # Thêm số 0, 1 vào ô
                                    for i in range(num_rows):
                                        for j in range(len(onehot_cols)):
                                            value = df_encoded[onehot_cols].iloc[i, j]
                                            fig.add_annotation(
                                                x=j, y=i,
                                                text=str(int(value)),
                                                showarrow=False,
                                                font=dict(color='black' if value == 0 else 'white', size=12),
                                            )
                                    
                                    fig.update_layout(
                                        xaxis=dict(side='top'),
                                        coloraxis_showscale=False
                                    )
                                    
                                    # Hiển thị biểu đồ
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Hiển thị kết quả với CSV
                                with st.expander("Xem dữ liệu One-Hot dưới dạng bảng"):
                                    # Hiển thị bảng dữ liệu chỉ với các cột one-hot
                                    sample_df = df.loc[:min(9, len(df)-1), selected_columns].reset_index(drop=True)
                                    onehot_df = df_encoded.loc[:min(9, len(df_encoded)-1), onehot_cols].reset_index(drop=True)
                                    
                                    st.write("### Dữ liệu gốc:")
                                    st.dataframe(sample_df)
                                    
                                    st.write("### Ma trận One-Hot:")
                                    st.dataframe(onehot_df.style.applymap(
                                        lambda x: 'background-color: #2196F3; color: white' if x == 1 else 'background-color: white; color: black'
                                    ))
                                
                                # Tải xuống
                                st.markdown(get_table_download_link(df_encoded, "onehot_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                            
                    # TAB 2: LABEL ENCODING
                    with method_tabs[1]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Label Encoding</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này chuyển đổi mỗi giá trị phân loại thành một số nguyên (0, 1, 2, ...).</p>
                        <p><strong>Phù hợp với:</strong> Mô hình dựa trên cây quyết định (Decision Tree, Random Forest, XGBoost).</p>
                        <p><strong>Ưu điểm:</strong> Đơn giản, không tăng kích thước dữ liệu.</p>
                        <p><strong>Nhược điểm:</strong> Áp đặt thứ tự không mong muốn giữa các giá trị phân loại.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Áp dụng Label Encoding", key="btn_label"):
                            with st.spinner("Đang thực hiện Label Encoding..."):
                                # Thực hiện encoding
                                df_encoded, encoders = label_encode(df, selected_columns)
                                
                                # Hiển thị kết quả
                                st.success(f"Đã mã hóa Label thành công cho {len(selected_columns)} cột!")
                                st.write("### Dữ liệu sau khi mã hóa")
                                st.dataframe(df_encoded.head(10))
                                
                                # Hiển thị mapping
                                st.write("### Ánh xạ giá trị phân loại sang số")
                                for col, mapping in encoders.items():
                                    st.write(f"**Cột {col}:**")
                                    st.json(mapping)
                                
                                # Trực quan hóa
                                if len(selected_columns) > 0:
                                    st.write("### Trực quan hóa kết quả mã hóa")
                                    col_to_viz = selected_columns[0]
                                    encoded_col = f"{col_to_viz}_label_encoded"
                                    
                                    fig = show_encoding_comparison(df, df_encoded, col_to_viz, encoded_col)
                                    st.pyplot(fig)
                                
                                # Tải xuống
                                st.markdown(get_table_download_link(df_encoded, "label_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 3: BINARY ENCODING
                    with method_tabs[2]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Binary Encoding</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này chuyển đổi mỗi giá trị phân loại thành biểu diễn nhị phân.</p>
                        <p><strong>Phù hợp với:</strong> Khi có nhiều giá trị phân loại, cần tiết kiệm không gian.</p>
                        <p><strong>Ưu điểm:</strong> Ít tạo cột hơn One-Hot (log₂N cột cho N giá trị), hiệu quả với dữ liệu có nhiều giá trị phân loại.</p>
                        <p><strong>Nhược điểm:</strong> Kém trực quan, khó giải thích.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Áp dụng Binary Encoding", key="btn_binary"):
                            with st.spinner("Đang thực hiện Binary Encoding..."):
                                try:
                                    # Thực hiện encoding
                                    df_encoded, binary_mappings = binary_encode(df, selected_columns, return_mappings=True)
                                    
                                    # Hiển thị kết quả
                                    st.success(f"Đã mã hóa Binary thành công cho {len(selected_columns)} cột!")
                                    st.write("### Dữ liệu sau khi mã hóa")
                                    st.dataframe(df_encoded.head(10))
                                    
                                    # Hiển thị mapping
                                    if st.checkbox("Hiển thị ánh xạ giá trị -> nhị phân", value=False):
                                        st.write("### Ánh xạ giá trị phân loại sang nhị phân")
                                        for col, mapping in binary_mappings.items():
                                            st.write(f"**Cột {col}:**")
                                            # Giới hạn hiển thị 10 giá trị đầu tiên nếu có quá nhiều
                                            display_map = dict(list(mapping.items())[:10])
                                            if len(mapping) > 10:
                                                st.json(display_map)
                                                st.info(f"Chỉ hiển thị 10/{len(mapping)} giá trị. Tất cả giá trị có thể tải xuống dưới dạng CSV.")
                                            else:
                                                st.json(mapping)
                                    
                                    # Trực quan hóa
                                    if len(selected_columns) == 1:
                                        st.write("### Trực quan hóa kết quả mã hóa")
                                        binary_cols = [col for col in df_encoded.columns if col.startswith(f"{selected_columns[0]}_")]
                                        
                                        fig = px.imshow(
                                            df_encoded[binary_cols].head(20),
                                            title=f"Ma trận mã hóa nhị phân (20 mẫu đầu tiên)",
                                            color_continuous_scale="Blues"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Tải xuống
                                    st.markdown(get_table_download_link(df_encoded, "binary_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                                    
                                    # Tạo bảng mapping
                                    if st.checkbox("Tạo file ánh xạ (mapping)", value=False):
                                        mapping_dfs = []
                                        for col, mapping in binary_mappings.items():
                                            for val, binary in mapping.items():
                                                mapping_dfs.append({
                                                    "column": col, 
                                                    "original_value": val,
                                                    "binary_representation": str(binary)
                                                })
                                        mapping_df = pd.DataFrame(mapping_dfs)
                                        st.markdown(get_table_download_link(mapping_df, "binary_mapping.csv", "📥 Tải xuống bảng ánh xạ (CSV)"), unsafe_allow_html=True)
                                        
                                except Exception as e:
                                    st.error(f"Lỗi khi thực hiện Binary Encoding: {e}")
                                    st.info("Hãy đảm bảo đã cài đặt thư viện 'category_encoders': pip install category_encoders")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 4: TARGET ENCODING
                    with method_tabs[3]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Target Encoding</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này thay thế mỗi giá trị phân loại bằng giá trị trung bình của biến mục tiêu tương ứng.</p>
                        <p><strong>Phù hợp với:</strong> Khi có nhiều giá trị phân loại và tồn tại mối quan hệ với biến mục tiêu.</p>
                        <p><strong>Ưu điểm:</strong> Hiệu quả cao cho các biến có nhiều giá trị phân loại, bắt được mối quan hệ với biến mục tiêu.</p>
                        <p><strong>Nhược điểm:</strong> Có thể dẫn đến overfitting nếu không áp dụng cross-validation, yêu cầu có biến mục tiêu.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Chọn cột mục tiêu
                        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                        if numeric_columns:
                            target_column = st.selectbox("Chọn cột mục tiêu:", numeric_columns)
                            
                            # Cài đặt nâng cao
                            with st.expander("Tùy chọn nâng cao"):
                                cv_folds = st.slider("Số lượng fold cho cross-validation:", 2, 10, 5)
                                add_noise = st.checkbox("Thêm nhiễu (giảm overfitting)", value=False)
                                if add_noise:
                                    noise_level = st.slider("Mức độ nhiễu:", 0.001, 0.1, 0.01, step=0.001)
                                else:
                                    noise_level = 0.0
                            
                            if st.button("Áp dụng Target Encoding", key="btn_target"):
                                with st.spinner("Đang thực hiện Target Encoding..."):
                                    try:
                                        # Thực hiện encoding
                                        df_encoded, encoders = target_encode(
                                            df, selected_columns, target_column, 
                                            cv=cv_folds, 
                                            add_noise=add_noise, 
                                            noise_level=noise_level
                                        )
                                        
                                        # Hiển thị kết quả
                                        st.success(f"Đã mã hóa Target thành công cho {len(selected_columns)} cột!")
                                        st.write("### Dữ liệu sau khi mã hóa")
                                        st.dataframe(df_encoded.head(10))
                                        
                                        # Hiển thị thông tin encoding
                                        st.write("### Thông tin mã hóa")
                                        st.info(f"Mã hóa sử dụng biến mục tiêu: **{target_column}**")
                                        st.info(f"Số lượng fold cross-validation: **{cv_folds}**")
                                        if add_noise:
                                            st.info(f"Đã thêm nhiễu với mức độ: **{noise_level}**")
                                        
                                        # Trực quan hóa
                                        st.write("### Trực quan hóa kết quả mã hóa")
                                        
                                        # Cho phép chọn cột để trực quan hóa nếu có nhiều cột
                                        if len(selected_columns) > 1:
                                            viz_col = st.selectbox("Chọn cột để trực quan hóa:", selected_columns)
                                        else:
                                            viz_col = selected_columns[0]
                                        
                                        encoded_col = f"{viz_col}_target_encoded"
                                        
                                        # Tạo scatter plot
                                        fig = px.scatter(
                                            df_encoded, 
                                            x=encoded_col, 
                                            y=target_column,
                                            color=viz_col,
                                            title=f"Mối quan hệ giữa mã hóa Target và biến mục tiêu",
                                            hover_data=[viz_col]
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Tạo box plot cho từng category
                                        value_counts = df[viz_col].value_counts()
                                        top_categories = value_counts.nlargest(10).index.tolist()
                                        
                                        if len(top_categories) > 1:
                                            fig = px.box(
                                                df[df[viz_col].isin(top_categories)],
                                                x=viz_col,
                                                y=target_column,
                                                title=f"Phân phối biến mục tiêu theo giá trị phân loại (top 10)"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Hiển thị ánh xạ
                                        if st.checkbox("Hiển thị ánh xạ giá trị -> target mean", value=False):
                                            st.write("### Ánh xạ giá trị phân loại sang giá trị trung bình mục tiêu")
                                            for col, mapping in encoders.items():
                                                if len(mapping) > 20:
                                                    st.write(f"**Cột {col}:** (hiển thị 20/{len(mapping)} giá trị)")
                                                    # Convert to DataFrame for better display
                                                    mapping_df = pd.DataFrame(
                                                        {"Giá trị": list(mapping.keys())[:20], 
                                                         "Target Mean": list(mapping.values())[:20]}
                                                    ).sort_values("Target Mean", ascending=False)
                                                    st.dataframe(mapping_df)
                                                else:
                                                    st.write(f"**Cột {col}:**")
                                                    mapping_df = pd.DataFrame(
                                                        {"Giá trị": list(mapping.keys()), 
                                                         "Target Mean": list(mapping.values())}
                                                    ).sort_values("Target Mean", ascending=False)
                                                    st.dataframe(mapping_df)
                                        
                                        # Tải xuống
                                        st.markdown(get_table_download_link(df_encoded, "target_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                                        
                                    except Exception as e:
                                        st.error(f"Lỗi khi thực hiện Target Encoding: {e}")
                                        st.info("Hãy đảm bảo đã cài đặt thư viện 'category_encoders': pip install category_encoders")
                        else:
                            st.warning("❗ Không có cột số trong dữ liệu để sử dụng làm biến mục tiêu!")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 5: HASH ENCODING
                    with method_tabs[4]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Hash Encoding</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này sử dụng hàm băm để chuyển đổi giá trị phân loại thành chỉ số của một vector có kích thước cố định.</p>
                        <p><strong>Phù hợp với:</strong> Dữ liệu có rất nhiều giá trị phân loại, cần xử lý dữ liệu online hoặc hạn chế bộ nhớ.</p>
                        <p><strong>Ưu điểm:</strong> Xử lý được số lượng giá trị phân loại không giới hạn, không cần lưu trữ từ điển ánh xạ, hiệu quả với dữ liệu streaming.</p>
                        <p><strong>Nhược điểm:</strong> Có thể xảy ra xung đột hash, không thể giải mã ngược, khó debug.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Cài đặt
                        n_components = st.slider("Số lượng thành phần (số cột output):", 2, 32, 8)
                        
                        if st.button("Áp dụng Hash Encoding", key="btn_hash"):
                            with st.spinner("Đang thực hiện Hash Encoding..."):
                                try:
                                    # Thực hiện encoding
                                    df_encoded = hash_encode(df, selected_columns, n_components=n_components)
                                    
                                    # Hiển thị kết quả
                                    st.success(f"Đã mã hóa Hash thành công cho {len(selected_columns)} cột!")
                                    st.write("### Dữ liệu sau khi mã hóa")
                                    st.dataframe(df_encoded.head(10))
                                    
                                    # Thông tin
                                    orig_cols = len(df.columns)
                                    new_cols = len(df_encoded.columns)
                                    st.info(f"Kích thước dữ liệu ban đầu: {orig_cols} cột → Kích thước sau mã hóa: {new_cols} cột")
                                    st.info(f"Số lượng cột hash được tạo: {new_cols - orig_cols}")
                                    
                                    # Trực quan hóa
                                    st.write("### Trực quan hóa kết quả mã hóa")
                                    
                                    if len(selected_columns) == 1:
                                        col = selected_columns[0]
                                        hash_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
                                        
                                        # Hiển thị ma trận hash cho các mẫu đầu tiên
                                        fig = px.imshow(
                                            df_encoded[hash_cols].head(20),
                                            title=f"Ma trận mã hóa hash (20 mẫu đầu tiên)",
                                            color_continuous_scale="Viridis"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Hiển thị phân phối giá trị
                                        value_counts = df[col].value_counts().nlargest(10)
                                        fig = px.bar(
                                            x=value_counts.index, 
                                            y=value_counts.values,
                                            title=f"Top 10 giá trị phổ biến nhất của cột {col}"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Tải xuống
                                    st.markdown(get_table_download_link(df_encoded, "hash_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                                    
                                    # Lưu ý về xung đột hash
                                    st.warning("⚠️ Lưu ý: Hash encoding có thể gây ra xung đột, làm nhiều giá trị phân loại khác nhau được ánh xạ vào cùng một chỉ số. Tăng số lượng thành phần sẽ giảm khả năng xung đột.")
                                
                                except Exception as e:
                                    st.error(f"Lỗi khi thực hiện Hash Encoding: {e}")
                                    st.info("Hãy đảm bảo đã cài đặt thư viện 'category_encoders': pip install category_encoders")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 6: KẾT HỢP NHIỀU PHƯƠNG PHÁP
                    with method_tabs[5]:
                        st.markdown('<div class="encoding-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Kết hợp nhiều phương pháp mã hóa</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Kết hợp nhiều phương pháp mã hóa khác nhau cho các cột khác nhau, tùy theo đặc tính của từng cột và yêu cầu của bài toán.</p>
                        <p><strong>Phù hợp với:</strong> Khi các cột phân loại có đặc tính khác nhau, hoặc cần tối ưu hiệu suất của mô hình.</p>
                        <p><strong>Ưu điểm:</strong> Linh hoạt, tận dụng điểm mạnh của từng phương pháp cho từng cột dữ liệu.</p>
                        <p><strong>Nhược điểm:</strong> Phức tạp hơn, cần lưu trữ nhiều thông tin về quá trình mã hóa.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tạo bảng lựa chọn phương pháp cho từng cột
                        st.write("### Chọn phương pháp encoding cho từng cột")
                        
                        encoding_choices = {}
                        encoding_options = {
                            "One-Hot": "one_hot",
                            "Label": "label",
                            "Binary": "binary",
                            "Target": "target",
                            "Hash": "hash"
                        }
                        
                        # Tạo DataFrame để hiển thị lựa chọn
                        choice_data = []
                        for col in selected_columns:
                            selected_method = st.selectbox(
                                f"Chọn phương pháp mã hóa cho cột **{col}**:",
                                list(encoding_options.keys()),
                                key=f"combo_{col}"
                            )
                            encoding_choices[col] = encoding_options[selected_method]
                            choice_data.append({"Cột": col, "Phương pháp mã hóa": selected_method})
                        
                        # Hiển thị bảng lựa chọn
                        choice_df = pd.DataFrame(choice_data)
                        st.dataframe(choice_df)
                        
                        # Cài đặt cho từng phương pháp
                        with st.expander("Cài đặt cho các phương pháp mã hóa"):
                            # One-Hot settings
                            st.write("#### Cài đặt One-Hot Encoding")
                            drop_first_combo = st.checkbox("Loại bỏ cột đầu tiên", value=False, key="combo_drop_first")
                            
                            # Target settings
                            if "target" in encoding_choices.values():
                                st.write("#### Cài đặt Target Encoding")
                                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                                if numeric_columns:
                                    target_column = st.selectbox(
                                        "Chọn cột mục tiêu cho Target Encoding:",
                                        numeric_columns,
                                        key="combo_target"
                                    )
                                    cv_folds = st.slider(
                                        "Số lượng fold cho cross-validation:",
                                        2, 10, 5,
                                        key="combo_cv"
                                    )
                                    add_noise_combo = st.checkbox("Thêm nhiễu", value=False, key="combo_noise")
                                    noise_level_combo = 0.01
                                    if add_noise_combo:
                                        noise_level_combo = st.slider(
                                            "Mức độ nhiễu:",
                                            0.001, 0.1, 0.01, step=0.001,
                                            key="combo_noise_level"
                                        )
                                else:
                                    st.warning("Không có cột số trong dữ liệu để sử dụng làm biến mục tiêu!")
                            
                            # Hash settings
                            if "hash" in encoding_choices.values():
                                st.write("#### Cài đặt Hash Encoding")
                                n_components_combo = st.slider(
                                    "Số lượng thành phần cho Hash Encoding:",
                                    2, 32, 8,
                                    key="combo_hash"
                                )
                        
                        if st.button("Áp dụng encoding kết hợp", key="btn_combo"):
                            with st.spinner("Đang thực hiện encoding kết hợp..."):
                                try:
                                    # Khởi tạo DataFrame kết quả
                                    df_result = df.copy()
                                    encoding_report = []
                                    
                                    # Áp dụng từng phương pháp cho từng cột
                                    for col, method in encoding_choices.items():
                                        if method == "one_hot":
                                            # One-hot encoding
                                            df_onehot = one_hot_encode(df_result[[col]], [col])
                                            # Loại bỏ cột gốc và thêm cột mới
                                            df_result = df_result.drop(columns=[col])
                                            df_result = pd.concat([df_result, df_onehot], axis=1)
                                            encoding_report.append(f"Cột {col}: One-Hot Encoding")
                                        
                                        elif method == "label":
                                            # Label encoding
                                            df_label, encoders = label_encode(df_result, [col])
                                            # Thêm cột mới
                                            df_result[f"{col}_label_encoded"] = df_label[f"{col}_label_encoded"]
                                            encoding_report.append(f"Cột {col}: Label Encoding")
                                        
                                        elif method == "binary":
                                            # Binary encoding
                                            df_binary = binary_encode(df_result, [col])
                                            # Loại bỏ các cột từ df_binary đã có trong df_result
                                            new_cols = [c for c in df_binary.columns if c not in df_result.columns]
                                            df_result = pd.concat([df_result, df_binary[new_cols]], axis=1)
                                            encoding_report.append(f"Cột {col}: Binary Encoding")
                                        
                                        elif method == "target":
                                            # Target encoding
                                            if "target_column" in locals() and numeric_columns:
                                                df_target, _ = target_encode(
                                                    df_result, [col], target_column, 
                                                    cv=cv_folds,
                                                    add_noise=add_noise_combo,
                                                    noise_level=noise_level_combo
                                                )
                                                # Thêm cột mới
                                                df_result[f"{col}_target_encoded"] = df_target[f"{col}_target_encoded"]
                                                encoding_report.append(f"Cột {col}: Target Encoding (biến mục tiêu: {target_column})")
                                        
                                        elif method == "hash":
                                            # Hash encoding
                                            if "n_components_combo" in locals():
                                                df_hash = hash_encode(df_result, [col], n_components=n_components_combo)
                                                # Loại bỏ các cột từ df_hash đã có trong df_result
                                                new_cols = [c for c in df_hash.columns if c not in df_result.columns]
                                                df_result = pd.concat([df_result, df_hash[new_cols]], axis=1)
                                                encoding_report.append(f"Cột {col}: Hash Encoding ({n_components_combo} thành phần)")
                                    
                                    # Hiển thị kết quả
                                    st.success("Đã mã hóa kết hợp thành công!")
                                    
                                    # Báo cáo encoding
                                    st.write("### Báo cáo mã hóa")
                                    for report in encoding_report:
                                        st.write(f"✅ {report}")
                                    
                                    # Hiển thị dữ liệu
                                    st.write("### Dữ liệu sau khi mã hóa")
                                    st.dataframe(df_result.head(10))
                                    
                                    # Thông tin kích thước
                                    st.info(f"Kích thước dữ liệu ban đầu: {df.shape[1]} cột → Kích thước sau mã hóa: {df_result.shape[1]} cột")
                                    
                                    # Tải xuống
                                    st.markdown(get_table_download_link(df_result, "combined_encoded_data.csv", "📥 Tải xuống dữ liệu đã mã hóa (CSV)"), unsafe_allow_html=True)
                                
                                except Exception as e:
                                    st.error(f"Lỗi khi thực hiện mã hóa kết hợp: {e}")
                                    st.info("Hãy đảm bảo đã cài đặt các thư viện cần thiết: pip install category_encoders scikit-learn")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Vui lòng chọn ít nhất một cột để mã hóa.")
            
            else:
                st.warning("⚠️ Không tìm thấy cột phân loại (dạng object/category) trong dữ liệu!")
                st.info("💡 Nếu dữ liệu của bạn có cột phân loại nhưng không được phát hiện, hãy kiểm tra kiểu dữ liệu của các cột.")
                
                # Hiển thị kiểu dữ liệu
                st.write("### Kiểu dữ liệu của các cột:")
                dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
                dtypes_df.index.name = "Column"
                st.dataframe(dtypes_df.reset_index())
                
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý dữ liệu: {e}")
            st.info("💡 Hãy kiểm tra định dạng file CSV và thử lại.")
    
    else:
        # Hiển thị hướng dẫn khi chưa có dữ liệu
        # st.markdown("""
        # <div class="info-box" style="text-align: center;">
        #     <h3>👋 Chào mừng đến với công cụ Mã hóa dữ liệu</h3>
        #     <p>Hãy tải lên file CSV chứa dữ liệu cần mã hóa để bắt đầu.</p>
        #     <p>Công cụ này hỗ trợ các phương pháp mã hóa phổ biến:</p>
        #     <ul style="list-style-type: none; text-align: left; display: inline-block;">
        #         <li>✅ One-Hot Encoding</li>
        #         <li>✅ Label Encoding</li>
        #         <li>✅ Binary Encoding</li>
        #         <li>✅ Target Encoding</li>
        #         <li>✅ Hash Encoding</li>
        #     </ul>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Hiển thị ví dụ
        with st.expander("📊 Xem ví dụ dữ liệu"):
            example_data = {
                "category": ["A", "B", "A", "C", "B", "C", "A", "D", "B", "D"],
                "color": ["red", "blue", "red", "green", "blue", "yellow", "red", "green", "yellow", "blue"],
                "city": ["Hanoi", "HCM", "Danang", "Hanoi", "HCM", "Hanoi", "HCM", "Danang", "Hanoi", "HCM"],
                "value": [10, 20, 15, 25, 22, 30, 12, 28, 18, 24]
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            
            # Tạo button tải ví dụ
            if st.button("📥 Tải xuống dữ liệu ví dụ"):
                st.markdown(get_table_download_link(example_df, "example_data.csv", "📥 Tải xuống CSV"), unsafe_allow_html=True)


app()
