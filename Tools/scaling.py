import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Thiết kế trang
def designPage():
    st.set_page_config(
        page_title="Chuẩn hóa dữ liệu cho Machine Learning",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://example.com/help',
            'Report a bug': 'https://example.com/bug',
            'About': 'Công cụ chuẩn hóa dữ liệu cho Machine Learning'
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
        background: linear-gradient(-45deg, #2ecc71, #3498db, #1abc9c, #f39c12);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 6s ease infinite;
        margin-bottom: 20px;
    }
    .scaling-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .scaling-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #2ecc71;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        font-weight: 500;
    }
    .info-box h3 {
        color: #16a085;
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
        color: #2ecc71;
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
        color: #000000;
        font-weight: bold;
    }
    .download-button {
        display: inline-block;
        padding: 8px 16px;
        background-color: #2ecc71;
        color: white;
        border-radius: 4px;
        text-decoration: none;
        font-weight: bold;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #27ae60;
    }
    </style>
    <p class="animated-gradient-title">Chuẩn hóa dữ liệu (Scaling) cho Machine Learning</p>
    """, unsafe_allow_html=True)

# Hàm tạo link tải xuống
def get_table_download_link(df, filename="data.csv", text="Tải xuống dữ liệu"):
    """Tạo link tải xuống cho DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# Các hàm scaling
def min_max_scale(df, columns, feature_range=(0, 1)):
    """Thực hiện Min-Max Scaling"""
    df_scaled = df.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    # Lưu thông tin scaling
    scaling_info = {
        'min_values': scaler.data_min_.tolist(),
        'max_values': scaler.data_max_.tolist(),
        'scale_values': scaler.scale_.tolist(),
        'min_bound': feature_range[0],
        'max_bound': feature_range[1]
    }
    
    return df_scaled, scaling_info

def standardize(df, columns):
    """Thực hiện Standardization (Z-score normalization)"""
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    # Lưu thông tin scaling
    scaling_info = {
        'mean_values': scaler.mean_.tolist(),
        'std_values': scaler.scale_.tolist()
    }
    
    return df_scaled, scaling_info

def robust_scale(df, columns, quantile_range=(25.0, 75.0)):
    """Thực hiện Robust Scaling sử dụng median và IQR"""
    df_scaled = df.copy()
    scaler = RobustScaler(quantile_range=quantile_range)
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    # Lưu thông tin scaling
    scaling_info = {
        'center_values': scaler.center_.tolist(),
        'scale_values': scaler.scale_.tolist(),
        'quantile_range': quantile_range
    }
    
    return df_scaled, scaling_info

# Hiển thị phân phối dữ liệu trước và sau khi scale
def plot_distribution_comparison(original_df, scaled_df, columns, method_name):
    num_cols = min(3, len(columns))  # Giới hạn số cột để trực quan hóa
    
    figs = []
    for i, col in enumerate(columns[:num_cols]):
        fig = plt.figure(figsize=(15, 5))
        
        # Biểu đồ phân phối gốc
        plt.subplot(1, 2, 1)
        sns.histplot(original_df[col], kde=True)
        plt.title(f"Phân phối ban đầu: {col}")
        plt.xlabel(col)
        
        # Biểu đồ phân phối sau khi scale
        plt.subplot(1, 2, 2)
        sns.histplot(scaled_df[col], kde=True, color='green')
        plt.title(f"Phân phối sau {method_name}: {col}")
        plt.xlabel(f"{col} (Scaled)")
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs

# Hiển thị biểu đồ scatter
def create_scatter_plot(df, x_col, y_col, title, color=None):
    if color:
        fig = px.scatter(df, x=x_col, y=y_col, color=color, opacity=0.7,
                       title=title)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, opacity=0.7, title=title)
    
    return fig

def app():
    # Thiết kế trang
    designPage()
    
    # Hiển thị tiêu đề
    designTitle()
    
    # Giới thiệu về scaling
    with st.expander("💡 Giới thiệu về Scaling trong Machine Learning", expanded=False):
        st.markdown("""
        <div class="info-box">
        <h3>Tại sao cần Scaling?</h3>
        <p>Scaling (chuẩn hóa) là quá trình biến đổi dữ liệu số sao cho chúng có phạm vi hay phân phối tương đồng. Điều này giúp các thuật toán học máy hoạt động hiệu quả hơn và tránh bị ảnh hưởng bởi các đặc trưng có thang đo khác nhau.</p>
        
        <h3>Các phương pháp Scaling</h3>
        <ul>
            <li><strong>Min-Max Scaling:</strong> Biến đổi dữ liệu vào khoảng [0,1] hoặc bất kỳ phạm vi nào. Nhạy cảm với outliers.</li>
            <li><strong>Standardization (Z-score):</strong> Biến đổi dữ liệu có trung bình 0 và độ lệch chuẩn 1. Phù hợp khi dữ liệu có phân phối gần chuẩn.</li>
            <li><strong>Robust Scaling:</strong> Sử dụng trung vị (median) và khoảng tứ phân vị (IQR) để scale. Ít nhạy cảm với outliers.</li>
        </ul>
        
        <h3>Khi nào cần áp dụng Scaling?</h3>
        <ul>
            <li>Khi sử dụng các thuật toán dựa trên khoảng cách (K-means, KNN, SVM...)</li>
            <li>Khi các đặc trưng có thang đo (scale) khác nhau</li>
            <li>Khi sử dụng gradient descent và neural networks</li>
            <li>Khi dữ liệu có phân phối lệch hoặc có outliers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload dữ liệu
    st.markdown('<h2 style="color:#2ecc71;"><span class="feature-icon">📤</span> Tải lên dữ liệu</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu cần chuẩn hóa", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Đọc dữ liệu
            df = pd.read_csv(uploaded_file)
            
            # Hiển thị thông tin dữ liệu
            st.markdown('<h2 style="color:#2ecc71;"><span class="feature-icon">📊</span> Thông tin dữ liệu</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Số hàng:** {df.shape[0]}")
                st.write(f"**Số cột:** {df.shape[1]}")
            
            with col2:
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                st.write(f"**Số cột số:** {len(numeric_columns)}")
                st.write(f"**Các cột số:** {', '.join(numeric_columns) if numeric_columns else 'Không có'}")
                
            # Hiển thị dữ liệu
            st.markdown('<div class="scaling-card">', unsafe_allow_html=True)
            st.write("### Dữ liệu đã tải lên")
            st.dataframe(df.head(10))
            
            # Hiển thị thống kê mô tả
            with st.expander("Xem thống kê mô tả"):
                st.write("### Thống kê mô tả")
                st.dataframe(df.describe())
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Phần chuẩn hóa dữ liệu
            if numeric_columns:
                st.markdown('<h2 style="color:#2ecc71;"><span class="feature-icon">🔄</span> Chuẩn hóa dữ liệu</h2>', unsafe_allow_html=True)
                
                # Chọn cột cần chuẩn hóa
                selected_columns = st.multiselect(
                    "Chọn các cột cần chuẩn hóa:",
                    numeric_columns,
                    default=numeric_columns[:min(3, len(numeric_columns))]
                )
                
                if selected_columns:
                    # Tạo tabs cho các phương pháp scaling
                    method_tabs = st.tabs([
                        "Min-Max Scaling", 
                        "Standardization (Z-score)", 
                        "Robust Scaling"
                    ])
                    
                    # TAB 1: MIN-MAX SCALING
                    with method_tabs[0]:
                        st.markdown('<div class="scaling-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Min-Max Scaling</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này biến đổi dữ liệu vào một khoảng cố định, thường là [0, 1].</p>
                        <p><strong>Công thức:</strong> X_scaled = (X - X_min) / (X_max - X_min) * (max_bound - min_bound) + min_bound</p>
                        <p><strong>Phù hợp với:</strong> Hầu hết các thuật toán, đặc biệt là neural networks và các thuật toán yêu cầu dữ liệu dương.</p>
                        <p><strong>Ưu điểm:</strong> Dễ hiểu, giữ nguyên phân phối của dữ liệu (trừ việc co giãn).</p>
                        <p><strong>Nhược điểm:</strong> Rất nhạy cảm với outliers, có thể dẫn đến giảm độ rộng của phân phối dữ liệu chính.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tùy chọn
                        col1, col2 = st.columns(2)
                        with col1:
                            min_bound = st.number_input("Giá trị min:", value=0.0, step=0.1)
                        with col2:
                            max_bound = st.number_input("Giá trị max:", value=1.0, step=0.1)
                        
                        if st.button("Áp dụng Min-Max Scaling", key="btn_minmax"):
                            with st.spinner("Đang thực hiện Min-Max Scaling..."):
                                # Thực hiện scaling
                                feature_range = (min_bound, max_bound)
                                df_scaled, scaling_info = min_max_scale(df, selected_columns, feature_range)
                                
                                # Hiển thị kết quả
                                st.success(f"Đã chuẩn hóa Min-Max thành công cho {len(selected_columns)} cột!")
                                st.write("### Dữ liệu sau khi chuẩn hóa")
                                st.dataframe(df_scaled.head(10))
                                
                                # Hiển thị thông tin scaling
                                with st.expander("Xem thông tin chuẩn hóa"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("### Giá trị tối thiểu ban đầu")
                                        min_df = pd.DataFrame({'Column': selected_columns, 'Min Value': scaling_info['min_values']})
                                        st.dataframe(min_df)
                                    
                                    with col2:
                                        st.write("### Giá trị tối đa ban đầu")
                                        max_df = pd.DataFrame({'Column': selected_columns, 'Max Value': scaling_info['max_values']})
                                        st.dataframe(max_df)
                                
                                # Trực quan hóa kết quả
                                st.write("### Trực quan hóa kết quả")
                                figs = plot_distribution_comparison(df, df_scaled, selected_columns, "Min-Max Scaling")
                                for fig in figs:
                                    st.pyplot(fig)
                                
                                # Nếu có 2 cột trở lên, hiển thị biểu đồ scatter
                                if len(selected_columns) >= 2:
                                    st.write("### So sánh biểu đồ phân tán trước và sau khi scaling")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("Dữ liệu gốc")
                                        fig1 = create_scatter_plot(
                                            df, selected_columns[0], selected_columns[1],
                                            f"Dữ liệu gốc: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig1, use_container_width=True)
                                    
                                    with col2:
                                        st.write("Dữ liệu sau khi scaling")
                                        fig2 = create_scatter_plot(
                                            df_scaled, selected_columns[0], selected_columns[1],
                                            f"Sau scaling: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                
                                # Hiển thị thống kê mô tả sau khi scale
                                with st.expander("Xem thống kê sau khi chuẩn hóa"):
                                    st.write("### Thống kê mô tả sau khi chuẩn hóa")
                                    st.dataframe(df_scaled[selected_columns].describe())
                                
                                # Tải xuống
                                st.markdown(get_table_download_link(df_scaled, "minmax_scaled_data.csv", "📥 Tải xuống dữ liệu đã chuẩn hóa (CSV)"), unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 2: STANDARDIZATION
                    with method_tabs[1]:
                        st.markdown('<div class="scaling-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Standardization (Z-score)</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này biến đổi dữ liệu sao cho có trung bình bằng 0 và độ lệch chuẩn bằng 1.</p>
                        <p><strong>Công thức:</strong> X_scaled = (X - μ) / σ</p>
                        <p><strong>Phù hợp với:</strong> Các thuật toán như SVM, Linear/Logistic Regression, Neural Network và các thuật toán giả định dữ liệu có phân phối chuẩn.</p>
                        <p><strong>Ưu điểm:</strong> Giúp các thuật toán hội tụ nhanh hơn, ít bị ảnh hưởng bởi đơn vị đo.</p>
                        <p><strong>Nhược điểm:</strong> Không đưa dữ liệu về một phạm vi cố định, vẫn nhạy cảm với outliers.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Áp dụng Standardization", key="btn_standard"):
                            with st.spinner("Đang thực hiện Standardization..."):
                                # Thực hiện scaling
                                df_scaled, scaling_info = standardize(df, selected_columns)
                                
                                # Hiển thị kết quả
                                st.success(f"Đã chuẩn hóa Standardization thành công cho {len(selected_columns)} cột!")
                                st.write("### Dữ liệu sau khi chuẩn hóa")
                                st.dataframe(df_scaled.head(10))
                                
                                # Hiển thị thông tin scaling
                                with st.expander("Xem thông tin chuẩn hóa"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("### Giá trị trung bình ban đầu")
                                        mean_df = pd.DataFrame({'Column': selected_columns, 'Mean': scaling_info['mean_values']})
                                        st.dataframe(mean_df)
                                    
                                    with col2:
                                        st.write("### Độ lệch chuẩn ban đầu")
                                        std_df = pd.DataFrame({'Column': selected_columns, 'Std Dev': scaling_info['std_values']})
                                        st.dataframe(std_df)
                                
                                # Trực quan hóa kết quả
                                st.write("### Trực quan hóa kết quả")
                                figs = plot_distribution_comparison(df, df_scaled, selected_columns, "Standardization")
                                for fig in figs:
                                    st.pyplot(fig)
                                
                                # Nếu có 2 cột trở lên, hiển thị biểu đồ scatter
                                if len(selected_columns) >= 2:
                                    st.write("### So sánh biểu đồ phân tán trước và sau khi scaling")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("Dữ liệu gốc")
                                        fig1 = create_scatter_plot(
                                            df, selected_columns[0], selected_columns[1],
                                            f"Dữ liệu gốc: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig1, use_container_width=True)
                                    
                                    with col2:
                                        st.write("Dữ liệu sau khi scaling")
                                        fig2 = create_scatter_plot(
                                            df_scaled, selected_columns[0], selected_columns[1],
                                            f"Sau scaling: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                
                                # Hiển thị thống kê mô tả sau khi scale
                                with st.expander("Xem thống kê sau khi chuẩn hóa"):
                                    st.write("### Thống kê mô tả sau khi chuẩn hóa")
                                    st.dataframe(df_scaled[selected_columns].describe())
                                
                                # Tải xuống
                                st.markdown(get_table_download_link(df_scaled, "standardized_data.csv", "📥 Tải xuống dữ liệu đã chuẩn hóa (CSV)"), unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # TAB 3: ROBUST SCALING
                    with method_tabs[2]:
                        st.markdown('<div class="scaling-card">', unsafe_allow_html=True)
                        st.markdown('<h3 class="method-title">Robust Scaling</h3>', unsafe_allow_html=True)
                        
                        # Giải thích
                        st.markdown("""
                        <div class="info-box">
                        <p>Phương pháp này sử dụng trung vị (median) và khoảng tứ phân vị (IQR) thay vì trung bình và độ lệch chuẩn.</p>
                        <p><strong>Công thức:</strong> X_scaled = (X - median) / IQR</p>
                        <p><strong>Phù hợp với:</strong> Dữ liệu có outliers, các thuật toán nhạy cảm với outliers.</p>
                        <p><strong>Ưu điểm:</strong> Ít bị ảnh hưởng bởi outliers, giữ được thông tin về điểm dữ liệu ngoại lai.</p>
                        <p><strong>Nhược điểm:</strong> Không đưa dữ liệu về phạm vi cố định, không phổ biến bằng các phương pháp khác.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tùy chọn
                        col1, col2 = st.columns(2)
                        with col1:
                            q_low = st.number_input("Phân vị dưới (%):", value=25.0, min_value=0.0, max_value=49.0, step=5.0)
                        with col2:
                            q_high = st.number_input("Phân vị trên (%):", value=75.0, min_value=51.0, max_value=100.0, step=5.0)
                        
                        if st.button("Áp dụng Robust Scaling", key="btn_robust"):
                            with st.spinner("Đang thực hiện Robust Scaling..."):
                                # Thực hiện scaling
                                quantile_range = (q_low, q_high)
                                df_scaled, scaling_info = robust_scale(df, selected_columns, quantile_range)
                                
                                # Hiển thị kết quả
                                st.success(f"Đã chuẩn hóa Robust thành công cho {len(selected_columns)} cột!")
                                st.write("### Dữ liệu sau khi chuẩn hóa")
                                st.dataframe(df_scaled.head(10))
                                
                                # Hiển thị thông tin scaling
                                with st.expander("Xem thông tin chuẩn hóa"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("### Giá trị trung vị ban đầu")
                                        median_df = pd.DataFrame({'Column': selected_columns, 'Median': scaling_info['center_values']})
                                        st.dataframe(median_df)
                                    
                                    with col2:
                                        st.write("### Khoảng tứ phân vị ban đầu")
                                        iqr_df = pd.DataFrame({'Column': selected_columns, 'IQR': scaling_info['scale_values']})
                                        st.dataframe(iqr_df)
                                
                                # Trực quan hóa kết quả
                                st.write("### Trực quan hóa kết quả")
                                figs = plot_distribution_comparison(df, df_scaled, selected_columns, "Robust Scaling")
                                for fig in figs:
                                    st.pyplot(fig)
                                
                                # Nếu có 2 cột trở lên, hiển thị biểu đồ scatter
                                if len(selected_columns) >= 2:
                                    st.write("### So sánh biểu đồ phân tán trước và sau khi scaling")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("Dữ liệu gốc")
                                        fig1 = create_scatter_plot(
                                            df, selected_columns[0], selected_columns[1],
                                            f"Dữ liệu gốc: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig1, use_container_width=True)
                                    
                                    with col2:
                                        st.write("Dữ liệu sau khi scaling")
                                        fig2 = create_scatter_plot(
                                            df_scaled, selected_columns[0], selected_columns[1],
                                            f"Sau scaling: {selected_columns[0]} vs {selected_columns[1]}"
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                
                                # Hiển thị thống kê mô tả sau khi scale
                                with st.expander("Xem thống kê sau khi chuẩn hóa"):
                                    st.write("### Thống kê mô tả sau khi chuẩn hóa")
                                    st.dataframe(df_scaled[selected_columns].describe())
                                
                                # Tải xuống
                                st.markdown(get_table_download_link(df_scaled, "robust_scaled_data.csv", "📥 Tải xuống dữ liệu đã chuẩn hóa (CSV)"), unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Vui lòng chọn ít nhất một cột để chuẩn hóa.")
            
            else:
                st.warning("⚠️ Không tìm thấy cột số (numeric) trong dữ liệu!")
                st.info("💡 Scaling chỉ có thể được áp dụng cho dữ liệu số. Hãy kiểm tra kiểu dữ liệu của các cột trong file CSV của bạn.")
                
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
        
        # Hiển thị ví dụ
        with st.expander("📊 Xem ví dụ dữ liệu"):
            example_data = {
                "age": [25, 30, 45, 60, 35, 25, 30, 85, 40, 30],
                "income": [45000, 72000, 150000, 95000, 67000, 52000, 80000, 120000, 95000, 62000],
                "experience": [1, 5, 15, 30, 10, 2, 7, 40, 15, 5],
                "expenses": [30000, 45000, 90000, 60000, 40000, 35000, 55000, 75000, 60000, 42000],
                "category": ["A", "B", "A", "C", "B", "A", "B", "C", "C", "A"]
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            
            # Tạo button tải ví dụ
            if st.button("📥 Tải xuống dữ liệu ví dụ"):
                st.markdown(get_table_download_link(example_df, "example_data.csv", "📥 Tải xuống CSV"), unsafe_allow_html=True)


app()