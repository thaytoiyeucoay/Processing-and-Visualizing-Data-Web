import streamlit as st
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Data Processing & Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the entire app
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(to bottom, #0F172A, #1E293B);
        color: white;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Button styling with gradient and animation */
    .stButton button {
        background: linear-gradient(45deg, #6366F1, #A855F7);
        border: none;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.7);
        background: linear-gradient(45deg, #4F46E5, #9333EA);
    }
    
    /* Card styling for content sections */
    .css-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Custom input styling */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        color: white;
        padding: 8px 12px;
    }
    
    .stTextInput input:focus {
        border-color: #6366F1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Social media buttons */
    .social-buttons {
        display: flex;
        gap: 10px;
    }
    
    .social-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        transition: all 0.3s ease;
    }
    
    .social-button:hover {
        transform: translateY(-3px);
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Author section */
    .author-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .author-image {
        border-radius: 50%;
        border: 2px solid #6366F1;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Footer styling */
    footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Info box styling */
    .info-box {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366F1;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
        margin: 15px 0;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        background-color: rgba(99, 102, 241, 0.2);
        color: white;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
    }
    
    /* Progress bar styling */
    .custom-progress {
        height: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        margin-bottom: 10px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(45deg, #6366F1, #A855F7);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; padding-top: 20px;'>Chào mừng đến với Website Xử lý và Trực quan hóa Dữ liệu</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Khám phá những công cụ phân tích dữ liệu tiên tiến và trực quan hóa thông tin mạnh mẽ.</p>", unsafe_allow_html=True)

# Main image (placeholder)
st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop", 
         caption="Data Visualization", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Main content section
st.markdown("<div class='css-card'>", unsafe_allow_html=True)
st.markdown("""
Trang web của chúng tôi cung cấp các giải pháp phân tích và xử lý dữ liệu từ trực quan hóa số liệu
cho đến dự báo xu hướng, giúp bạn đưa ra quyết định dựa trên dữ liệu một cách nhanh chóng và
chính xác.
""")
st.markdown("</div>", unsafe_allow_html=True)

# Features section with custom info boxes
st.markdown("<h2>Các tính năng chính</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Tìm kiếm dữ liệu")
    st.markdown("Hỗ trợ tìm kiếm các bộ dữ liệu nhỏ và vừa theo từ khóa dựa trên Kaggle")
    
    # Custom progress indicator
    st.markdown("""
    <div class='info-box'>
        <span class='badge'>Mới</span> FIND DATA THAT YOU WANT 
    </div>
    <p>Độ chính xác</p>
    <div class='custom-progress'>
        <div class='progress-bar' style='width: 85%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Trực quan hóa")
    st.markdown("Tạo biểu đồ tương tác và bảng điều khiển trực quan để hiểu rõ hơn về dữ liệu của bạn.")
    
    # Custom progress indicator
    st.markdown("""
    <div class='info-box'>
        <span class='badge'>Hot</span> Interactive Dashboards
    </div>
    <p>Tính năng</p>
    <div class='custom-progress'>
        <div class='progress-bar' style='width: 92%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 Làm sạch dữ liệu của bạn hơn")
    st.markdown("Áp dụng các phương pháp tiền xử lý dữ liệu như điền khuyết, tách số,... để dữ liệu sạch đẹp hơn")
    
    # Custom progress indicator
    st.markdown("""
    <div class='info-box'>
        <span class='badge'>Pro</span> Cleaning and Processing Data
    </div>
    <p>Hiệu suất</p>
    <div class='custom-progress'>
        <div class='progress-bar' style='width: 78%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
with col4:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("#### Và nhiều tính năng khác")
    st.markdown("Có thêm nhiều tính năng mới giúp người dùng làm việc được với dữ liệu nhiều hơn đang được phát triển")
    
    # Custom progress indicator
    st.markdown("""
    <div class='info-box'>
        <span class='badge'>Pro</span> Coming Soon...
    </div>
    <p>Tiến độ</p>
    <div class='custom-progress'>
        <div class='progress-bar' style='width: 70%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)    

# Get Started button
st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
if st.button("Bắt đầu ngay"):
    st.balloons()
    st.session_state.home = True
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Author section
st.markdown("<h2>Về tác giả</h2>", unsafe_allow_html=True)
#st.markdown("<div class='css-card'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    # Author profile image
    st.markdown("<div class='author-section'>", unsafe_allow_html=True)
    st.image("project\IMG_7940.JPG", width=200)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3>Bùi Khánh Duy</h3>", unsafe_allow_html=True)
    st.markdown("""
    Sinh viên ngành Toán - tin K67, Đại học Bách Khoa Hà Nội.
    """)
    
    # Social media links
    st.markdown("<div class='social-buttons'>", unsafe_allow_html=True)
    
    # Facebook link
    st.markdown(f"""
    <a href="https://www.facebook.com/duydangbuon1605/" target="_blank" style="text-decoration: none;">
        <div class="social-button">
            <svg width="20" height="20" fill="white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2.04C6.5 2.04 2 6.53 2 12.06C2 17.06 5.66 21.21 10.44 21.96V14.96H7.9V12.06H10.44V9.85C10.44 7.34 11.93 5.96 14.22 5.96C15.31 5.96 16.45 6.15 16.45 6.15V8.62H15.19C13.95 8.62 13.56 9.39 13.56 10.18V12.06H16.34L15.89 14.96H13.56V21.96C18.34 21.21 22 17.06 22 12.06C22 6.53 17.5 2.04 12 2.04Z"/>
            </svg>
        </div>
    </a>
    """, unsafe_allow_html=True)
    
    # Github link
    st.markdown(f"""
    <a href="https://github.com/thaytoiyeucoay" target="_blank" style="text-decoration: none;">
        <div class="social-button">
            <svg width="20" height="20" fill="white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.477 2 2 6.477 2 12C2 16.419 4.865 20.166 8.839 21.489C9.339 21.581 9.533 21.28 9.533 21.017C9.533 20.781 9.523 19.988 9.523 19.194C7 19.694 6.35 18.613 6.15 18.033C6.037 17.745 5.537 16.816 5.125 16.581C4.787 16.393 4.287 15.893 5.125 15.881C5.912 15.868 6.487 16.613 6.675 16.901C7.55 18.352 8.937 18.072 9.573 17.808C9.665 17.172 9.937 16.739 10.237 16.489C7.988 16.239 5.65 15.372 5.65 11.613C5.65 10.49 6.037 9.573 6.7 8.859C6.6 8.609 6.25 7.619 6.8 6.219C6.8 6.219 7.612 5.956 9.525 7.275C10.325 7.037 11.175 6.919 12.025 6.919C12.875 6.919 13.725 7.037 14.525 7.275C16.437 5.956 17.25 6.219 17.25 6.219C17.8 7.619 17.45 8.609 17.35 8.859C18.012 9.573 18.4 10.49 18.4 11.613C18.4 15.382 16.062 16.239 13.812 16.489C14.189 16.803 14.512 17.417 14.512 18.369C14.512 19.725 14.501 20.683 14.501 21.017C14.501 21.28 14.695 21.593 15.195 21.489C17.168 20.821 18.885 19.538 20.101 17.818C21.316 16.098 21.995 14.01 22 11.875C22 6.477 17.523 2 12 2Z"/>
            </svg>
        </div>
    </a>
    """, unsafe_allow_html=True)
    
    # LinkedIn link
    st.markdown(f"""
    <a href="https://www.linkedin.com/in/khanhduyhust160804/" target="_blank" style="text-decoration: none;">
        <div class="social-button">
            <svg width="20" height="20" fill="white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 3A2 2 0 0 1 21 5V19A2 2 0 0 1 19 21H5A2 2 0 0 1 3 19V5A2 2 0 0 1 5 3H19M18.5 18.5V13.2A3.26 3.26 0 0 0 15.24 9.94C14.39 9.94 13.4 10.46 12.92 11.24V10.13H10.13V18.5H12.92V13.57C12.92 12.8 13.54 12.17 14.31 12.17A1.4 1.4 0 0 1 15.71 13.57V18.5H18.5M6.88 8.56A1.68 1.68 0 0 0 8.56 6.88C8.56 5.95 7.81 5.19 6.88 5.19A1.69 1.69 0 0 0 5.19 6.88C5.19 7.81 5.95 8.56 6.88 8.56M8.27 18.5V10.13H5.5V18.5H8.27Z"/>
            </svg>
        </div>
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# # Feature tabs (just the UI, no data)
# st.markdown("<h2>Tính năng của chúng tôi</h2>", unsafe_allow_html=True)
# st.markdown("<div class='css-card'>", unsafe_allow_html=True)

# tabs = st.tabs(["Biểu đồ thống kê", "Dự đoán xu hướng", "Phân tích văn bản"])

# with tabs[0]:
#     st.markdown("### Biểu đồ thống kê")
#     st.markdown("""
#     <div class='info-box'>
#         Công cụ tạo biểu đồ trực quan hiện đại giúp bạn nhanh chóng biến dữ liệu thành thông tin có giá trị.
#         Hỗ trợ nhiều loại biểu đồ khác nhau: cột, đường, tròn, bong bóng, heatmap và nhiều hơn nữa.
#     </div>
    
#     <div style="text-align: center; margin: 20px 0; padding: 40px; background-color: rgba(99, 102, 241, 0.1); border-radius: 8px;">
#         <p>👨‍💻 Biểu đồ thống kê sẽ xuất hiện tại đây 📊</p>
#     </div>
#     """, unsafe_allow_html=True)
    
# with tabs[1]:
#     st.markdown("### Dự đoán xu hướng")
#     st.markdown("""
#     <div class='info-box'>
#         Sử dụng các mô hình Machine Learning tiên tiến để dự đoán xu hướng dữ liệu trong tương lai.
#         Tích hợp các thuật toán ARIMA, Prophet, và các mô hình học sâu.
#     </div>
    
#     <div style="text-align: center; margin: 20px 0; padding: 40px; background-color: rgba(99, 102, 241, 0.1); border-radius: 8px;">
#         <p>📈 Biểu đồ dự đoán sẽ xuất hiện tại đây 📉</p>
#     </div>
#     """, unsafe_allow_html=True)

# with tabs[2]:
#     st.markdown("### Phân tích văn bản")
#     st.markdown("""
#     <div class='info-box'>
#         Áp dụng kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) để phân tích văn bản, tìm hiểu cảm xúc,
#         trích xuất từ khóa và phân loại nội dung.
#     </div>
    
#     <div style="display: flex; justify-content: center; margin: 20px 0;">
#         <div style="background-color: rgba(99, 102, 241, 0.1); border-radius: 8px; padding: 20px; width: 80%;">
#             <textarea placeholder="Nhập văn bản để phân tích..." style="width: 100%; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; padding: 10px; color: white; height: 100px;"></textarea>
#             <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
#                 <button style="background: linear-gradient(45deg, #6366F1, #A855F7); border: none; color: white; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Phân tích</button>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("</div>", unsafe_allow_html=True)

# # Call to action section
# st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
# st.markdown("<h2>Sẵn sàng để bắt đầu phân tích dữ liệu của bạn?</h2>", unsafe_allow_html=True)
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.markdown("<div class='css-card' style='text-align: center;'>", unsafe_allow_html=True)
#     st.markdown("Đăng ký để nhận thông báo khi có tính năng mới")
#     email = st.text_input("Email của bạn")
#     st.button("Đăng ký ngay")
#     st.markdown("</div>", unsafe_allow_html=True)

# # Footer
# st.markdown("<footer>", unsafe_allow_html=True)
# st.markdown("© 2025 Data Processing & Visualization. All rights reserved.", unsafe_allow_html=True)
# st.markdown("</footer>", unsafe_allow_html=True)