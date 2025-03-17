import streamlit as st
import pandas as pd
import os
import json
import tempfile
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
import time
import numpy as np
import duckdb as db
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import io

def designPage():
    st.set_page_config(
    page_title="Ứng dụng Xử Lý Dữ Liệu",
    page_icon="🚀",
    layout="wide",  # "centered" hoặc "wide"
    initial_sidebar_state="expanded",  # "expanded" hoặc "collapsed"
    menu_items={
         'Get Help': 'https://example.com/help',
         'Report a bug': 'https://example.com/bug',
         'About': 'Ứng dụng xử lý dữ liệu được thiết kế bởi Khanh Duy Bui'
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
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(-45deg, #FF5733, #33C9FF, #9B59B6, #E74C3C);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 8s ease infinite;
    }
    </style>
    <p class="animated-gradient-title"> Phần mềm xử lý dữ liệu - Khanh Duy Bui</p>
    """, unsafe_allow_html=True)

def uploadData():
  st.subheader("Tải lên bộ dữ liệu của bạn")
  uploaded_file = st.file_uploader("Chọn file CSV để tải lên", type=["csv"])
  return uploaded_file

# Tải lên và hiển thị dữ liệu từ file CSV
def displayData():
  if uploaded_file is not None:
    try:
      encodings = ['latin1', 'ISO-8859-1', 'cp1252', 'windows-1252']
      for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except Exception as e:
            continue
      st.subheader("Thông tin bộ dữ liệu")
      st.write(f"Số hàng: {df.shape[0]}")
      st.write(f"Số cột: {df.shape[1]}")
      st.subheader("Dữ liệu đã tải lên")
      st.dataframe(df)
      st.subheader("Thống kê mô tả")
      st.write(df.describe())
      csv = df.to_csv(index=False)
      st.download_button(
        label="Tải xuống dữ liệu dưới dạng CSV",
        data=csv,
        file_name="data_processed.csv",
        mime="text/csv",
        )
    except Exception as e:
      st.error(f"Đã xảy ra lỗi khi xử lý file CSV: {e}")
      
    return df
    
        
def filterData(file_data):
    if file_data is not None:
      st.header("Lọc dữ liệu")
      filter_columns = st.multiselect("Chọn cột để lọc", file_data.columns)
        
      filtered_df = file_data.copy()
      for column in filter_columns:
        unique_values = filtered_df[column].unique()
        selected_value = st.selectbox(f"Chọn giá trị cho {column}", unique_values, key=column)
        filtered_df = filtered_df[filtered_df[column] == selected_value]
        
      st.write("### Dữ liệu sau khi lọc:")
      st.dataframe(filtered_df)
  

def visualizeData(file_data):
  if file_data is not None:
    st.header("Trực quan hóa dữ liệu")
    numeric_cols = file_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = file_data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = file_data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Chart selection tabs
    chart_type = st.selectbox(
        "Chọn loại biểu đồ:",
        ["Biểu đồ tần suất/phân phối (Histogram/Distribution)", "Biểu đồ phân tán (Scatter Plot)", "Biểu đồ thanh (Bar Chart)", "Biểu đồ đường (Line Chart)", "Biểu đồ tương quan (Correlation Heatmap)", "Biểu đồ hộp (Box Plot)", "Biểu đồ tròn (Pie Chart)"]
    )
    
    if chart_type == "Biểu đồ tần suất/phân phối (Histogram/Distribution)":
        if numeric_cols:
            col_to_plot = st.selectbox("Lựa chọn cột để trực quan:", numeric_cols)
            bins = st.slider("Number of bins:", 5, 100, 30)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Tần suất của cột {col_to_plot}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(file_data[col_to_plot].dropna(), bins=bins, alpha=0.7)
                ax.set_xlabel(col_to_plot)
                ax.set_ylabel("Tần suất")
                st.pyplot(fig)
            
            with col2:
                st.subheader(f"Phân phối của cột {col_to_plot}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(file_data[col_to_plot].dropna(), fill=True, ax=ax)
                ax.set_xlabel(col_to_plot)
                ax.set_ylabel("Phân phối")
                st.pyplot(fig)
                
            # Display descriptive statistics
            st.subheader(f"Thống kê mô tả cho {col_to_plot}")
            stats = file_data[col_to_plot].describe()
            st.dataframe(stats)
        else:
            st.warning("Không có cột số nào được tìm thấy trong bộ dữ liệu để trực quan hóa biểu đồ")
    
    elif chart_type == "Biểu đồ phân tán (Scatter Plot)":
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("Lựa chọn trục X:", numeric_cols)
            
            with col2:
                y_col = st.selectbox("Lựa chọn trục Y:", 
                                    [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols)
            
            color_col = st.selectbox("Lựa chọn Cột màu (Tùy chọn)::", 
                                    ["None"] + categorical_cols + numeric_cols)
            
            if color_col == "None":
                fig = px.scatter(file_data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(file_data, x=x_col, y=y_col, color=color_col, 
                                title=f"{y_col} vs {x_col} (colored by {color_col})")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            correlation = file_data[[x_col, y_col]].corr().iloc[0, 1]
            st.write(f"**Hệ số tương quan:** {correlation:.4f}")
        else:
            st.warning("Biểu đồ phân tán yêu cầu ít nhất 2 cột số. Bộ dữ liệu của bạn không đáp ứng được yêu cầu!")
    
    elif chart_type == "Biểu đồ thanh (Bar Chart)":
        if categorical_cols:
            x_col = st.selectbox("Chọn cột phân loại : (cột X):", categorical_cols)
            
            # Option for aggregation
            agg_options = ["Count", "Sum", "Mean", "Median", "Min", "Max"]
            agg_type = st.selectbox("Lựa chọn hình thức:", agg_options)
            
            if agg_type == "Count":
                # Just count the occurrences
                data = file_data[x_col].value_counts().reset_index()
                data.columns = [x_col, 'Count']
                y_val = 'Count'
                title = f"Count of {x_col}"
            else:
                # Need a numeric column to aggregate
                if numeric_cols:
                    y_col = st.selectbox("Chọn cột thực hiện hình thức (cột Y):", numeric_cols)
                    agg_func = agg_type.lower()
                    
                    if agg_func == "median":
                        data = file_data.groupby(x_col)[y_col].median().reset_index()
                    else:
                        data = file_data.groupby(x_col).agg({y_col: agg_func}).reset_index()
                    
                    y_val = y_col
                    title = f"{agg_type} of {y_col} by {x_col}"
                else:
                    st.warning("Biểu đồ thanh yêu cầu các cột số không có sẵn trong bộ dữ liệu của bạn!")
                    return
            
            # Sort options
            sort_by = st.selectbox("Sắp xếp theo:", ["Alphabet", "Giá trị"])
            if sort_by == "Giá trị":
                data = data.sort_values(y_val)
            else:
                data = data.sort_values(x_col)
            
            # Limit number of categories if too many
            max_categories = st.slider("Số lượng danh mục tối đa để hiển thị:", 5, 50, 20)
            if len(data) > max_categories:
                data = data.iloc[-max_categories:]
            
            fig = px.bar(data, x=x_col, y=y_val, title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Biểu đồ thanh yêu cầu các cột số không có sẵn trong bộ dữ liệu của bạn!")
    
    elif chart_type == "Biểu đồ đường (Line Chart)":
        if datetime_cols:
            date_col = st.selectbox("Lựa chọn cột thời gian (cột X):", datetime_cols)
            y_col = st.selectbox("Lựa chọn cột số (cột Y):", numeric_cols if numeric_cols else ["Không có cột số!"])
            
            if numeric_cols:
                group_by = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                
                if group_by == "None":
                    fig = px.line(file_data.sort_values(date_col), x=date_col, y=y_col, title=f"{y_col} over Time")
                else:
                    fig = px.line(file_data.sort_values(date_col), x=date_col, y=y_col, color=group_by, title=f"{y_col} over Time by {group_by}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Biểu đồ đường yêu cầu các cột số hoặc DateTime không có sẵn trong bộ dữ liệu của bạn!")
        elif numeric_cols:
            st.info("Không tìm thấy cột DateTime. Bạn vẫn có thể tạo biểu đồ dòng với trục X.")
            
            x_options = ["Chỉ số"] + numeric_cols
            x_col = st.selectbox("Chọn cột X:", x_options)
            y_col = st.selectbox("Chọn cột Y:", [col for col in numeric_cols if col != x_col] if x_col in numeric_cols else numeric_cols)
            
            if x_col == "Chỉ số":
                fig = px.line(file_data, y=y_col, title=f"{y_col} by Index")
            else:
                fig = px.line(file_data.sort_values(x_col), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Biểu đồ đường yêu cầu các cột số hoặc DateTime không có sẵn trong bộ dữ liệu của bạn!")
    
    elif chart_type == "Biểu đồ tương quan (Correlation Heatmap)":
        if len(numeric_cols) > 1:
            st.subheader("Ma trận tương quan")
            
            # Allow user to select columns
            selected_cols = st.multiselect(
                "Chọn các cột để phân tích tương quan (mặc định: Tất cả các cột số):",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))]  # Default to first 8 columns
            )
            
            if selected_cols:
                corr_matrix = file_data[selected_cols].corr()
                
                # Plot with Plotly for interactivity
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Ma trận tương quan"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to display the correlation table
                if st.checkbox("Hiển thị các giá trị tương quan dưới dạng bảng"):
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
            else:
                st.warning("Vui lòng chọn ít nhất 1 cột để phân tích tương quan")
        else:
            st.warning("Biểu đồ tương quan yêu cầu ít nhất 2 cột số. Bộ dữ liệu của bạn không đáp ứng được!")
    
    elif chart_type == "Biểu đồ hộp (Box Plot)":
        if numeric_cols:
            numeric_col = st.selectbox("Lựa chọn cột dữ liệu số:", numeric_cols)
            category_col = st.selectbox("Chọn danh mục để nhóm (tùy chọn):", ["None"] + categorical_cols)
            
            if category_col == "None":
                fig = px.box(file_data, y=numeric_col, title=f"Box Plot of {numeric_col}")
            else:
                # Limit categories if too many
                cat_counts = file_data[category_col].value_counts()
                if len(cat_counts) > 15:
                    top_cats = cat_counts.nlargest(15).index.tolist()
                    filtered_df = file_data[file_data[category_col].isin(top_cats)]
                    fig = px.box(filtered_df, x=category_col, y=numeric_col, 
                                title=f"Box Plot of {numeric_col} by {category_col} (top 15 categories)")
                    st.info("Chỉ hiển thị 15 hàng đầu do số lượng lớn các danh mục.")
                else:
                    fig = px.box(file_data, x=category_col, y=numeric_col,
                                title=f"Box Plot of {numeric_col} by {category_col}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show descriptive statistics by group
            if category_col != "None":
                st.subheader(f"Statistics of {numeric_col} by {category_col}")
                st.dataframe(file_data.groupby(category_col)[numeric_col].describe())
        else:
            st.warning("Biểu đồ hộp yêu cầu các cột số không có sẵn trong bộ dữ liệu của bạn. Vui lòng kiểm tra lại!")
    
    elif chart_type == "Biểu đồ tròn (Pie Chart)":
        if categorical_cols:
            cat_col = st.selectbox("Chọn cột phân loại:", categorical_cols)
            
            # Option to count or sum
            agg_method = st.selectbox("Chọn phương pháp phân loại:", ["Count", "Sum of numeric column"])
            
            if agg_method == "Count":
                # Get value counts and prepare data
                counts = file_data[cat_col].value_counts()
                
                # Limit slices if too many categories
                max_slices = st.slider("Maximum number of slices:", 3, 15, 8)
                if len(counts) > max_slices:
                    # Keep top N-1 categories and group the rest as "Other"
                    top_counts = counts.nlargest(max_slices - 1)
                    other_count = counts.sum() - top_counts.sum()
                    
                    data = pd.DataFrame({
                        cat_col: list(top_counts.index) + ['Other'],
                        'Count': list(top_counts.values) + [other_count]
                    })
                    
                    fig = px.pie(data, names=cat_col, values='Count', 
                                title=f"Distribution of {cat_col} (top {max_slices-1} categories + Other)")
                else:
                    # Use all categories if not too many
                    data = pd.DataFrame({
                        cat_col: counts.index,
                        'Count': counts.values
                    })
                    fig = px.pie(data, names=cat_col, values='Count', 
                                title=f"Distribution of {cat_col}")
            else:
                if numeric_cols:
                    num_col = st.selectbox("Chọn cột để tính thành tổng::", numeric_cols)
                    
                    # Get sum by category
                    sums = file_data.groupby(cat_col)[num_col].sum().reset_index()
                    
                    # Limit slices if too many
                    max_slices = st.slider("Maximum number of slices:", 3, 15, 8)
                    if len(sums) > max_slices:
                        # Sort and keep top N-1
                        sums = sums.sort_values(num_col, ascending=False)
                        top_sums = sums.head(max_slices - 1)
                        
                        # Create "Other" category
                        other_sum = sums[num_col][max_slices-1:].sum()
                        other_row = pd.DataFrame({cat_col: ['Other'], num_col: [other_sum]})
                        
                        # Combine
                        data = pd.concat([top_sums, other_row])
                        fig = px.pie(data, names=cat_col, values=num_col, 
                                    title=f"Sum of {num_col} by {cat_col} (top {max_slices-1} categories + Other)")
                    else:
                        fig = px.pie(sums, names=cat_col, values=num_col, 
                                    title=f"Sum of {num_col} by {cat_col}")
                else:
                    st.warning("This aggregation method requires numeric columns which are not available in your dataset.")
                    return
            
            # Display the pie chart
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Pie charts require categorical columns which are not available in your dataset.")

def clean_data(file_data):
    """
    Process and clean the dataframe:
    - Extract numbers from mixed text-numeric columns 
    - Convert percentage strings to numeric values
    - Handle currency values
    - Convert date-like strings to datetime
    """
    if file_data is not None:
        processed_df = file_data.copy()
    
    # Track changes made for reporting
        cleaning_report = []
    
    # Process each column
        for col in processed_df.columns:
        # Skip if column is already numeric or datetime
            if pd.api.types.is_numeric_dtype(processed_df[col]) or pd.api.types.is_datetime64_dtype(processed_df[col]):
                continue
            
        # Check if column has string values
            if processed_df[col].dtype == object:
            # Sample values for pattern detection (avoid NaN)
                sample_values = processed_df[col].dropna().sample(min(5, len(processed_df[col].dropna()))).tolist()
            
            # Skip if no sample values
                if not sample_values:
                    continue
                
            # Check for currency patterns (e.g., "$1,000", "1000 USD", "€2,000")
                currency_pattern = r'(?:[$€£¥]\s*[\d,.]+)|(?:[\d,.]+\s*[$€£¥])|(?:[\d,.]+\s*(?:USD|EUR|GBP|JPY|CAD|AUD))'
            
            # Check for percentage patterns (e.g., "10%", "10.5 %")
                percentage_pattern = r'[\d,.]+\s*%'
            
            # Check for mixed numeric patterns (e.g., "900 units", "1,200 items")
                mixed_numeric_pattern = r'[\d,.]+\s*\w+'
            
            # Function to extract numeric value from string
                def extract_numeric(val):
                    if pd.isna(val) or not isinstance(val, str):
                        return val
                # Extract all digits, commas, dots, minus signs
                    matches = re.findall(r'-?[\d,.]+', str(val))
                    if matches:
                    # Replace commas with empty and convert to float
                        return float(matches[0].replace(',', ''))
                    return val
            
            # Check patterns in sample values
                has_currency = any(re.search(currency_pattern, str(val)) for val in sample_values)
                has_percentage = any(re.search(percentage_pattern, str(val)) for val in sample_values)
                has_mixed_numeric = any(re.search(mixed_numeric_pattern, str(val)) for val in sample_values)
            
            # Apply transformations based on detected patterns
                if has_currency:
                    original_col = processed_df[col].copy()
                    processed_df[col] = processed_df[col].apply(extract_numeric)
                
                # Report changes if successful
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        cleaning_report.append(f"Converted currency column '{col}' to numeric values")
                    
                    # Store currency unit if consistent
                        currency_units = set()
                        for val in original_col.dropna():
                            if isinstance(val, str):
                            # Extract currency symbol or code
                                currency_match = re.search(r'([$€£¥]|USD|EUR|GBP|JPY|CAD|AUD)', str(val))
                                if currency_match:
                                    currency_units.add(currency_match.group(0))
                    
                        if len(currency_units) == 1:
                            unit = list(currency_units)[0]
                        # Create metadata for the column
                            processed_df[f"{col}_unit"] = unit
                            cleaning_report.append(f"Stored currency unit '{unit}' for column '{col}'")
                
                elif has_percentage:
                    processed_df[col] = processed_df[col].apply(lambda x: extract_numeric(x) / 100 if isinstance(x, str) and '%' in x else x)
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        cleaning_report.append(f"Converted percentage column '{col}' to decimal values")
                        processed_df[f"{col}_is_percentage"] = True
                
                elif has_mixed_numeric:
                    original_col = processed_df[col].copy()
                    processed_df[col] = processed_df[col].apply(extract_numeric)
                
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        cleaning_report.append(f"Extracted numeric values from mixed column '{col}'")
                    
                    # Try to extract consistent units
                        units = set()
                        for val in original_col.dropna():
                            if isinstance(val, str):
                            # Extract potential unit after number
                                unit_match = re.search(r'[\d,.]+\s*(\w+)', str(val))
                                if unit_match:
                                    units.add(unit_match.group(1))
                    
                        if len(units) == 1:
                            unit = list(units)[0]
                            processed_df[f"{col}_unit"] = unit
                            cleaning_report.append(f"Stored unit '{unit}' for column '{col}'")

            # Try to convert to datetime if not already processed
                if processed_df[col].dtype == object:
                    try:
                    # Check if at least 90% of non-NA values can be parsed as dates
                        non_na_count = processed_df[col].dropna().shape[0]
                        if non_na_count > 0:
                            date_conversion = pd.to_datetime(processed_df[col], errors='coerce')
                            if date_conversion.notna().sum() / non_na_count >= 0.9:
                                processed_df[col] = date_conversion
                                cleaning_report.append(f"Converted column '{col}' to datetime format")
                    except:
                        pass
    
        return processed_df, cleaning_report

def processingData(file_data):
  cleaning_method = st.radio(
        "Lựa chọn phương pháp làm sạch dữ liệu:",
        ["Tự động (Beta)", "Lựa chọn cột thủ công"]
    )
    
  if cleaning_method == "Tự động (Beta)" and file_data is not None:
    processed_df, cleaning_report = clean_data(file_data)
    if cleaning_report:
      st.success("Làm sạch thành công! Hãy kiểm tra lại bộ dữ liệu. Nếu chưa phù hợp hãy làm sạch thủ công ")
      for item in cleaning_report:
        st.write(f"✓ {item}")
    else:
      st.info("Bộ dữ liệu không cần làm sạch!")
        
    if st.checkbox("So sánh dữ liệu trước và sau khi làm sạch"):
      col1, col2 = st.columns(2)
      with col1:
        st.subheader("Trước")
        st.dataframe(file_data.head(10))
      with col2:
        st.subheader("Sau")
        st.dataframe(processed_df.head(10))
                
        return processed_df
    else:
        return processed_df
  elif cleaning_method == "Lựa chọn cột thủ công" and file_data is not None:
    st.markdown("#### Lựa chọn cột cần xử lý:")
    col_changes = {}
    for col in file_data.columns:
      if pd.api.types.is_numeric_dtype(file_data[col]) or pd.api.types.is_datetime64_dtype(file_data[col]):
        continue
      sample_values = file_data[col].dropna().sample(min(3, len(file_data[col].dropna()))).tolist()
      if sample_values and isinstance(sample_values[0], str):
        st.markdown(f"**{col}**  \nDữ liệu mẫu: `{', '.join(str(x) for x in sample_values[:3])}`")
        options = ["Giữ nguyên"]
        if any(re.search(r'[\d,.]+', str(val)) for val in sample_values):
          options.append("Trích xuất giá trị số")
        if any(re.search(r'%', str(val)) for val in sample_values):
          options.append("Chuyển đổi tỷ lệ phần trăm thành số thập phân")
        if any(re.search(r'[$€£¥]|USD|EUR', str(val)) for val in sample_values):
          options.append("Trích xuất các giá trị tiền tệ")
        try:
          pd.to_datetime(file_data[col].iloc[0], errors='raise')
          options.append("Chuyển đổi định dạng ngày/giờ")
        except:
          pass
                
        choice = st.selectbox(f"Xử lý cột '{col}' :", options, key=f"process_{col}")
        if choice != "Giữ nguyên":
          col_changes[col] = choice
                    
    if col_changes:
      processed_df = file_data.copy()
      for col, change_type in col_changes.items():
        if change_type == "Trích xuất giá trị số":
          processed_df[col] = processed_df[col].apply(
              lambda x: float(re.findall(r'-?[\d,.]+', str(x))[0].replace(',', '')) 
              if isinstance(x, str) and re.findall(r'-?[\d,.]+', str(x)) else x
          )
          st.success(f"Trích xuất giá trị từ cột '{col}'")
        elif change_type == "Chuyển đổi tỷ lệ phần trăm thành số thập phân":
          processed_df[col] = processed_df[col].apply(
            lambda x: float(re.findall(r'-?[\d,.]+', str(x))[0].replace(',', '')) / 100 
            if isinstance(x, str) and '%' in x and re.findall(r'-?[\d,.]+', str(x)) else x
          )
          st.success(f"Chuyển đổi giá trị phần trăm của cột '{col}' thành giá trị thập phân")
        elif change_type == "Trích xuất các giá trị tiền tệ":
          processed_df[col] = processed_df[col].apply(
            lambda x: float(re.findall(r'-?[\d,.]+', str(x))[0].replace(',', '')) 
            if isinstance(x, str) and re.findall(r'-?[\d,.]+', str(x)) else x
          )
          st.success(f"Chuyển đổi định dạng tiền tệ '{col}'")
        elif change_type == "Chuyển đổi định dạng ngày/giờ":
          processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
          st.success(f"Chuyển đổi cột '{col}' thành định dạng ngày giờ")
            
      if st.checkbox("So sánh dữ liệu trước và sau khi làm sạch"):
        col1, col2 = st.columns(2)
        with col1:
          st.subheader("Trước")
          st.dataframe(file_data.head(10))
        with col2:
          st.subheader("Sau")
          st.dataframe(processed_df.head(10))
                    
      return processed_df
    else:
      st.info("Không cột nào được lựa chọn để xử lý. Sử dụng dữ liệu ban đầu!!!")
      return file_data
    

designPage()   
designTitle()
uploaded_file = uploadData()
df = displayData()
processed_data = processingData(df)
filterData(processed_data)
visualizeData(processed_data)

