import streamlit as st
import pandas as pd
import io
import os
import base64
from datetime import datetime

st.set_page_config(page_title="Multi-CSV Uploader", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .upload-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .file-info {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .download-btn {
        text-decoration: none;
        color: white;
        background-color: #3498db;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Create session state for uploaded files if it doesn't exist
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

def get_csv_download_link(df, filename):
    """Generate a download link for a DataFrame as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def display_file_info(file_key, file_data):
    """Display information about an uploaded file"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**{file_data['name']}**")
        st.write(f"Rows: {file_data['df'].shape[0]} | Columns: {file_data['df'].shape[1]}")
    
    with col2:
        if st.button("Preview", key=f"preview_{file_key}"):
            st.session_state.preview_file = file_key
    
    with col3:
        download_link = get_csv_download_link(file_data['df'], file_data['name'])
        st.markdown(f'<a href="{download_link}" download="{file_data["name"]}" class="download-btn">Download</a>', unsafe_allow_html=True)
        if st.button("Remove", key=f"remove_{file_key}"):
            del st.session_state.uploaded_files[file_key]
            st.experimental_rerun()

st.markdown("<h1>Multiple CSV File Uploader</h1>", unsafe_allow_html=True)

# File uploader section
st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
st.markdown("<h2>Upload your CSV files</h2>", unsafe_allow_html=True)
st.markdown("Upload one or more CSV files to analyze and process them.")

uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        # Create a unique key for this file
        file_key = f"{file.name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Only process if this is a new file
        if file_key not in st.session_state.uploaded_files:
            try:
                # Read the file
                df = pd.read_csv(file)
                
                # Store the file data in session state
                st.session_state.uploaded_files[file_key] = {
                    'name': file.name,
                    'df': df,
                    'size': file.size,
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success(f"Successfully uploaded: {file.name}")
            except Exception as e:
                st.error(f"Error uploading {file.name}: {str(e)}")

# Add options for file encoding and separator
st.markdown("<h3>Upload Options</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    encoding = st.selectbox("File Encoding", ["utf-8", "latin-1", "iso-8859-1", "cp1252"])
with col2:
    separator = st.selectbox("Separator", [",", ";", "\\t", "|"], format_func=lambda x: "Tab" if x == "\\t" else x)
with col3:
    header = st.checkbox("First row as header", value=True)

st.markdown("</div>", unsafe_allow_html=True)

# Manage uploaded files section
if st.session_state.uploaded_files:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("<h2>Uploaded Files</h2>", unsafe_allow_html=True)
    
    for file_key, file_data in list(st.session_state.uploaded_files.items()):
        st.markdown("<div class='file-info'>", unsafe_allow_html=True)
        display_file_info(file_key, file_data)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Preview selected file
    if 'preview_file' in st.session_state and st.session_state.preview_file in st.session_state.uploaded_files:
        preview_data = st.session_state.uploaded_files[st.session_state.preview_file]
        st.markdown(f"<h3>Preview: {preview_data['name']}</h3>", unsafe_allow_html=True)
        
        # Show basic statistics
        st.markdown("<h4>Basic Statistics</h4>", unsafe_allow_html=True)
        numeric_cols = preview_data['df'].select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            st.write(preview_data['df'][numeric_cols].describe())
        
        # Show data types
        st.markdown("<h4>Column Data Types</h4>", unsafe_allow_html=True)
        dtypes = pd.DataFrame(preview_data['df'].dtypes, columns=['Data Type'])
        dtypes.index.name = 'Column'
        st.write(dtypes)
        
        # Show actual data
        st.markdown("<h4>Data Preview</h4>", unsafe_allow_html=True)
        st.dataframe(preview_data['df'].head(10))
        
        # Show missing values
        st.markdown("<h4>Missing Values</h4>", unsafe_allow_html=True)
        missing = pd.DataFrame(preview_data['df'].isnull().sum(), columns=['Missing Values'])
        missing['Percentage'] = round(missing['Missing Values'] / len(preview_data['df']) * 100, 2)
        missing.index.name = 'Column'
        st.write(missing)
    
    # Process all files section
    st.markdown("<h3>Process All Files</h3>", unsafe_allow_html=True)
    if st.button("Merge All Files"):
        try:
            # Merge all dataframes
            merged_df = pd.concat([file_data['df'] for file_data in st.session_state.uploaded_files.values()], ignore_index=True)
            st.success(f"Successfully merged {len(st.session_state.uploaded_files)} files into one dataset with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
            
            # Show preview of merged data
            st.markdown("<h4>Merged Data Preview</h4>", unsafe_allow_html=True)
            st.dataframe(merged_df.head(10))
            
            # Download link for merged data
            merged_download_link = get_csv_download_link(merged_df, "merged_data.csv")
            st.markdown(f'<a href="{merged_download_link}" download="merged_data.csv" class="download-btn">Download Merged Data</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error merging files: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No files uploaded yet. Please upload at least one CSV file.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)