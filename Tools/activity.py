import streamlit as st
import pandas as pd
import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# Thêm CSS tùy chỉnh
st.markdown("""
<style>
    /* Card container */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(87, 44, 44, 0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metrics grid */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Charts container */
    .chart-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    .section-header {
        color: #1f1f1f;
        font-size: 24px;
        font-weight: 600;
        margin: 30px 0 20px 0;
    }
    
    /* Status indicators */
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state nếu chưa có
if 'visitor_count' not in st.session_state:
    st.session_state.visitor_count = 0
if 'visits_data' not in st.session_state:
    st.session_state.visits_data = []

# Tăng số lượng visitor và lưu thông tin
def log_visit():
    st.session_state.visitor_count += 1
    
    # Lấy thông tin location thông qua IP
    try:
        ip_info = requests.get('https://ipapi.co/json/').json()
        location = f"{ip_info.get('city', 'Unknown')}, {ip_info.get('country_name', 'Unknown')}"
    except:
        location = "Unknown"
    
    # Lưu thông tin visit
    visit_info = {
        'timestamp': datetime.datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
        'location': location,
        'status': 'Active'
    }
    st.session_state.visits_data.append(visit_info)

# Gọi hàm log_visit khi page được load
log_visit()

# Header
st.markdown("<h1 style='text-align: center;'>Activity Monitor</h1>", unsafe_allow_html=True)

# Metrics Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Total Visitors",
        value=st.session_state.visitor_count,
        delta="Today"
    )
with col2:
    active_users = len([x for x in st.session_state.visits_data if x['status'] == 'Active'])
    st.metric(
        label="Active Users",
        value=active_users,
        delta="Now"
    )
with col3:
    st.metric(
        label="Locations",
        value=len(set([x['location'] for x in st.session_state.visits_data])),
        delta="Unique"
    )

# Visitors Over Time Chart
st.markdown("<h2 class='section-header'>Visitors Over Time</h2>", unsafe_allow_html=True)
df_visits = pd.DataFrame(st.session_state.visits_data)
if not df_visits.empty:
    df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
    fig_timeline = px.line(
        df_visits.groupby(df_visits['timestamp'].dt.hour).size().reset_index(),
        x='timestamp',
        y=0,
        title='Visitors by Hour',
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# Location Map
st.markdown("<h2 class='section-header'>Visitor Locations</h2>", unsafe_allow_html=True)
if not df_visits.empty:
    location_counts = df_visits['location'].value_counts()
    fig_locations = go.Figure(data=[go.Bar(
        x=location_counts.index,
        y=location_counts.values,
    )])
    fig_locations.update_layout(
        title='Visitors by Location',
        xaxis_title='Location',
        yaxis_title='Number of Visitors'
    )
    st.plotly_chart(fig_locations, use_container_width=True)

# Recent Activity Table
st.markdown("<h2 class='section-header'>Recent Activity</h2>", unsafe_allow_html=True)
if not df_visits.empty:
    st.dataframe(
        df_visits.tail(10).style.apply(
            lambda x: ['background-color: #e8f4ea' if v == 'Active' else '' for v in x],
            subset=['status']
        )
    ) 