import streamlit as st
import pandas as pd
import os
import json
import tempfile
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
import numpy as np



# Hàm chính
def main():

    if "home" not in st.session_state:
        st.session_state.home = False

    home_page = st.Page("Home/home.py", title= "Trang chủ", icon = ":material/house:")
    dashboard = st.Page("reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)
    bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
    alerts = st.Page("reports/alerts.py", title="System alerts", icon=":material/notification_important:")

    search = st.Page("Tools/search.py", title="Search", icon=":material/search:")
    history = st.Page("Tools/history.py", title="History", icon=":material/history:")

    if st.session_state.home:
        pg = st.navigation(
            {
                #"Account": [logout_page],
                "Reports": [dashboard, bugs, alerts],
                "Tools": [search, history],
            }
    )
    else:
        pg = st.navigation([home_page])

    pg.run()

if __name__ == "__main__":
    main()
